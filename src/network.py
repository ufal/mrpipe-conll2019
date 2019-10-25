#!/usr/bin/env python3
import collections
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from mapping import Mapping
import mtool_evaluate
from sentence import Sentence

NODE_NO, NODE_PARENT, NODE_CHILD = range(3)

def get_layers(s):
    # Shortest paths from start, compute anchors
    paths = [0] * s.n_tokens() + [None] * s.n_nodes()
    anchors = [{i} if i < s.n_tokens() else set() for i in range(s.n_tokens() + s.n_nodes())]
    queue = collections.deque(range(s.n_tokens()))

    while queue:
        n = queue.popleft()
        for e in s.parents[n]:
            p = s.factors[s.EDGE_PARENTS][e]
            anchors[p] |= anchors[n]
            if not paths[p]:
                paths[p] = paths[n] + 1
                queue.append(p)

    # Compute a sequence of created nodes
    layers = []

    nodes = set(range(s.n_tokens()))
    while len(nodes) < s.n_nodes() + s.n_tokens():
        new_nodes, new_layer = set(), []
        for n in nodes:
            best, best_e = None, None
            for e in s.parents[n]:
                p = s.factors[s.EDGE_PARENTS][e]
                if p not in nodes and p not in new_nodes:
                    if not best or len(anchors[p]) < len(anchors[best]) or sorted(anchors[p]) < sorted(anchors[best]):
                        best, best_e = p, e
            for e in s.children[n]:
                p = s.factors[s.EDGE_CHILDREN][e]
                if paths[p]: continue
                if p not in nodes and p not in new_nodes:
                    if not best or len(anchors[p]) < len(anchors[best]) or sorted(anchors[p]) < sorted(anchors[best]):
                        best, best_e = p, e
            if best:
                new_nodes.add(best)
                new_layer.append((n, best, best_e))

        if not new_nodes: break
        nodes |= new_nodes
        layers.append(new_layer)

    if len(nodes) != s.n_nodes() + s.n_tokens():
        raise ValueError("Cannot get layers for sentence {}".format(s.id))

    return layers


class Network:
    def __init__(self, train, args):
        # Create required layers
        self.layers = tf.keras.Model()

        # Encoder
        self.layers.encoder_we = tf.keras.layers.Embedding(len(train.mappings[Sentence.TOKENS].strings), args.we_dim, mask_zero=True)
        self.layers.encoder_cle = tf.keras.layers.Embedding(len(train.mappings[Sentence.TOKENS].alphabet), args.cle_dim, mask_zero=True)
        self.layers.encoder_cle_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(args.cle_dim), merge_mode="concat")
        self.layers.encoder_token_values = [
            tf.keras.layers.Embedding(len(train.mappings[Sentence.TOKEN_VALUES][prop].strings), 128, mask_zero=True)
            for prop in train.token_properties
        ]
        self.layers.encoder_rnns = [
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(args.encoder_dim, return_sequences=True), merge_mode="sum")
            for _ in range(args.encoder_rnn_layers)
        ]
        self.layers.encoder_node_values = [
            tf.keras.layers.Embedding(len(train.mappings[Sentence.NODE_VALUES][prop].strings), args.encoder_dim)
            for prop in train.node_properties
        ]
        self.edge_values = [len(train.mappings[Sentence.EDGE_VALUES][prop].strings) for prop in train.edge_properties]
        self.layers.encoder_edge_values = [
            tf.keras.layers.Embedding(2 * edge_values, args.encoder_dim)
            for edge_values in self.edge_values
        ]
        self.layers.encoder_token_transfer = [
            tf.keras.layers.Dense(args.encoder_dim, activation=tf.nn.tanh)
            for _ in range(args.decoder_iterations)]

        # Decoder
        self.layers.decoder_node_operation = [
            tf.keras.layers.Dense(3, activation=tf.nn.softmax) for _ in range(args.decoder_iterations)]
        self.layers.decoder_node_values = [
            [tf.keras.layers.Dense(len(train.mappings[Sentence.NODE_VALUES][prop].strings), activation=tf.nn.softmax) for prop in train.node_properties]
            for _ in range(args.decoder_iterations)]
#         self.layers.decoder_edge_values = [
#             [tf.keras.layers.Dense(len(train.mappings[Sentence.EDGE_VALUES][prop].strings), activation=tf.nn.softmax) for prop in train.edge_properties]
#             for _ in range(args.decoder_iterations)]
        self.layers.decoder_edge_parents = [
            tf.keras.layers.Dense(args.edge_dim)
            for _ in range(args.decoder_iterations)]
        self.layers.decoder_edge_children = [
            tf.keras.layers.Dense(args.edge_dim)
            for _ in range(args.decoder_iterations)]
        self.layers.decoder_edge_arc = [
            tf.keras.layers.Dense(2, activation=tf.nn.softmax)
            for _ in range(args.decoder_iterations)]
        if args.highway:
            self.layers.decoder_edge_highway = [
                tf.keras.layers.Dense(args.edge_dim, activation=tf.nn.tanh)
                for _ in range(args.decoder_iterations)]
        self.layers.decoder_deprel_parents = [
            tf.keras.layers.Dense(args.deprel_dim)
            for _ in range(args.decoder_iterations)]
        self.layers.decoder_deprel_children = [
            tf.keras.layers.Dense(args.deprel_dim)
            for _ in range(args.decoder_iterations)]
        self.layers.decoder_deprel_values = [
            [tf.keras.layers.Dense(len(train.mappings[Sentence.EDGE_VALUES][prop].strings), activation=tf.nn.softmax) for prop in train.edge_properties]
            for _ in range(args.decoder_iterations)]
        if args.highway:
            self.layers.decoder_deprel_highway = [
                tf.keras.layers.Dense(args.edge_dim, activation=tf.nn.tanh)
                for _ in range(args.decoder_iterations)]
        self.layers.decoder_tops = tf.keras.layers.Dense(2, activation=tf.nn.softmax)

        # Generic helpers
        self.layers.dropout = tf.keras.layers.Dropout(args.dropout)
        self.layers.concat = tf.keras.layers.Concatenate()
        self.layers.sum = tf.keras.layers.Add()
        self.word_dropout = args.word_dropout
        self.optimizer_step_sentences = len(train.sentence_lens) / ((len(train.sentence_lens) - 1) // (args.batch_size * args.batch_aggregation) + 1)

        if args.predict: return

        # Create optimizer, losses, summary writer and accuracies
        self.optimizer = tfa.optimizers.LazyAdam(beta_2=args.beta_2)
        self.loss_sce = tf.keras.losses.SparseCategoricalCrossentropy()
        self.writer = tf.summary.create_file_writer(args.logdir, flush_millis=10*1000)
        self.accuracies = dict([
            ("node/ops", tf.keras.metrics.SparseCategoricalAccuracy())] + [
            ("node/" + prop, tf.keras.metrics.SparseCategoricalAccuracy()) for prop in train.node_properties] + [
            ("edge/arc", tf.keras.metrics.SparseCategoricalAccuracy())] + [
            ("edge/" + prop, tf.keras.metrics.SparseCategoricalAccuracy()) for prop in train.edge_properties] + [
            ("edge/tops", tf.keras.metrics.SparseCategoricalAccuracy())])

    def load_weights(self, path):
        self.layers.load_weights(path)

    def save_weights(self, path):
        self.layers.save_weights(path)

    def _encoder(self, token_ids, token_charseqs, token_charseq_ids, token_values, token_additionals, training):
        if training and self.word_dropout:
            word_dropout_mask = tf.cast(tf.math.greater_equal(tf.random.uniform(token_ids.shape), self.word_dropout), tf.int32)
            unk_mask = tf.cast(tf.math.not_equal(token_ids, 0), tf.int32) * Mapping.UNK
            token_ids = word_dropout_mask * token_ids + (1 - word_dropout_mask) * unk_mask
        inputs = [self.layers.encoder_we(token_ids)]

        characters_embedded = self.layers.encoder_cle(token_charseqs)
        characters_embedded = self.layers.dropout(characters_embedded, training=training)
        charseqs = self.layers.encoder_cle_rnn(characters_embedded)
        charseqs = tf.gather(charseqs, token_charseq_ids)
        inputs.append(charseqs)

        for i, embedding in enumerate(self.layers.encoder_token_values):
            inputs.append(embedding(token_values[:, :, i]))

        inputs.extend(token_additionals)

        hidden = self.layers.concat(inputs)
        hidden = self.layers.dropout(hidden, training=training)
        for i, rnn in enumerate(self.layers.encoder_rnns):
            rnn_output = rnn(hidden)
            rnn_output = self.layers.dropout(rnn_output, training=training)
            if i == 0:
                hidden = rnn_output
            else:
                hidden = self.layers.sum([hidden, rnn_output])

        delattr(hidden, "_keras_mask")
        return hidden

    class State:
        def __init__(self, network, sentence, tokens, training):
            self.network = network
            self.sentence = sentence
            self.tokens = tokens
            self.training = training

            self.nodes = tokens
            self.n_nodes = sentence.n_tokens()
            self.nodes_token = list(range(sentence.n_tokens()))

            self.n_nodes_cached = self.n_nodes
            self.node_mapping = {i: i for i in range(self.n_nodes)}

        def add_node(self, values, predecessor, mapping=None):
            self.sentence.add_node(values)
            self.nodes_token.append(self.nodes_token[predecessor])
            if mapping is not None:
                self.node_mapping[mapping] = self.n_nodes
            self.n_nodes += 1

        def add_edge(self, parent, child, values):
            if values[0] == Mapping.ANCHOR and child >= self.sentence.n_tokens(): return
            if values[0] == Mapping.ANCHOR and parent < self.sentence.n_tokens(): return
            self.sentence.add_edge(parent, child, values)

        @staticmethod
        def recompute_nodes(nodes, states, iteration):
            nodes_token, nodes_values = [], []
            start = 0
            for state in states:
                nodes_token.extend([start + t for t in state.nodes_token[state.n_nodes_cached:]])
                nodes_values.extend(state.sentence.factors[Sentence.NODE_VALUES][state.n_nodes_cached - state.sentence.n_tokens():])
                start += state.n_nodes_cached

            new_nodes = states[0].network.layers.encoder_token_transfer[iteration](tf.gather(nodes, np.array(nodes_token, np.int32)))
            for i, encoder in enumerate(states[0].network.layers.encoder_node_values):
                new_nodes += encoder(np.array([node_values[i] for node_values in nodes_values], np.int32))

            new_nodes = states[0].network.layers.dropout(new_nodes, training=states[0].training)

            start = 0
            for state in states:
                num_nodes = state.n_nodes - state.n_nodes_cached
                if num_nodes:
                    state.nodes = tf.concat([state.nodes, new_nodes[start:start + num_nodes]], axis=0)
                start += num_nodes
                state.n_nodes_cached = state.n_nodes

        @staticmethod
        def recompute_edges(ori_nodes, states, iteration):
            node_ids, edge_values = [], []
            start = 0
            for i, state in enumerate(states):
                for n in range(ori_nodes[i], state.n_nodes):
                    for e in state.sentence.children[n]:
                        node_ids.append(start + n - ori_nodes[i])
                        edge_values.append(state.sentence.factors[Sentence.EDGE_VALUES][e])
                    for e in state.sentence.parents[n]:
                        node_ids.append(start + n - ori_nodes[i])
                        edge_values.append([a + b for a, b in zip(
                            state.network.edge_values, state.sentence.factors[Sentence.EDGE_VALUES][e])])
                start += state.n_nodes - ori_nodes[i]

            edge_embeddings = 0
            for i, encoder in enumerate(states[0].network.layers.encoder_edge_values):
                edge_embeddings += encoder(np.array([edge_values[i] for edge_values in edge_values], np.int32))

            node_values = tf.math.segment_mean(edge_embeddings, node_ids)
            max_node_id = max(node_ids)
            if max_node_id + 1 < start:
                node_values = tf.pad(node_values, [[0, start - max_node_id - 1], [0, 0]])

            start = 0
            for i, state in enumerate(states):
                if state.n_nodes == ori_nodes[i]: continue
                state.nodes += tf.pad(node_values[start:start + state.n_nodes - ori_nodes[i]],
                                      [[ori_nodes[i], 0], [0, 0]])
                start += state.n_nodes - ori_nodes[i]

    def _summary_step(self):
        return int(int(self.optimizer.iterations) * self.optimizer_step_sentences)

    def train_epoch(self, train, learning_rate, args):
        self.optimizer.learning_rate = learning_rate

        skipped = 0
        num_gradients = 0
        while not train.epoch_finished():
            batch = train.next_batch(args.batch_size)
            sentences = batch["sentences"]

            layers = []
            for sentence in sentences:
                layers.append(get_layers(sentence))

            for accuracy in self.accuracies.values():
                accuracy.reset_states()

            with tf.GradientTape() as tape:
                encoded_tokens = self._encoder(
                    token_ids=batch["batch_factors"][Sentence.TOKENS],
                    token_charseqs=batch["charseqs"],
                    token_charseq_ids=batch["charseq_ids"],
                    token_values=batch["batch_factors"][Sentence.TOKEN_VALUES],
                    token_additionals=[batch[x] for x in ["bert", "fasttext", "word2vec"] if x in batch],
                    training=True)

                states = []
                for i, sentence in enumerate(sentences):
                    states.append(self.State(self, sentence.empty_copy(), encoded_tokens[i][:sentence.n_tokens()], True))

                loss = 0
                for iteration in range(args.decoder_iterations):
                    # 1) New nodes
                    ori_nodes = [s.n_nodes for s in states]
                    sum_nodes = sum(ori_nodes)
                    nodes = tf.concat([s.nodes for s in states], axis=0)

                    target_ops = np.zeros([sum_nodes], np.int32)
                    target_node_values = np.zeros([sum_nodes, len(train.node_properties)], np.int32)
                    target_node_values_mask = np.zeros([sum_nodes, 1], np.int32)
#                     target_edge_values = np.zeros([sum_nodes, len(train.edge_properties)], np.int32)
#                     target_edge_values_mask = np.zeros([sum_nodes, 1], np.int32)

                    start = 0
                    for i in range(len(sentences)):
                        if iteration < len(layers[i]):
                            state, layer, sentence = states[i], layers[i][iteration], sentences[i]
                            for n, target, edge in layer:
                                n = state.node_mapping[n]
                                target_ops[start + n] = NODE_PARENT if sentence.factors[Sentence.EDGE_PARENTS][edge] == target else NODE_CHILD
                                target_node_values[start + n] = sentence.factors[Sentence.NODE_VALUES][target - sentence.n_tokens()]
                                target_node_values_mask[start + n] = 1
#                                 target_edge_values[start + n] = sentence.factors[Sentence.EDGE_VALUES][edge]
#                                 target_edge_values_mask[start + n] = 1 if n >= state.sentence.n_tokens() else 0
                                state.add_node(target_node_values[start + n], n, target)
                        start += states[i].n_nodes_cached
                    self.State.recompute_nodes(nodes, states, iteration)

                    predictions = self.layers.decoder_node_operation[iteration](nodes)
                    loss += self.loss_sce(target_ops, predictions)
                    self.accuracies["node/ops"](target_ops, predictions)
                    for i, prop in enumerate(train.node_properties):
                        predictions = self.layers.decoder_node_values[iteration][i](nodes)
                        loss += self.loss_sce(target_node_values[:, i], predictions, target_node_values_mask)
                        self.accuracies["node/" + prop](target_node_values[:, i], predictions, target_node_values_mask)
#                     for i, prop in enumerate(train.edge_properties):
#                         predictions = self.layers.decoder_edge_values[iteration][i](nodes)
#                         loss += self.loss_sce(target_edge_values[:, i], predictions, target_edge_values_mask)
#                         self.accuracies["edge/" + prop](target_edge_values[:, i], predictions, target_edge_values_mask)

                    # 2) Edges
                    sum_nodes = sum(s.n_nodes for s in states)
                    nodes = tf.concat([s.nodes for s in states], axis=0)
                    target_indices_a, target_indices_b = [], []
                    target_a_parent, target_a_child = [], []
                    target_deprel_parents, target_deprel_children = [], []
                    target_deprel_values = []

                    start = 0
                    for i in range(len(sentences)):
                        state, sentence = states[i], sentences[i]
                        offset = len(target_indices_a)
                        for j in range(ori_nodes[i], state.n_nodes):
                            target_indices_a.append(start + np.repeat(np.int32(j), state.n_nodes))
                            target_indices_b.append(start + np.arange(0, state.n_nodes, dtype=np.int32))
                            target_a_parent.append(np.zeros(state.n_nodes, np.float32))
                            target_a_child.append(np.zeros(state.n_nodes, np.float32))
                        for n_ori, n in state.node_mapping.items():
                            if n >= ori_nodes[i]:
                                for e in sentence.parents[n_ori]:
                                    if sentence.factors[Sentence.EDGE_VALUES][e][0] == Mapping.ROOT: continue
                                    p = sentence.factors[sentence.EDGE_PARENTS][e]
                                    if p in state.node_mapping:
                                        p = state.node_mapping[p]
                                        if not args.no_anchors or not sentence.factors[Sentence.EDGE_VALUES][e][0] == Mapping.ANCHOR:
                                            target_a_child[offset + n - ori_nodes[i]][p] = 1
                                        target_deprel_parents.append(start + p)
                                        target_deprel_children.append(start + n)
                                        target_deprel_values.append(sentence.factors[Sentence.EDGE_VALUES][e])
                                        state.add_edge(p, n, target_deprel_values[-1])
                                for e in sentence.children[n_ori]:
                                    if sentence.factors[Sentence.EDGE_VALUES][e][0] == Mapping.ROOT: continue
                                    c = sentence.factors[sentence.EDGE_CHILDREN][e]
                                    if c in state.node_mapping:
                                        c = state.node_mapping[c]
                                        if not args.no_anchors or not sentence.factors[Sentence.EDGE_VALUES][e][0] == Mapping.ANCHOR:
                                            target_a_parent[offset + n-ori_nodes[i]][c] = 1
                                        target_deprel_parents.append(start + n)
                                        target_deprel_children.append(start + c)
                                        target_deprel_values.append(sentence.factors[Sentence.EDGE_VALUES][e])
                                        state.add_edge(n, c, target_deprel_values[-1])
                        start += state.n_nodes

                    if not target_indices_a or not target_deprel_parents:
                        continue
                    target_indices_a = np.concatenate(target_indices_a, axis=0)
                    target_indices_b = np.concatenate(target_indices_b, axis=0)
                    target_a_parent = np.concatenate(target_a_parent, axis=0)
                    target_a_child = np.concatenate(target_a_child, axis=0)
                    target_deprel_parents = np.array(target_deprel_parents, np.int32)
                    target_deprel_children = np.array(target_deprel_children, np.int32)
                    target_deprel_values = np.array(target_deprel_values, np.int32)

                    # 2.1) Compute arcs
                    edge_parents = self.layers.decoder_edge_parents[iteration](nodes)
                    edge_children = self.layers.decoder_edge_children[iteration](nodes)

                    a_parent = tf.nn.tanh(self.layers.sum([tf.gather(edge_parents, target_indices_a),
                                                           tf.gather(edge_children, target_indices_b)]))
                    if args.highway:
                        a_parent += self.layers.decoder_edge_highway[iteration](a_parent)
                    a_parent = self.layers.decoder_edge_arc[iteration](a_parent)
                    loss += self.loss_sce(target_a_parent, a_parent)
                    self.accuracies["edge/arc"](target_a_parent, a_parent)

                    a_child = tf.nn.tanh(self.layers.sum([tf.gather(edge_parents, target_indices_b),
                                                          tf.gather(edge_children, target_indices_a)]))
                    if args.highway:
                        a_child += self.layers.decoder_edge_highway[iteration](a_child)
                    a_child = self.layers.decoder_edge_arc[iteration](a_child)
                    loss += self.loss_sce(target_a_child, a_child)
                    self.accuracies["edge/arc"](target_a_child, a_child)

                    # 2.2) Compute deprels
                    deprel_parents = self.layers.decoder_deprel_parents[iteration](nodes)
                    deprel_children = self.layers.decoder_deprel_children[iteration](nodes)
                    deprel_weights = tf.nn.tanh(self.layers.sum(
                        [tf.gather(deprel_parents, target_deprel_parents),
                         tf.gather(deprel_children, target_deprel_children)]))
                    if args.highway:
                        deprel_weights += self.layers.decoder_deprel_highway[iteration](deprel_weights)
                    for i, prop in enumerate(train.edge_properties):
                        predictions = self.layers.decoder_deprel_values[iteration][i](deprel_weights)
                        loss += self.loss_sce(target_deprel_values[:, i], predictions)
                        self.accuracies["edge/" + prop](target_deprel_values[:, i], predictions)

                    self.State.recompute_edges(ori_nodes, states, iteration)

                # Tops
                sum_nodes = sum(s.n_nodes for s in states)
                nodes = tf.concat([s.nodes for s in states], axis=0)
                target_tops = np.zeros([sum_nodes], np.int32)
                start = 0
                for i, sentence in enumerate(sentences):
                    for e in sentence.children[0]:
                        if sentence.factors[Sentence.EDGE_VALUES][e][0] == Mapping.ROOT:
                            c = sentence.factors[Sentence.EDGE_CHILDREN][e]
                            if c in states[i].node_mapping:
                                target_tops[start + states[i].node_mapping[c]] = 1
                    start += states[i].n_nodes
                predictions = self.layers.decoder_tops(nodes)
                loss += self.loss_sce(target_tops, predictions)
                self.accuracies["edge/tops"](target_tops, predictions)

            tg = tape.gradient(loss, self.layers.trainable_variables)
            tg_none = [variable.name for g, variable in zip(tg, self.layers.trainable_variables) if g is None]
            if tg_none:
                print("Skipping a batch with None gradient for variables {}".format(tg_none), file=sys.stderr, flush=True)
                continue

            if num_gradients == 0:
                gradients = [g.numpy() if not isinstance(g, tf.IndexedSlices) else [(g.values.numpy(), g.indices.numpy())] for g in tg]
            else:
                for g, ng in zip(gradients, tg):
                    if isinstance(g, list):
                        g.append((ng.values.numpy(), ng.indices.numpy()))
                    else:
                        g += ng.numpy()
            num_gradients += 1
            if num_gradients == args.batch_aggregation or len(train._permutation) == 0:
                gradients = [tf.IndexedSlices(*map(np.concatenate, zip(*g))) if isinstance(g, list) else g for g in gradients]
                self.optimizer.apply_gradients(zip(gradients, self.layers.trainable_variables))
                num_gradients = 0
                if int(self.optimizer.iterations) % 100 == 0:
                    tf.summary.experimental.set_step(self._summary_step())
                    with self.writer.as_default():
                        for name, accuracy in self.accuracies.items():
                            tf.summary.scalar("train/" + name, accuracy.result())

        tf.summary.experimental.set_step(self._summary_step())
        with self.writer.as_default():
            tf.summary.scalar("train/skipped", skipped)

    def predict(self, data, args):
        output = []

        while not data.epoch_finished():
            batch = data.next_batch(args.batch_size)
            sentences = batch["sentences"]

            encoded_tokens = self._encoder(
                token_ids=batch["batch_factors"][Sentence.TOKENS],
                token_charseqs=batch["charseqs"],
                token_charseq_ids=batch["charseq_ids"],
                token_values=batch["batch_factors"][Sentence.TOKEN_VALUES],
                token_additionals=[batch[x] for x in ["bert", "fasttext", "word2vec"] if x in batch],
                training=False)

            states = []
            for i, sentence in enumerate(sentences):
                states.append(self.State(self, sentence.empty_copy(), encoded_tokens[i][:sentence.n_tokens()], False))

            for iteration in range(args.decoder_iterations):
                # 1) New nodes
                ori_nodes = [s.n_nodes for s in states]
                sum_nodes = sum(ori_nodes)
                nodes = tf.concat([s.nodes for s in states], axis=0)

                prediction_ops = tf.argmax(self.layers.decoder_node_operation[iteration](nodes), axis=1).numpy()
                prediction_node_values = np.stack(
                    [tf.argmax(decoder(nodes), axis=1).numpy() for decoder in self.layers.decoder_node_values[iteration]], axis=1)
#                 prediction_edge_values = np.stack(
#                     [tf.argmax(decoder(nodes), axis=1).numpy() for decoder in self.layers.decoder_edge_values[iteration]], axis=1)
                start = 0
                for i, state in enumerate(states):
                    for n in range(1, ori_nodes[i]):
                        if prediction_ops[start + n] == NODE_NO: continue
                        state.add_node(prediction_node_values[start + n], n)
#                         edge_values = prediction_edge_values[start + n] if n >= state.sentence.n_tokens() else [Mapping.ANCHOR] + [Mapping.NONE] * (len(data.edge_properties) - 1)
#                         if prediction_ops[start + n] == NODE_PARENT:
#                             state.add_edge(state.n_nodes - 1, n, edge_values)
#                         else:
#                             state.add_edge(n, state.n_nodes - 1, edge_values)
                    start += ori_nodes[i]
                self.State.recompute_nodes(nodes, states, iteration)

                # 2) Edges
                sum_nodes = sum(s.n_nodes for s in states)
                nodes = tf.concat([s.nodes for s in states], axis=0)
                indices_a, indices_b = [], []
                indices_parents, indices_children = [], []

                start = 0
                for i, state in enumerate(states):
                    for j in range(ori_nodes[i], state.n_nodes):
                        indices_a.append(start + np.repeat(np.int32(j), state.n_nodes))
                        indices_b.append(start + np.arange(0, state.n_nodes, dtype=np.int32))
                    start += state.n_nodes
                if not indices_a: continue

                indices_a = np.concatenate(indices_a, axis=0)
                indices_b = np.concatenate(indices_b, axis=0)

                # 2.1) Compute arcs
                edge_parents = self.layers.decoder_edge_parents[iteration](nodes)
                edge_children = self.layers.decoder_edge_children[iteration](nodes)

                a_parent = tf.nn.tanh(self.layers.sum([tf.gather(edge_parents, indices_a),
                                                       tf.gather(edge_children, indices_b)]))
                if args.highway:
                    a_parent += self.layers.decoder_edge_highway[iteration](a_parent)
                a_parent = tf.argmax(self.layers.decoder_edge_arc[iteration](a_parent), axis=1).numpy()
                a_child = tf.nn.tanh(self.layers.sum([tf.gather(edge_parents, indices_b),
                                                      tf.gather(edge_children, indices_a)]))
                if args.highway:
                    a_child += self.layers.decoder_edge_highway[iteration](a_child)
                a_child = tf.argmax(self.layers.decoder_edge_arc[iteration](a_child), axis=1).numpy()

                start, offset = 0, 0
                for i, state in enumerate(states):
                    node = 0
                    for n in range(1, ori_nodes[i]):
                        if prediction_ops[start + n] == NODE_NO: continue
                        if prediction_ops[start + n] == NODE_PARENT:
                            a_parent[offset + node * state.n_nodes + n] = 1
                        else:
                            a_child[offset + node * state.n_nodes + n] = 1
                        node += 1
                    start += ori_nodes[i]
                    offset += (state.n_nodes - ori_nodes[i]) * state.n_nodes
                for i in np.where(a_parent > 0)[0]:
                    indices_parents.append(indices_a[i])
                    indices_children.append(indices_b[i])
                for i in np.where(a_child > 0)[0]:
                    indices_parents.append(indices_b[i])
                    indices_children.append(indices_a[i])
                indices_parents = np.array(indices_parents, np.int32)
                indices_children = np.array(indices_children, np.int32)

                # 2.2) Compute deprels
                deprel_parents = self.layers.decoder_deprel_parents[iteration](nodes)
                deprel_children = self.layers.decoder_deprel_children[iteration](nodes)
                deprel_weights = tf.nn.tanh(self.layers.sum([tf.gather(deprel_parents, indices_parents),
                                                             tf.gather(deprel_children, indices_children)]))
                if args.highway:
                    deprel_weights += self.layers.decoder_deprel_highway[iteration](deprel_weights)
                edge_predictions = []
                for i, prop in enumerate(data.edge_properties):
                    edge_predictions.append(tf.argmax(
                        self.layers.decoder_deprel_values[iteration][i](deprel_weights), axis=1).numpy())
                edge_predictions = np.stack(edge_predictions, axis=1)

                offset = 0
                for _ in range(2):
                    start = 0
                    for state in states:
                        while offset < len(indices_parents) and indices_parents[offset] >= start and indices_parents[offset] < start + state.n_nodes:
                            state.add_edge(
                                indices_parents[offset] - start,
                                indices_children[offset] - start,
                                edge_predictions[offset])
                            offset += 1
                        start += state.n_nodes
                assert offset == len(indices_parents)

                self.State.recompute_edges(ori_nodes, states, iteration)

            # Tops
            sum_nodes = sum(s.n_nodes for s in states)
            nodes = tf.concat([s.nodes for s in states], axis=0)
            predictions = tf.argmax(self.layers.decoder_tops(nodes), axis=1)
            start = 0
            for i, state in enumerate(states):
                for n in range(1, state.n_nodes):
                    if predictions[start + n]:
                        state.add_edge(0, n, [Mapping.ROOT] + [Mapping.NONE] * (len(data.edge_properties) - 1))
                start += state.n_nodes

            for state in states:
                output.append(state.sentence.write(data.mappings))
                output.append("\n")

        return "".join(output)

    @staticmethod
    def predict_ensemble(networks, data, args):
        output = []

        while not data.epoch_finished():
            batch = data.next_batch(args.batch_size)
            sentences = batch["sentences"]

            for network in networks:
                encoded_tokens = network._encoder(
                    token_ids=batch["batch_factors"][Sentence.TOKENS],
                    token_charseqs=batch["charseqs"],
                    token_charseq_ids=batch["charseq_ids"],
                    token_values=batch["batch_factors"][Sentence.TOKEN_VALUES],
                    token_additionals=[batch[x] for x in ["bert", "fasttext", "word2vec"] if x in batch],
                    training=False)

                network.states = []
                for i, sentence in enumerate(sentences):
                    network.states.append(network.State(network, sentence.empty_copy(), encoded_tokens[i][:sentence.n_tokens()], False))

            for iteration in range(args.decoder_iterations):
                # 1) New nodes
                ori_nodes = [s.n_nodes for s in networks[0].states]
                sum_nodes = sum(ori_nodes)
                for network in networks:
                    network.nodes = tf.concat([s.nodes for s in network.states], axis=0)

                prediction_ops = np.argmax(
                    np.mean([network.layers.decoder_node_operation[iteration](network.nodes).numpy() for network in networks], axis=0),
                    axis=1)
                prediction_node_values = np.stack([
                    np.argmax(
                        np.mean([network.layers.decoder_node_values[iteration][i](network.nodes).numpy() for network in networks], axis=0),
                        axis=1)
                    for i in range(len(networks[0].layers.decoder_node_values[iteration]))
                ], axis=1)

                for network in networks:
                    start = 0
                    for i, state in enumerate(network.states):
                        for n in range(1, ori_nodes[i]):
                            if prediction_ops[start + n] == NODE_NO: continue
                            state.add_node(prediction_node_values[start + n], n)
                        start += ori_nodes[i]
                    network.State.recompute_nodes(network.nodes, network.states, iteration)

                # 2) Edges
                sum_nodes = sum(s.n_nodes for s in networks[0].states)
                for network in networks:
                    network.nodes = tf.concat([s.nodes for s in network.states], axis=0)
                indices_a, indices_b = [], []
                indices_parents, indices_children = [], []

                start = 0
                for i, state in enumerate(networks[0].states):
                    for j in range(ori_nodes[i], state.n_nodes):
                        indices_a.append(start + np.repeat(np.int32(j), state.n_nodes))
                        indices_b.append(start + np.arange(0, state.n_nodes, dtype=np.int32))
                    start += state.n_nodes
                if not indices_a: continue

                indices_a = np.concatenate(indices_a, axis=0)
                indices_b = np.concatenate(indices_b, axis=0)

                # 2.1) Compute arcs
                a_parent, a_child = [], []
                for network in networks:
                    edge_parents = network.layers.decoder_edge_parents[iteration](network.nodes)
                    edge_children = network.layers.decoder_edge_children[iteration](network.nodes)

                    a_parent.append(tf.nn.tanh(network.layers.sum([tf.gather(edge_parents, indices_a),
                                                                   tf.gather(edge_children, indices_b)])))
                    if args.highway:
                        a_parent[-1] += network.layers.decoder_edge_highway[iteration](a_parent[-1])
                    a_parent[-1] = network.layers.decoder_edge_arc[iteration](a_parent[-1]).numpy()

                    a_child.append(tf.nn.tanh(network.layers.sum([tf.gather(edge_parents, indices_b),
                                                                   tf.gather(edge_children, indices_a)])))
                    if args.highway:
                        a_child[-1] += network.layers.decoder_edge_highway[iteration](a_child[-1])
                    a_child[-1] = network.layers.decoder_edge_arc[iteration](a_child[-1]).numpy()
                a_parent = np.argmax(np.mean(a_parent, axis=0), axis=1)
                a_child = np.argmax(np.mean(a_child, axis=0), axis=1)

                start, offset = 0, 0
                for i, state in enumerate(networks[0].states):
                    node = 0
                    for n in range(1, ori_nodes[i]):
                        if prediction_ops[start + n] == NODE_NO: continue
                        if prediction_ops[start + n] == NODE_PARENT:
                            a_parent[offset + node * state.n_nodes + n] = 1
                        else:
                            a_child[offset + node * state.n_nodes + n] = 1
                        node += 1
                    start += ori_nodes[i]
                    offset += (state.n_nodes - ori_nodes[i]) * state.n_nodes
                for i in np.where(a_parent > 0)[0]:
                    indices_parents.append(indices_a[i])
                    indices_children.append(indices_b[i])
                for i in np.where(a_child > 0)[0]:
                    indices_parents.append(indices_b[i])
                    indices_children.append(indices_a[i])
                indices_parents = np.array(indices_parents, np.int32)
                indices_children = np.array(indices_children, np.int32)

                # 2.2) Compute deprels
                edge_predictions = []
                for i, prop in enumerate(data.edge_properties):
                    predictions = []
                    for network in networks:
                        deprel_parents = network.layers.decoder_deprel_parents[iteration](network.nodes)
                        deprel_children = network.layers.decoder_deprel_children[iteration](network.nodes)
                        deprel_weights = tf.nn.tanh(network.layers.sum([tf.gather(deprel_parents, indices_parents),
                                                                        tf.gather(deprel_children, indices_children)]))
                        if args.highway:
                            deprel_weights += network.layers.decoder_deprel_highway[iteration](deprel_weights)
                        predictions.append(network.layers.decoder_deprel_values[iteration][i](deprel_weights).numpy())
                    edge_predictions.append(np.argmax(np.mean(predictions, axis=0), axis=1))
                edge_predictions = np.stack(edge_predictions, axis=1)

                for network in networks:
                    offset = 0
                    for _ in range(2):
                        start = 0
                        for state in network.states:
                            while offset < len(indices_parents) and indices_parents[offset] >= start and indices_parents[offset] < start + state.n_nodes:
                                state.add_edge(
                                    indices_parents[offset] - start,
                                    indices_children[offset] - start,
                                    edge_predictions[offset])
                                offset += 1
                            start += state.n_nodes
                    assert offset == len(indices_parents)

                    network.State.recompute_edges(ori_nodes, network.states, iteration)

            # Tops
            sum_nodes = sum(s.n_nodes for s in networks[0].states)
            for network in networks:
                network.nodes = tf.concat([s.nodes for s in network.states], axis=0)
            predictions = np.argmax(
                np.mean([network.layers.decoder_tops(network.nodes).numpy() for network in networks], axis=0),
                axis=1)
            for network in networks:
                start = 0
                for i, state in enumerate(network.states):
                    for n in range(1, state.n_nodes):
                        if predictions[start + n]:
                            state.add_edge(0, n, [Mapping.ROOT] + [Mapping.NONE] * (len(data.edge_properties) - 1))
                    start += state.n_nodes

            for state in networks[0].states:
                output.append(state.sentence.write(data.mappings))
                output.append("\n")

        return "".join(output)

    def evaluate(self, dataset, data, gold_path, args):
        step = self._summary_step()

        predicted_path = os.path.join(args.logdireval, "{}_{:07d}.mrp".format(dataset, step))
        assert not os.path.exists(predicted_path)

        with open(predicted_path, "w", encoding="utf-8") as predicted_file:
            predicted = self.predict(data, args)
            print(predicted, end="", file=predicted_file)

        if dataset in ["dev", "bdev", "lpss"]:
            mtool_evaluate.mrp_eval(
                gold_path, predicted_path, "{}_eval".format(dataset), rrhc=4, mces=100000, parallelize=8, logdir=args.logdireval, logstep=step, distributed=True)
