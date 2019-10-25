#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""MRP dataset."""

import json
import os
import pickle
import sys

import numpy as np

from mapping import Mapping
from sentence import Sentence
from tokenizer import RuleBasedTokenizer

class MRPDataset:

    def __new__(cls, filename=None, mode=None, train=None, companion=None,
                jamr=None, ner=None, max_sentences=None, word2vec=None,
                fasttext=None, bert=None, max_sentence_len=None):
        """Load dataset from JSON file.
        Arguments:
            filename: Path to MRP file.
            mode: dm|eds|psd|ucca|amr.
            train: If given, the words and alphabets are reused from the
                training data.
            companion: Companion MRPDataset (optional;
                required only for mode == "amr").
            jamr: JAMR alignments filename (optional;
                required only for mode == "amr").
            ner: NE tagged MRPDataset (optional).
            max_sentences: If given, only use max_sentences (for debugging).
            word2vec: word2vec model (result of word2vec.load).
            fasttext: FastText model (result of fasttext.load).
            bert: BERT filename.
        """

        if filename is None:
            return object.__new__(cls)

        ######################
        ### INITIALIZATION ###
        ######################

        # Generate name for the pickled dataset
        pickled_name = "datasets/{}{}-{}-{}-{}-{}-{}.pickle".format(
            filename.replace("/", "+"),
            max_sentences or "",
            mode,
            (train.filename if train else "").replace("/", "+"),
            (companion.filename if companion else "").replace("/", "+"),
            (ner.filename if ner else "").replace("/", "+"),
            (jamr or "").replace("/", "+")
        )

        if os.path.exists(pickled_name):
            with open(pickled_name, "rb") as pickled_file:
                self = pickle.load(pickled_file)
        else:
            self = object.__new__(cls)
            self.filename = filename

            # Create mappings
            self._mappings = [None] * Sentence.FACTORS
            for m in Sentence.MAPPING_1D:
                self._mappings[m] = (Mapping(train._mappings[m] if train else None,
                                             include_characters = True if m in Sentence.FACTORS_CHARS else False,
                                             is_encoded = False if m in Sentence.FACTORS_NOT_ENCODED else None))
            for m in Sentence.MAPPING_2D:
                self._mappings[m] = dict()

            # Create structures
            self._sentences = []

            # Initialize tokenizer
            tokenizer = RuleBasedTokenizer(finegrained = True if mode == "ucca" else False)

            # Sentence ids mapping
            self.sentence_ids = dict()

            #####################
            ### PREPROCESSING ###
            #####################

            # Find node and edge properties and instantiate their mappings
            if train:
                self.node_properties = train.node_properties
                self.node_property_types = train.node_property_types
                for prop in self.node_properties:
                    self._mappings[Sentence.NODE_VALUES][prop] = Mapping(train=train.mappings[Sentence.NODE_VALUES][prop])

                self.edge_properties = train.edge_properties
                self.edge_property_types = train.edge_property_types
                for prop in self.edge_properties:
                    self._mappings[Sentence.EDGE_VALUES][prop] = Mapping(train=train.mappings[Sentence.EDGE_VALUES][prop])

                self.token_properties = train.token_properties
                self._mappings[Sentence.TOKEN_VALUES] = train._mappings[Sentence.TOKEN_VALUES]
                self.token_edge_properties = train.token_edge_properties
                self._mappings[Sentence.TOKEN_EDGE_VALUES] = train._mappings[Sentence.TOKEN_EDGE_VALUES]
            else:
                self._mappings[Sentence.NODE_VALUES]["label"] = Mapping(is_encoded = False if Sentence.NODE_VALUES in Sentence.FACTORS_NOT_ENCODED else None)
                self.node_properties = ["label"]
                self.node_property_types = [str]
                self._mappings[Sentence.EDGE_VALUES]["label"] = Mapping(is_encoded = False if Sentence.EDGE_VALUES in Sentence.FACTORS_NOT_ENCODED else None)
                self.edge_properties = ["label"]
                self.edge_property_types = [str]

                with open(filename, "r", encoding="utf-8") as file:
                    for line in file:
                        data = json.loads(line)

                        # Node label and properties
                        if "nodes" in data:
                            for node in data["nodes"]:
                                if "properties" in node:
                                    for p, prop in enumerate(node["properties"]):
                                        if prop not in self._mappings[Sentence.NODE_VALUES]:
                                            self._mappings[Sentence.NODE_VALUES][prop] = Mapping(is_encoded = False if Sentence.NODE_VALUES in Sentence.FACTORS_NOT_ENCODED else None)
                                            self.node_properties.append(prop)
                                            self.node_property_types.append(type(node["values"][p]))

                        # Edge label and properties
                        if "edges" in data:
                            for edge in data["edges"]:
                                if "properties" in edge:
                                    for p, prop in enumerate(edge["properties"]):
                                        if prop not in self._mappings[Sentence.EDGE_VALUES]:
                                            self._mappings[Sentence.EDGE_VALUES][prop] = Mapping(is_encoded = False if Sentence.EDGE_VALUES in Sentence.FACTORS_NOT_ENCODED else None)
                                            self.edge_properties.append(prop)
                                            self.edge_property_types.append(type(edge["values"][p]))

                # Token and token_edges properties
                self.token_properties = []
                self._mappings[Sentence.TOKEN_VALUES] = {}
                self.token_edge_properties = []
                self._mappings[Sentence.TOKEN_EDGE_VALUES] = {}
                if companion:
                    self.token_properties.extend(["companion_" + p for p in companion.node_properties])
                    self._mappings[Sentence.TOKEN_VALUES].update(
                        {"companion_" + p: companion.mappings[Sentence.NODE_VALUES][p] for p in companion.node_properties})
                    self.token_edge_properties.extend(["companion_" + p for p in companion.edge_properties])
                    self._mappings[Sentence.TOKEN_EDGE_VALUES].update(
                        {"companion_" + p: companion.mappings[Sentence.EDGE_VALUES][p] for p in companion.edge_properties})
                if ner:
                    self.token_properties.extend(["ner_" + p for p in ner.node_properties])
                    self._mappings[Sentence.TOKEN_VALUES].update(
                        {"ner_" + p: ner.mappings[Sentence.NODE_VALUES][p] for p in ner.node_properties})

            # Get node to companion alignment from JAMR file for AMR.
            self.node_to_companion = dict()
            if mode == "amr" and jamr:
                with open(jamr, "r", encoding="utf-8") as file:
                    for line in file:
                        data = json.loads(line)
                        self.node_to_companion[data["id"]] = dict()
                        if "nodes" in data:
                            for node in data["nodes"]:
                                links_dict = dict()
                                if "label" in node:
                                    for label in node["label"]:
                                        links_dict[label] = 1
                                if "values" in node:
                                    for value_list in node["values"]:
                                        for value in value_list:
                                            links_dict[value] = 1
                                self.node_to_companion[data["id"]][node["id"]] = sorted(links_dict.keys())

            ##################
            ### PROCESSING ###
            ##################

            # Load the sentences from JSON file
            print("Reading MRPDataset from file {}".format(filename), file=sys.stderr, flush=True)
            with open(filename, "r", encoding="utf-8") as file:
                # Count some stats for warning printouts
                anchors, anchors_without_token = 0, 0
                ne_values, ne_values_without_token = 0, 0
                ne_labels, ne_labels_without_values = 0, 0

                for line in file:                   # 1 line = 1 sentence
                    data = json.loads(line)

                    # For eds, replace + in node labels to Sentence.FORM_SEP
                    if data.get("framework", None) == "eds" and "nodes" in data:
                        for node in data["nodes"]:
                            if "label" in node:
                                node["label"] = node["label"].replace("+", Sentence.FORM_SEP)

                    ### SENTENCE ###
                    self.sentence_ids[data["id"]] = len(self._sentences)
                    self._sentences.append(Sentence(self,
                                                    {"id": data["id"],
                                                     "flavor": data["flavor"],
                                                     "framework": data["framework"],
                                                     "version": data["version"],
                                                     "time": data["time"],
                                                     "input": data["input"],
                                                     "mode": mode}))

                    ### TOKENS, with ROOT being the first ###
                    tokens = ["<root>"]
                    token_ids = [Mapping.ROOT]
                    charseq_ids = [Mapping.ROOT]
                    token_ranges = dict()

                    for i, token_range in enumerate(tokenizer.tokenize(data["input"])):
                        token = data["input"][token_range[0]:token_range[1]]
                        (token_id, charseq_id) = self._mappings[Sentence.TOKENS].add_string(token, train=train._mappings[Sentence.TOKENS] if train else None)
                        tokens.append(token)
                        token_ids.append(token_id)
                        charseq_ids.append(charseq_id)
                        token_ranges["{}-{}".format(token_range[0], token_range[1])] = i + 1

                    # Companion Token data -- token_values and token_edges
                    token_values = np.zeros((len(tokens), 0), np.int32)
                    token_edge_parents, token_edge_children, token_edge_values = [], [], []

                    for extra in [companion, ner]:
                        if not extra: continue

                        extra_sentence = extra.sentences[extra.sentence_ids[data["id"]]]
                        extra_values = np.ones((len(tokens), len(extra.node_properties)), np.int32) * Mapping.UNK
                        extra_token_from_node = {}
                        for i in range(len(tokens)):
                            if i == 0:
                                extra_values[0] = [Mapping.ROOT for _ in extra.node_properties]
                                extra_token_from_node[0] = 0
                            else:
                                for e in extra_sentence.parents[i][:1]:
                                    node = extra_sentence.factors[Sentence.EDGE_PARENTS][e]
                                    extra_token_from_node[node] = i
                                    extra_values[i] = extra_sentence.factors[Sentence.NODE_VALUES][node - extra_sentence.n_tokens()]
                        token_values = np.concatenate([token_values, extra_values], axis=1)

                        if extra == companion:
                            for parent, child, values in zip(*[extra_sentence.factors[f] for f in
                                                               [Sentence.EDGE_PARENTS, Sentence.EDGE_CHILDREN, Sentence.EDGE_VALUES]]):
                                if parent in extra_token_from_node and child in extra_token_from_node:
                                    token_edge_parents.append(extra_token_from_node[parent])
                                    token_edge_children.append(extra_token_from_node[child])
                                    token_edge_values.append(values)

                    self._sentences[-1].init_tokens({"tokens": tokens,
                                                     "token_ids": token_ids,
                                                     "token_values": token_values,
                                                     "token_ranges": token_ranges,
                                                     "charseq_ids": charseq_ids})

                    self._sentences[-1].init_token_edges({"token_edge_parents": token_edge_parents,
                                                          "token_edge_children": token_edge_children,
                                                          "token_edge_values": token_edge_values})

                    # Compute corrected node ids in sequential order, offset by number of tokens
                    node_ids_to_ordered_ids = dict()
                    if "nodes" in data:
                        for n, node in enumerate(data["nodes"]):
                            node_ids_to_ordered_ids[node["id"]] = n + len(tokens)


                    ### TOPS, NODES AND EDGES ###
                    node_values = []

                    edge_parents, edge_children, edge_values = [], [], []

                    ### TOPS ###
                    # Tops are modeled as special edges (with label Mapping.ROOT) to ROOT token.
                    if "tops" in data:
                        for top in data["tops"]:
                            edge_parents.append(0)
                            edge_children.append(node_ids_to_ordered_ids[top])
                            # Create edge values
                            edge_values.append([Mapping.ROOT for _ in self.edge_properties])

                    ### NODES ###

                    if "nodes" not in data:
                        data["nodes"] = []

                    # Get corresponding companion parse
                    if companion:
                        companion_sentence = companion.sentences[companion.sentence_ids[data["id"]]]

                    # Move node labels to node properties
                    for i, node in enumerate(data["nodes"]):
                        if "label" not in node: continue

                        if not "properties" in node: data["nodes"][i]["properties"] = []
                        data["nodes"][i]["properties"].append("label")

                        if not "values" in node: data["nodes"][i]["values"] = []
                        data["nodes"][i]["values"].append(node["label"])

                    # Get nodes
                    for i, node in enumerate(data["nodes"]):

                        # Anchors
                        node_anchored_tokens = []

                        if mode == "amr" and self.node_to_companion:
                            # Create anchors in AMR file from companion parse and JAMR alignment.
                            if data["id"] in self.node_to_companion and node["id"] in self.node_to_companion[data["id"]]:
                                for companion_node_id in self.node_to_companion[data["id"]][node["id"]]:
                                    for e in companion_sentence.children[companion_node_id + len(tokens)]:
                                        if companion_sentence.factors[Sentence.EDGE_VALUES][e][0] == Mapping.ANCHOR:
                                            # Create edge from node to token id
                                            edge_parents.append(i + len(tokens))
                                            edge_children.append(companion_sentence.factors[Sentence.EDGE_CHILDREN][e])
                                            # Create edge values
                                            edge_values.append([Mapping.ANCHOR for _ in self.edge_properties])

                                            # Remember this anchor for this node
                                            node_anchored_tokens.append(self._sentences[-1].factor_strings[Sentence.TOKENS][edge_children[-1]])
                        else:
                            if "anchors" in node:
                                for anchor in node["anchors"]:
                                    anchors += 1
                                    start = str(anchor["from"])
                                    end = str(anchor["to"])
                                    if start not in self._sentences[-1].token_starts \
                                            or end not in self._sentences[-1].token_ends:
                                        anchors_without_token += 1
                                        continue
                                    for t in range(self._sentences[-1].token_starts[start], self._sentences[-1].token_ends[end]+1):
                                        # Create edge from node to token id
                                        edge_parents.append(i + len(tokens))
                                        edge_children.append(t)
                                        # Create edge values
                                        edge_values.append([Mapping.ANCHOR for _ in self.edge_properties])
                                        # Remember this anchor for this node
                                        node_anchored_tokens.append(self._sentences[-1].factor_strings[Sentence.TOKENS][t])
                        node_anchored_tokens = Sentence.FORM_SEP.join(node_anchored_tokens)

                        # if mode == "eds":
                        #     with open("stats/eds_joined_labels.txt", mode="a", encoding="utf-8") as eds_fw:
                        #         print("{}\t{}".format(node["label"], node_anchored_tokens), file=eds_fw)

                        # if mode == "amr" and "label" in node and node["label"] == "name":
                        #     with open("../generated/stats/amr_joined_ops.txt", mode="a", encoding="utf-8") as amr_fw:
                        #         values = []
                        #         for p, prop in enumerate(node["properties"]):
                        #             if not prop.startswith("op"): continue
                        #             values.append(node["values"][p])
                        #         print("{}\t{}".format(Sentence.FORM_SEP.join(values), node_anchored_tokens), file=amr_fw)

                        # Get node values (labels included)
                        node_values.append([])
                        for p, prop in enumerate(self.node_properties):
                            if "properties" not in node or prop not in node["properties"]:
                                node_values[-1].append(Mapping.NONE)
                            else:
                                # Get corresponding values
                                indexes = [i for i, j in enumerate(node["properties"]) if j == prop]
                                value = Sentence.PROP_SEP.join(map(str, [node["values"][i] for i in indexes]))
                                # Node values are encoded
                                string_id = self._mappings[Sentence.NODE_VALUES][prop].add_string(value,
                                                                                                  encoded_from=node_anchored_tokens,
                                                                                                  train=train.mappings[Sentence.NODE_VALUES][prop] if train else None)

                                node_values[-1].append(string_id)

                    # Add nodes to sentence
                    self._sentences[-1].init_nodes(node_values)

                    ### EDGES ###
                    if "edges" not in data:
                        data["edges"] = []

                    # Move edge labels to edge properties
                    for i, edge in enumerate(data["edges"]):
                        if "label" not in edge: continue

                        if not "properties" in edge: data["edges"][i]["properties"] = []
                        data["edges"][i]["properties"].append("label")

                        if not "values" in edge: data["edges"][i]["values"] = []
                        data["edges"][i]["values"].append(edge["label"])

                    # Get edges
                    for edge in data["edges"]:

                        # Get edge source and target
                        edge_parents.append(node_ids_to_ordered_ids[edge["source"]])
                        edge_children.append(node_ids_to_ordered_ids[edge["target"]])

                        # Get edge values
                        edge_values.append([])
                        for p, prop in enumerate(self.edge_properties):
                            if "properties" not in edge or prop not in edge["properties"]:
                                edge_values[-1].append(Mapping.NONE)
                                continue
                            # Get corresponding values
                            indexes = [i for i, j in enumerate(edge["properties"]) if j == prop]
                            value = Sentence.PROP_SEP.join(map(str, [edge["values"][i] for i in indexes]))
                            edge_values[-1].append(self._mappings[Sentence.EDGE_VALUES][prop].add_string(value, train=train.mappings[Sentence.EDGE_VALUES][prop] if train else None))

                    # Add edges to sentence
                    self._sentences[-1].init_edges({"edge_parents": edge_parents,
                                                    "edge_children": edge_children,
                                                    "edge_values": edge_values})


                    if len(self._sentences) % 10000 == 0:
                        print("Read {} sentences.".format(len(self._sentences)), file=sys.stderr, flush=True)

                    if max_sentences is not None and len(self._sentences) >= max_sentences:
                        break

                print("Read {} sentences.".format(len(self._sentences)), file=sys.stderr, flush=True)
                if anchors_without_token > 0:
                    print("Note: {} ({:.2f}%) anchors do not correspond to tokens.".format(anchors_without_token, anchors_without_token * 100 / anchors), file=sys.stderr, flush=True)
                if mode == "amr" and ne_labels_without_values > 0:
                    print("Note: {} ({:.2f}%) NE labeled nodes (label=\"{}\") are without node values.".format(ne_labels_without_values, ne_labels_without_values * 100 / ne_labels, "name"), file=sys.stderr, flush=True)
                if mode == "amr" and ne_values_without_token > 0:
                    print("Note: {} ({:.2f}%) values in NE labeled nodes (label=\"{}\") do not correspond to tokens.".format(ne_values_without_token, ne_values_without_token * 100 / ne_values, "name"), file=sys.stderr, flush=True)

            ######################
            ### POSTPROCESSING ###
            ######################

            # Choose relative or absolute rules for each mapping
            if not train:
                node_values_mapping = self._mappings[Sentence.NODE_VALUES]
                new_node_values_mapping = {}
                for name, mapping in node_values_mapping.items():
                    if len(mapping.strings) < len(mapping.strings_original):
                        new_node_values_mapping[name] = Mapping.from_absolute_encodings(mapping, is_encoded=True)
                    else:
                        new_node_values_mapping[name] = Mapping(include_characters=mapping._include_characters, is_encoded=False, train=None)
                for sentence in self._sentences:
                    for i in range(sentence.n_nodes()):
                        node_anchored_tokens = []
                        for e in sentence.children[i + sentence.n_tokens()]:
                            if sentence.factors[Sentence.EDGE_VALUES][e][0] == Mapping.ANCHOR:
                                child = sentence.factors[Sentence.EDGE_CHILDREN][e]
                                node_anchored_tokens.append(sentence.factor_strings[Sentence.TOKENS][child])
                        node_anchored_tokens = Sentence.FORM_SEP.join(node_anchored_tokens)

                        for p, prop in enumerate(self.node_properties):
                            value = node_values_mapping[prop].id_to_string(
                                sentence.factors[Sentence.NODE_VALUES][i][p], encoded_from=node_anchored_tokens)
                            sentence.factors[Sentence.NODE_VALUES][i][p] = new_node_values_mapping[prop].add_string(
                                value, encoded_from=node_anchored_tokens)
                self._mappings[Sentence.NODE_VALUES] = new_node_values_mapping

                mapping_logs = []
                for name, mapping in node_values_mapping.items():
                    mapping_logs.append("{}(u{},re{}):".format(name, len(mapping.strings_original), len(mapping.strings)))
                    if new_node_values_mapping[name].is_encoded:
                        mapping_logs[-1] += "relative-{}".format(len(new_node_values_mapping[name].strings))
                    else:
                        mapping_logs[-1] += "absolute"
                print("Mapping", ", ".join(mapping_logs), file=sys.stderr)

            # Computes factor lengths for all sentences
            self._factor_lens = []
            for f in range(Sentence.FACTORS):
                self._factor_lens.append(np.array([ len(s.factors[f]) for s in self._sentences ], np.int32))
            self._sentence_lens = self._factor_lens[Sentence.TOKENS]


            # Serialize the model
            with open(pickled_name, "wb") as pickled_file:
                pickle.dump(self, pickled_file)

        # Save max_sentence_len
        self._max_sentence_len = max_sentence_len

        # Generate random permutation.
        self._shuffle_batches = train is None
        self._permutation = np.random.permutation(len(self._sentence_lens)) if self._shuffle_batches else np.arange(len(self._sentence_lens))

        # Pretrained word embeddings
        self._word2vec = word2vec
        self._fasttext = fasttext

        # Load pretrained BERT embeddings
        self._bert = []  # [sentences x tokens x bert_embeddings]
        if bert:
            print("Loading BERT embeddings from file {}".format(bert), file=sys.stderr, flush=True)
            with np.load(bert) as file:
                for i, (_, value) in enumerate(file.items()):
                    self._bert.append(value)
                    if max_sentences is not None and len(self._bert) >= max_sentences:
                        break
            assert len(self._bert) == len(self._sentences)
            assert list(map(len, self._bert)) == list(self._sentence_lens - 1)

        # Asserts
        if companion:
            assert len(companion.sentences) == len(self._sentences), "Number of companion sentences != number of dataset sentences ({} != {}).".format(len(companion.sentences), len(self._sentences))
            for s in range(len(self._sentences)):
                assert companion.sentences[s].factor_strings[Sentence.TOKENS] == self._sentences[s].factor_strings[Sentence.TOKENS], "Companion tokens != sentence tokens:\n{}\n{}.".format(companion.sentences[s].factor_strings[Sentence.TOKENS], self._sentences[s].factor_strings[Sentence.TOKENS])
        if ner:
            assert len(ner.sentences) == len(self._sentences), "Number of ner sentences != number of dataset sentences ({} != {}).".format(len(ner.sentences), len(self._sentences))
            for s in range(len(self._sentences)):
                assert ner.sentences[s].factor_strings[Sentence.TOKENS] == self._sentences[s].factor_strings[Sentence.TOKENS], "Companion tokens != sentence tokens:\n{}\n{}.".format(ner.sentences[s].factor_strings[Sentence.TOKENS], self._sentences[s].factor_strings[Sentence.TOKENS])

        return self


    @property
    def bert_embeddings(self):
        return self._bert


    @property
    def sentence_lens(self):
        return self._sentence_lens


    @property
    def mappings(self):
        """Return the mappings of the dataset.
        """
        return self._mappings


    @property
    def sentences(self):
        """Returns the sentences."""
        return self._sentences


    def next_batch(self, batch_size):
        """Return the next batch.
        Arguments:
            batch_size: batch size (int)
            Returns: Next batch, a dictionary with keys:
                "batch_factors": Python array with np.arrays
                    corresponding to Sentence.FACTORS:
                    factor x batch_size x tokens/nodes/edges x props
                "batch_factor_lens": Python array with np.arrays
                    corresponding to Sentence.FACTORS:
                    factor x batch_size x tokens/nodes/edges lens
            Optionally:
                "companion_batch_factors": companion batch_factors,
                "companion_batch_factor_lens": companion batch_factor_lens,
                "word2vec": batch of pretrained word2vec embeddings,
                    if self._word2vec != None (for tokens).
                "fasttext": batch of pretrained FastText embeddings,
                    if fasttext != None (for tokens).
                "bert": batch of pretrained BERT embeddings,
                    if self._bert != None (for tokens).
                "charseq_ids": TODO (for tokens).
                "charseqs": TODO (for tokens).
                "charseq_lens": TODO (for tokens).
                "ner": NE tagging (for tokens).
                "ner_lens": NE tagging lens (for tokens).
        """

        batch_size = min(batch_size, len(self._permutation))
        batch_perm = self._permutation[:batch_size]
        self._permutation = self._permutation[batch_size:]
        return self._next_batch(batch_perm)


    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._sentence_lens)) if self._shuffle_batches else np.arange(len(self._sentence_lens))
            return True
        return False


    def bert_embeddings_dim(self):
        if self._bert:
            return self._bert[0].shape[1]
        else:
            return 0
       

    def _next_batch(self, batch_perm):
        if self._max_sentence_len is not None:
            batch_perm = np.array([i for i in batch_perm if self._sentence_lens[i] < self._max_sentence_len], np.int32)
        batch_size = len(batch_perm)
        batch_dict = dict()

        # Compute batch factor lens and batch max factor lens
        # factors x batch_size x tokens/nodes/edges
        batch_factor_lens = [ self._factor_lens[f][batch_perm] for f in range(Sentence.FACTORS) ]
        max_batch_factor_lens = [ np.max(batch_factor_lens[f]) for f in range(Sentence.FACTORS) ]
        batch_dict["batch_factor_lens"] = batch_factor_lens
        max_sentence_len = np.max(batch_factor_lens[Sentence.TOKENS])

        # Fill batch factors
        batch_factors = [] # factors x batch_size x (props) x tokens/nodes/edges
        for f in range(Sentence.FACTORS):
            shape = [batch_size, max_batch_factor_lens[f]]
            if f in Sentence.FACTORS_2D: shape.append(len(self.mappings[f]))
            batch_factors.append(np.zeros(shape, np.int32))
            for i in range(batch_size):
                if batch_factor_lens[f][i]:
                    batch_factors[-1][i, :batch_factor_lens[f][i]] = self._sentences[batch_perm[i]].factors[f]

        batch_dict["batch_factors"] = batch_factors

        # Character-level data
        charseq_ids, charseqs, charseq_lens = [], [], []

        mapping = self._mappings[Sentence.TOKENS]
        charseq_ids = np.zeros([batch_size, max_sentence_len], np.int32)
        charseqs_map = {}
        charseqs = []
        charseq_lens = []
        for i in range(batch_size):
            for j, charseq_id in enumerate(self._sentences[batch_perm[i]].charseq_ids):
                if charseq_id not in charseqs_map:
                    charseqs_map[charseq_id] = len(charseqs)
                    charseqs.append(mapping.charseqs[charseq_id])
                charseq_ids[i, j] = charseqs_map[charseq_id]

        charseq_lens = np.array([len(charseq) for charseq in charseqs], np.int32)
        batch_charseqs = np.zeros([len(charseqs), np.max(charseq_lens)], np.int32)
        for i in range(len(charseqs)):
            batch_charseqs[i, :charseq_lens[i]] = charseqs[i]
        batch_dict["charseq_ids"] = charseq_ids
        batch_dict["charseqs"] = batch_charseqs
        batch_dict["charseq_lens"] = charseq_lens

        # Pretrained word embeddings for tokens
        if self._word2vec:
            we_size = self._word2vec.vectors.shape[1] # get pretrained WEs dimension
            wes = np.zeros([batch_size, max_sentence_len, we_size], np.float32)
            for i in range(batch_size):
                for j, word in enumerate(self._sentences[batch_perm[i]].factor_strings[Sentence.TOKENS]):
                    if word in self._word2vec:
                        wes[i, j] = self._word2vec[word]
                    elif word.lower() in self._word2vec:
                         wes[i, j] = self._word2vec[word.lower()]
            batch_dict["word2vec"] = wes 

        # Fasttext word embeddings for tokens
        if self._fasttext:
            we_size = self._fasttext.get_dimension() # get pretrained WEs dimension
            wes = np.zeros([batch_size, max_sentence_len, we_size], np.float32)
            for i in range(batch_size):
                for j, word in enumerate(self._sentences[batch_perm[i]].factor_strings[Sentence.TOKENS]):
                    wes[i, j] = self._fasttext.get_word_vector(word)
            batch_dict["fasttext"] = wes 

        # Pretrained BERT embeddings for tokens
        if self._bert:
            we_size = self.bert_embeddings_dim()
            batch_bert = np.zeros([batch_size, max_sentence_len, we_size], np.float32)
            for i in range(batch_size):
                batch_bert[i, 1:self._bert[batch_perm[i]].shape[0] + 1] = self._bert[batch_perm[i]]
            batch_dict["bert"] = batch_bert

        # Original sentences
        batch_dict["sentences"] = [self._sentences[i] for i in batch_perm]

        return batch_dict


    def write(self, filename=None):
        if filename:
            fw = open(filename, "w", encoding="utf-8")
        else:
            fw = sys.stdout
        for sentence in self._sentences:
            print(sentence.write(self._mappings), file=fw)
