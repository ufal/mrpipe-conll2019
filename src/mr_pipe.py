#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""MPipe: graph to graph neural network for MRP 2019."""

import json
import os
import re
import sys

import numpy as np
import tensorflow as tf

import mtool_evaluate
from mrp_dataset import MRPDataset
from network import Network

DATASETS = ["train", "dev", "bdev", "lpss", "test"]

if __name__ == "__main__":
    import argparse
    import datetime

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_aggregation", default=2, type=int, help="Batch aggregation.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--beta_2", default=0.98, type=float, help="Beta 2.")
    for dataset in DATASETS:
        parser.add_argument("--bert_{}".format(dataset), default=None, type=str, help="BERT {} data.".format(dataset))
    for dataset in DATASETS:
        parser.add_argument("--companion_{}".format(dataset), default=None, type=str, help="Companion {} data.".format(dataset))
    parser.add_argument("--cle_dim", default=256, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--decoder_iterations", default=1, type=int, help="Decoder iterations.")
    parser.add_argument("--deprel_dim", default=1024, type=int, help="Deprel dim.")
    parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate.")
    parser.add_argument("--edge_dim", default=1024, type=int, help="Edge dim.")
    parser.add_argument("--encoder_dim", default=512, type=int, help="RNN cell dimension.")
    parser.add_argument("--encoder_rnn_layers", default=1, type=int, help="Number of hidden layers.")
    parser.add_argument("--epochs", default="10:1e-3", type=str, help="Epochs and learning rates.")
    parser.add_argument("--exp", default=None, type=str, help="Experiment name.")
    parser.add_argument("--fasttext", default=None, type=str, help="FastText model path.")
    parser.add_argument("--highway", default=False, action="store_true", help="Highway networks on edges.")
    parser.add_argument("--jamr", default=None, type=str, help="Jamr alignments.")
    parser.add_argument("--max_sentences", default=None, type=int, help="Number of training sentences (for debugging).")
    parser.add_argument("--mode", default=None, type=str, help="Mode: dm|eds|psd|ucca|amr.")
    for dataset in DATASETS:
        parser.add_argument("--ner_{}".format(dataset), default=None, type=str, help="NER {} data.".format(dataset))
    parser.add_argument("--no_anchors", default=False, action="store_true", help="Predict anchors.")
    parser.add_argument("--predict", default=None, type=str, help="Predict using the passed model.")
    parser.add_argument("--predict_dataset", default=None, type=str, help="Which dataset to predict.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    for dataset in DATASETS:
        parser.add_argument("--{}".format(dataset), default=None, type=str, help="{} data.".format(dataset))
    parser.add_argument("--we_dim", default=512, type=int, help="Word embedding dimension.")
    parser.add_argument("--word_dropout", default=0.2, type=float, help="Word dropout")
    parser.add_argument("--word2vec", default=None, type=str, help="Word2vec model path.")
    args = parser.parse_args()
    if args.predict:
        # Load saved options from the model
        with open("{}/options.json".format(args.predict.split("@")[0]), mode="r") as options_file:
            args = argparse.Namespace(**json.load(options_file))
        parser.parse_args(namespace=args)
    else:
        assert args.train
        if args.mode in ("dm", "psd"):
            args.decoder_iterations = 1
            args.no_anchors = True

        # Create logdir name
        logargs = {}
        for k, v in vars(args).items():
            if k in ["exp"]: continue
            if k.endswith(("dev", "lpss", "test")): continue
            if k.startswith(("train", "predict", "companion")): continue
            if k in ["bert_train", "ner_train", "jamr", "word2vec"]:
                logargs[k] = 1 if v else 0
                continue
            if k in ["fasttext"]:
                logargs[k] = os.path.basename(v)[:12] if v else 0
                continue
            logargs[k] = v

        args.logdir = "logs/{}-{}-{}".format(
            args.exp or os.path.basename(__file__),
            datetime.datetime.now().strftime("%y%m%d_%H%M%S"),
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), re.sub("^.*/", "", value) if type(value) == str else value)
                      for key, value in sorted(logargs.items())))
        )
        args.logdireval = os.path.join(args.logdir, "eval")
        os.makedirs(args.logdireval)

        # Dump passed options to allow future prediction.
        with open("{}/options.json".format(args.logdir), mode="w") as options_file:
            json.dump(vars(args), options_file, sort_keys=True)

    # Postprocess args
    args.epochs = [(int(epochs), float(lr)) for epochs, lr in (epochs_lr.split(":") for epochs_lr in args.epochs.split(","))]

    # Set limits and random numbers
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Load pretrained word embeddings
    if args.word2vec:
        import word2vec
        print("Loading word2vec word embeddings from file {}".format(args.word2vec), file=sys.stderr, flush=True)
        args.word2vec = word2vec.load(args.word2vec)
    if args.fasttext:
        import fasttext
        print("Loading FastText word embeddings from file {}".format(args.fasttext), file=sys.stderr, flush=True)
        args.fasttext = fasttext.load_model(args.fasttext)

    # Load the data
    companion, ner, mrp = {}, {}, {}
    for dataset in DATASETS:
        if getattr(args, "companion_" + dataset):
            companion[dataset] = MRPDataset(
                getattr(args, "companion_" + dataset), args.mode, max_sentences=args.max_sentences, train=companion.get("train", None))
        if getattr(args, "ner_" + dataset):
            ner[dataset] = MRPDataset(
                getattr(args, "ner_" + dataset), args.mode, max_sentences=args.max_sentences, train=ner.get("train", None))
        if getattr(args, dataset):
            mrp[dataset] = MRPDataset(
                getattr(args, dataset), args.mode, max_sentences=args.max_sentences,
                max_sentence_len=100 if dataset == "train" else None,
                train=mrp.get("train", None), companion=companion.get(dataset, None), ner=ner.get(dataset, None), jamr=args.jamr,
                word2vec=args.word2vec, fasttext=args.fasttext, bert=getattr(args, "bert_" + dataset))
    del companion, ner

    # Construct network
    if args.predict:
        batches = 0
        while not mrp[args.predict_dataset].epoch_finished():
            mrp[args.predict_dataset].next_batch(args.batch_size)
            batches += 1

        networks = []
        for path in args.predict.split("@"):
            for i in range(batches - 1):
                mrp[args.predict_dataset].next_batch(args.batch_size)
            networks.append(Network(mrp["train"], args))
            networks[-1].predict(mrp[args.predict_dataset], args)
            networks[-1].load_weights(os.path.join(path, "checkpoint.h5"))

        predicted = Network.predict_ensemble(networks, mrp[args.predict_dataset], args)
        print(predicted, end="")
    else:
        network = Network(mrp["train"], args)

        for i, (epochs, learning_rate) in enumerate(args.epochs):
            for epoch in range(epochs):
                network.train_epoch(mrp["train"], learning_rate, args)

                network.save_weights(os.path.join(args.logdir, "checkpoint.h5"))
                for dataset in DATASETS[1:]:
                    if dataset in mrp:
                        network.evaluate(dataset, mrp[dataset], getattr(args, dataset), args)
