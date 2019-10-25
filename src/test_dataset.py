#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""MRP dataset test."""

import os
import sys

import mrp_dataset

if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--jamr", default=None, type=str, help="Jamr alignments.")
    parser.add_argument("--companion_dev", default=None, type=str, help="Companion dev data.")
    parser.add_argument("--companion_test", default=None, type=str, help="Companion test data.")
    parser.add_argument("--companion_train", default=None, type=str, help="Companion train data.")
    parser.add_argument("--dev", default=None, type=str, help="Dev data.")
    parser.add_argument("--ner_dev", default=None, type=str, help="NER dev data.")
    parser.add_argument("--ner_test", default=None, type=str, help="NER test data.")
    parser.add_argument("--ner_train", default=None, type=str, help="NER train data.")
    parser.add_argument("--max_sentences", default=None, type=int, help="Max sentences.")
    parser.add_argument("--mode", default=None, type=str, help="dm|eds|psd|ucca|amr")
    parser.add_argument("--test", default=None, type=str, help="Test data.")
    parser.add_argument("--train", default=None, type=str, help="Train data.")
    args = parser.parse_args()

    if args.mode not in [ "dm", "eds", "psd", "ucca", "amr" ]:
        raise ValueError("--mode must be one of dm|eds|psd|ucca|amr")

    ner_train = mrp_dataset.MRPDataset(args.ner_train, args.mode, max_sentences=args.max_sentences) if args.ner_train else None
    ner_dev = mrp_dataset.MRPDataset(args.ner_dev, args.mode, train=ner_train, max_sentences=args.max_sentences) if args.ner_dev else None
    ner_test = mrp_dataset.MRPDataset(args.ner_test, args.mode, train=ner_train, max_sentences=args.max_sentences) if args.ner_test else None

    companion_train = mrp_dataset.MRPDataset(args.companion_train, args.mode, max_sentences=args.max_sentences) if args.companion_train else None
    companion_dev = mrp_dataset.MRPDataset(args.companion_dev, args.mode, train=companion_train, max_sentences=args.max_sentences) if args.companion_dev else None
    companion_test = mrp_dataset.MRPDataset(args.companion_test, args.mode, train=companion_train, max_sentences=args.max_sentences) if args.companion_test else None

    train = mrp_dataset.MRPDataset(args.train, args.mode, companion=companion_train, jamr=args.jamr, ner=ner_train, max_sentences=args.max_sentences)
    if args.dev:
        dev = mrp_dataset.MRPDataset(args.dev, args.mode, train=train, companion=companion_dev, jamr=args.jamr, ner=ner_dev, max_sentences=args.max_sentences)
    if args.test:
        test = mrp_dataset.MRPDataset(args.test, args.mode, train=train, companion=companion_test, jamr=args.jamr, ner=ner_test, max_sentences=args.max_sentences)

    # Test batches generation
    while not train.epoch_finished():
        batch_dict = train.next_batch(64)

    if args.test:
        while not test.epoch_finished():
            batch_dict = test.next_batch(64)

    # Test trainset writing
    train.write()
