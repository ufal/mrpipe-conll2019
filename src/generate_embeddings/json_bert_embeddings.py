#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Generates pretrained BERT embeddings from MRP JSON files."""

import argparse
import json
import re
import sys
import zipfile

import numpy as np

import bert_wrapper
sys.path.append("..")
from tokenizer import RuleBasedTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=str, help="Input JSON file")
    parser.add_argument("output_npz", type=str, help="Output NPZ file")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--casing", default=bert_wrapper.BertWrapper.CASING_UNCASED, help="Bert model casing")
    parser.add_argument("--context_sentences", default=0, type=int, help="Context sentences")
    parser.add_argument("--language", default=bert_wrapper.BertWrapper.LANGUAGE_MULTILINGUAL, help="Bert model language")
    parser.add_argument("--layer_indices", default="-1,-2,-3,-4", type=str, help="Bert model layers to average")
    parser.add_argument("--size", default=bert_wrapper.BertWrapper.SIZE_BASE, help="Bert model size")
    parser.add_argument("--threads", default=4, type=int, help="Threads to use")
    parser.add_argument("--with_cls", default=False, action="store_true", help="Return also CLS embedding")
    parser.add_argument("--mode", default=None, type=str, help="Mode: dm|eds|psd|ucca|amr.")
    args = parser.parse_args()
    args.layer_indices = list(map(int, args.layer_indices.split(",")))

    # Initialize tokenizer
    tokenizer = RuleBasedTokenizer(finegrained = True if args.mode and args.mode == "ucca" else False)

    # Load JSON file
    sentences = []
    with open(args.input_json, mode="r", encoding="utf-8") as json_file:
        for line in json_file:
            data = json.loads(line)
            sentences.append([])
            for token_range in tokenizer.tokenize(data["input"]):
                sentences[-1].append(data["input"][token_range[0]:token_range[1]])

    bert = bert_wrapper.BertWrapper(language=args.language, size=args.size, casing=args.casing, layer_indices=args.layer_indices,
                                    with_cls=args.with_cls, threads=args.threads, batch_size=args.batch_size,
                                    context_sentences=args.context_sentences)
    with zipfile.ZipFile(args.output_npz, mode="w", compression=zipfile.ZIP_STORED) as output_npz:
        for i, embeddings in enumerate(bert.bert_embeddings(sentences)):
            if (i + 1) % 100 == 0: print("Processed {}/{} sentences.".format(i + 1, len(sentences)), file=sys.stderr)
            with output_npz.open("arr_{}".format(i), mode="w") as embeddings_file:
                np.save(embeddings_file, embeddings)
    print("Done, all embeddings saved.", file=sys.stderr)
