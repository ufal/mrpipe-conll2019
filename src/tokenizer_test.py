#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Tokenizer test."""


import argparse
import collections
import json
import sys

from tokenizer import *

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer", default="MorphoditaTokenizer()", type=str, help="Tokenizer to use")
args = parser.parse_args()

tokenizer = eval(args.tokenizer)

anchors, not_mapped, lens = 0, 0, []
not_mapped_dict = collections.defaultdict(lambda: [])
longer_dict = collections.defaultdict(lambda: 0)

for line in sys.stdin:
    data = json.loads(line)
    text = data["input"]

    tokenized_starts, tokenized_ends = {}, {}
    for i, (start, end) in enumerate(tokenizer.tokenize(text)):
        tokenized_starts[start] = i
        tokenized_ends[end] = i

    froms, tos = collections.defaultdict(lambda: 0), collections.defaultdict(lambda: 0)
    for node in data["nodes"]:
        if "anchors" in node:
            for anchor in node["anchors"]:
                start, end = anchor["from"], anchor["to"]

                anchors += 1
                if start not in tokenized_starts or end not in tokenized_ends:
                    not_mapped += 1
                    not_mapped_dict[text[start:end]].append(text)
                else:
                    difference = tokenized_ends[end] - tokenized_starts[start]
                    while difference >= len(lens): lens.append(0)
                    lens[difference] += 1
                    if difference:
                        token = []
                        for i in range(start, end):
                            if i in tokenized_starts: token.append("<")
                            token.append(text[i])
                            if i + 1 in tokenized_ends: token.append(">")
                        longer_dict["".join(token)] += 1

print("Anchors", anchors)
print("Missing", not_mapped, "{:.2f}%".format(100 * not_mapped / anchors))
print("Lens", *["{}: {} ({:.2f}%)".format(i, value, 100 * value / anchors) for i, value in enumerate(lens)])

for not_mapped, examples in sorted(not_mapped_dict.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]:
    print(not_mapped, len(examples), examples[0])

for longer, examples in sorted(longer_dict.items(), key=lambda kv: kv[1], reverse=True)[:10]:
    print(longer, examples)
