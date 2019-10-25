#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Reads Illinois NE tagger output and creates MRP file."""

from __future__ import print_function

import json
import re
import sys

sys.path.append("..")
from tokenizer import RuleBasedTokenizer 

USAGE = "Usage: ./ner_output_to_mrp.py filename"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ner", default=None, type=str, help="NE annotated text file.")
    parser.add_argument("--mrp", default=None, type=str, help="MRP file.")
    parser.add_argument("--mode", default=None, type=str, help="dm|eds|psd|ucca|amr")
    args = parser.parse_args()

    if args.mode not in [ "dm", "eds", "psd", "ucca", "amr" ]:
        raise ValueError("--mode must be one of dm|eds|psd|ucca|amr")

    tokenizer = RuleBasedTokenizer(finegrained = True if args.mode == "ucca" else False)
    
    datas = []
    with open(args.mrp, "r", encoding="utf-8") as mrp_file:
        for line in mrp_file:
            datas.append(json.loads(line))

    with open(args.ner, "r", encoding="utf-8") as ner_file:
        ner_input = ner_file.readline()
        ner_input = ner_input.rstrip("\r\n")

    n = len(datas)
    for d, data in enumerate(datas): 
        output_dict = dict()
        output_dict["id"] = data["id"]
        output_dict["flavor"] = 0
        output_dict["framework"] = "ner"
        output_dict["time"] = data["time"]
        output_dict["version"] = data["version"]
        output_dict["input"] = data["input"]
        output_dict["mode"] = "ner"

        ne_label = "O"
        nodes = []

        token_ranges = tokenizer.tokenize(data["input"])
        for i, token_range in enumerate(token_ranges):
            token = data["input"][token_range[0]:token_range[1]]

            # Consume token
            iterations = 0
            limit = len(token) + 1
            while token:
                if iterations >= limit: break

                # Fix some NE tagger bad fixes:
                if token == "`" and ner_input.startswith("'s "):
                    ner_input = "` s{}".format(ner_input[3:])
                elif token == "`" and ner_input.startswith("'"):
                    ner_input = "`{}".format(ner_input[1:])
                if token == "&" and not ner_input.startswith(("&", "[", "]")):
                    if i + 2 < len(token_ranges):
                       next_token = data["input"][token_ranges[i+1][0]:token_ranges[i+1][1]]
                       next_next_token = data["input"][token_ranges[i+2][0]:token_ranges[i+2][1]]
                       if next_token in ["quot"] and next_next_token == ";":
                           print("Found HTML entity {}, fixed NER input.".format(next_token), file=sys.stderr, flush=True)
                           print("Sentence: {}".format(data["input"]), file=sys.stderr, flush=True)
                           print("Original NER input: {}".format(ner_input[:100]), file=sys.stderr, flush=True)
                           ner_input = "&{} ;{}".format(next_token, ner_input[1:])
                           print("Fixed NER input: {}".format(ner_input[:100]), file=sys.stderr, flush=True)
                if token in ["''", "``"] and ner_input.startswith("\""):
                    print("Token {} found, but \" found in NER input.".format(token), file=sys.stderr, flush=True)
                    print("Sentence: {}".format(data["input"]), file=sys.stderr, flush=True)
                    print("Original NER input: {}".format(ner_input[:100]), file=sys.stderr, flush=True)
                    ner_input = "{}{}".format(token, ner_input[1:])
                    print("Fixed NER input: {}".format(ner_input[:100]), file=sys.stderr, flush=True)

                if ner_input.startswith("[") and token != "[":
                    m = re.match(r'\[(\w+)', ner_input)
                    ne_label = m.group(1)
                    ner_input = ner_input[1 + len(ne_label):]
                    ner_input = ner_input.lstrip()
                if ner_input.startswith("]") and token != "]":
                    ner_input = ner_input[1:]
                    ner_input = ner_input.lstrip()
                    ne_label = "O"
                for i in range(len(token)+1):
                    if token[:i] != ner_input[:i]:
                        i -= 1
                        break
                if ner_input[:i] == token[:i]:
                    ner_input = ner_input[i:]
                    ner_input = ner_input.lstrip()
                    token = token[i:]
                iterations += 1

            nodes.append({"id": len(nodes), "label": ne_label, "anchors": [{"from": token_range[0], "to": token_range[1]}]})

            if iterations >= limit and  d + 1 < len(datas):
                print("Unable to match token {} to NER input: {} after {} iterations, trying to skip the sentence.".format(token, ner_input[:100], iterations), file=sys.stderr, flush=True)
                first_token_range = tokenizer.tokenize(datas[d+1]["input"])[0]
                first_token = datas[d+1]["input"][first_token_range[0]:first_token_range[1]]
                index = ner_input.find(first_token)
                assert index != -1, "Unable to skip sentence."
                print("Found first token {} of the next sentence: {}, skipping current sentence in NER input: {}.".format(first_token, datas[d+1]["input"], ner_input[:100]), file=sys.stderr, flush=True)
                ner_input = ner_input[index:]
                token = ""
                ne_label = "O"
                # Finish sentence nodes before break
                for j in range(i+1, len(token_ranges)):
                    nodes.append({"id": len(nodes), "label": "O", "anchors": [{"from": token_ranges[j][0], "to": token_ranges[j][1]}]})
                break

        output_dict["nodes"] = nodes
        output_dict["edges"] = []
        output_dict["tops"] = []
        print(json.dumps(output_dict))
        output_dict = dict()
        if d % 100 == 0:
            print("Processed {} sentences of {}".format(d, n), file=sys.stderr, flush=True)

