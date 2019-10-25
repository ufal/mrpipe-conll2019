#!/usr/bin/env python3
# conding=utf-8
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Split companion data by corpora to speed up reading."""

import json
import os
import sys

sys.path.append("..")
from mrp_dataset import MRPDataset

if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=None, type=str, help="Input MRP data filenames list, separated by commas.")
    parser.add_argument("--outputs", default=None, type=str, help="Output MRP data filenames list, separated by commas.")
    parser.add_argument("--companion", default=None, type=str, help="Companion data.")
    args = parser.parse_args()

    companion_sentences = dict()
    print("Reading companion data from file {}".format(args.companion), file=sys.stderr)
    with open(args.companion, mode="r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\r\n")
            data = json.loads(line)
            # Keep original line
            companion_sentences[data["id"]] = line

    output_filenames = args.outputs.split(",")
    for i, input_filename in enumerate(args.inputs.split(",")):
        print("Reading MRP data from file {} and writing companion data to file {}".format(input_filename, output_filenames[i]), file=sys.stderr)
        with open(input_filename, mode="r", encoding="utf-8") as file:
            with open(output_filenames[i], mode="w", encoding="utf-8") as fw:
                for line in file:
                    data = json.loads(line)
                    assert(data["id"] in companion_sentences), "Data sentence id {} not in companion data.".format(data["id"])
                    print(companion_sentences[data["id"]], file=fw)
