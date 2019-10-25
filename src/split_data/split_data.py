#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Splits the MRP data to train/dev portion by 90/10."""

from __future__ import print_function

import json
import os
import re
import sys

USAGE = "Usage: ./split_data.py --datadir=<data_dir> --splitdir=<split_dir>"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default=None, type=str, help="Directory with MRP data.")
    parser.add_argument("--splitdir", default="../../generated/data_split", type=str, help="Output split directory.")
    parser.add_argument("--ratio", default=0.9, type=float, help="Train ratio.")
    args = parser.parse_args()

    if not args.datadir or not args.splitdir:
        raise ValueError(USAGE)
    
    # Create output split directory
    if not os.path.isdir(args.splitdir):
        os.mkdir(args.splitdir)

    for corpus in os.listdir(args.datadir):
        if not os.path.isdir("{}/{}".format(args.datadir, corpus)): continue # skip files

        print("Reading directory {}{}".format(args.datadir, corpus), file=sys.stderr)

        if not os.path.isdir("{}/{}".format(args.splitdir, corpus)):
            print("Creating directory {}{}".format(args.splitdir, corpus), file=sys.stderr)
            os.mkdir("{}{}".format(args.splitdir, corpus))

        print("Writing to train file {}{}/train.mrp".format(args.splitdir, corpus), file=sys.stderr)
        with open("{}{}/train.mrp".format(args.splitdir, corpus), "w", encoding="utf-8") as fw_train:
            print("Writing to dev file {}{}/dev.mrp".format(args.splitdir, corpus), file=sys.stderr)
            with open("{}{}/dev.mrp".format(args.splitdir, corpus), "w", encoding="utf-8") as fw_dev:
                for filename in os.listdir("{}/{}".format(args.datadir, corpus)):
                    print("Reading JSON file {}".format("{}{}/{}".format(args.datadir, corpus, filename), file=sys.stderr))
                    with open("{}/{}/{}".format(args.datadir, corpus, filename), "r", encoding="utf-8") as fr:
                        data = []
                        for line in fr:
                            data.append(json.loads(line))
                    n = len(data)
                    print("All:\t{}\t{}:{}".format(n, 0, n), file=sys.stderr)
                    print("Train:\t{}\t0:{}".format(len(data[:int(n*args.ratio)]), int(n*args.ratio)), file=sys.stderr)
                    print("Dev:\t{}\t{}:{}".format(len(data[int(n*args.ratio):]), int(n*args.ratio), n), file=sys.stderr)
                    
                    for sentence in data[:int(n*args.ratio)]:
                        print(json.dumps(sentence), file=fw_train)
                        
                    for sentence in data[int(n*args.ratio):]:
                        print(json.dumps(sentence), file=fw_dev)
        fw_train.close()
        fw_dev.close()
