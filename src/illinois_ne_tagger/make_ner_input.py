#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Prints raw data, one sentence per line."""

from __future__ import print_function

import json

USAGE = "Usage: ./make_ner_input.py filename"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", default=None, type=str, help="MRP data filename.")
    args = parser.parse_args()

    with open(args.filename, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            print(data["input"])
