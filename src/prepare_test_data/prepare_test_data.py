#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Splits test data by given framework."""

from __future__ import print_function

import json
import os
import re
import sys

USAGE = "Usage: ./split_data.py mrp --framework=dm|psd|eds|ucca|amr"

FLAVORS = {"dm": 0, "psd": 0, "eds": 1, "ucca": 1, "amr": 2}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mrp", default=None, type=str, help="MRP test filename.")
    parser.add_argument("--framework", default=None, type=str, help="dm|psd|esd|ucca|amr")
    args = parser.parse_args()

    with open(args.mrp, mode="r", encoding="utf-8") as fw:
        for line in fw:
            data = json.loads(line)
            if args.framework in data["targets"]:
                del data["targets"]
                data["flavor"] = FLAVORS[args.framework]
                data["framework"] = args.framework
                print(json.dumps(data))
