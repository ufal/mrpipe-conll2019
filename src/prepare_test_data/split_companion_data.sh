#!/bin/bash
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

COMPANION=../../data/mrp/2019/evaluation/udpipe.mrp
DATA=../../generated/data_split
OUTPUT=../../generated/companion/conllu

../companion/split_companion_data.py --inputs $DATA/dm/test.mrp,$DATA/psd/test.mrp,$DATA/eds/test.mrp,$DATA/ucca/test.mrp,$DATA/amr/test.mrp --companion $COMPANION --outputs $OUTPUT/dm-test.mrp,$OUTPUT/psd-test.mrp,$OUTPUT/eds-test.mrp,$OUTPUT/ucca-test.mrp,$OUTPUT/amr-test.mrp
