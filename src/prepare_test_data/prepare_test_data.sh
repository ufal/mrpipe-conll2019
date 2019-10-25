#!/bin/bash
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

set -e

for corpus in dm psd eds ucca amr; do
  echo $corpus
  ./prepare_test_data.py ../../data/mrp/2019/evaluation/input.mrp --framework=$corpus > ../../generated/data_split/$corpus/test.mrp
  wc -l ../../generated/data_split/$corpus/test.mrp
  grep "\"$corpus\"" ../../data/mrp/2019/evaluation/input.mrp | wc -l
done
