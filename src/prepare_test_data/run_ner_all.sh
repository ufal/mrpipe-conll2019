#!/bin/bash
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

for corpus in dm psd eds ucca amr; do
  qsub -q cpu*@* -j y -N ner_$corpus-test -l mem_free=8G,act_mem_free=8G,h_data=16G bash run_ner.sh $corpus
done
