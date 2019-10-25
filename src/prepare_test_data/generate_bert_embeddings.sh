#!/bin/bash
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Generates BERT embeddings for test data.

PYTHON=~/venv/tf-1.12-cpu/bin/python3
DATA=../../generated/data_split
OUTPUT_DIR=../../generated/bert

submit_bert() {
  corpus=$1
 
  f=$DATA/$corpus/test.mrp
  if [ ! -f $f ]; then echo "$f does not exist"; return 1; fi

  qsub -q cpu-troja.q@* -j y -l mem_free=8G,act_mem_free=8G,h_vmem=16G -pe smp 4 -N bert_${corpus}-test $PYTHON ../generate_embeddings/json_bert_embeddings.py $f $OUTPUT_DIR/${corpus}-test.npz --language=english --threads=4
}

for corpus in dm psd eds ucca amr; do
  submit_bert $corpus
done
