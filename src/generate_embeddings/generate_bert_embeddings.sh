#!/bin/bash

# Generates Flair and ELMo embeddings where available.

PYTHON=withcuda\ ~/venvs/tf-1.12-gpu/bin/python3
DATA=../../generated/data_split
OUTPUT_DIR=../../generated/bert

submit_bert() {
  corpus=$1
  dataset=$2
  context_sentences=$3
 
  f=$DATA/$corpus/$dataset.mrp
  
  if [ ! -f $f ]; then echo "$f does not exist"; return 1; fi

  qsub -q gpu-ms.q@* -j y -l gpu=1,mem_free=16G,h_data=20G -pe smp 4 -N bert_${corpus}-${context_sentences#0}${dataset} $PYTHON json_bert_embeddings.py $f $OUTPUT_DIR/${corpus}-${context_sentences#0}${dataset}.npz --language=english --threads=4 --mode=$corpus --context_sentences=$context_sentences
}

for dataset in train dev test lpss btrain bdev; do
  for corpus in dm ucca amr; do # psd eds
    submit_bert $corpus $dataset 0
  done
done
