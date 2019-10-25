#!/bin/bash

for corpus in dm eds psd ucca amr; do
  for name in train dev sample; do
    qsub -q cpu*@* -j y -N $corpus-$name -l mem_free=8G,act_mem_free=8G,h_data=16G bash run_ner.sh $corpus $name
  done
done
