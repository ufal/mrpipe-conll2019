#!/bin/bash

MTOOL=../mtool/main.py
DATA=../generated/data_split
CONLLU=../generated/companion/conllu
NER=../generated/companion/illinois
JAMR=../generated/companion/jamr.mrp
# TODO: Use word2vec model allowed in the closed track.
# WORD2VEC=../generated/word2vec
FASTTEXT=../generated/fasttext/crawl-300d-2M-subword.bin
BERT=../generated/bert

[ $# -ge 1 ] || { echo Usage: $0 cmdfile args... >&2; exit 1; }
if [ "$1" = "-f" ]; then
  force=1
  shift
fi
cmdfile="$1"; shift
args="$@"; shift

# Assume the following is set
# - mode
# - cs
# - size

[ -f "$cmdfile" -a -z "$force" ] && { echo File $cmdfile already exists! >&2; exit ; }

case "$cmdfile" in
  *cpu*) runner="../generated/venv-cpu/bin/python";;
  *) runner="withcuda ../generated/venv-gpu/bin/python";;
esac

for corpus in dm psd eds ucca amr; do
  corpusargs="--dropout=0.3 --encoder_rnn_layers=2"
  case $corpus in
    ucca) corpusargs="$corpusargs --epochs=15:1e-3,10:1e-4 --encoder_dim=1024";;
    *) corpusargs="$corpusargs --epochs=10:1e-3,5:1e-4 --encoder_dim=768";;
  esac
  case $corpus in
    eds) corpusargs="$corpusargs --decoder_iterations=3";;
    *) corpusargs="$corpusargs --decoder_iterations=2";;
  esac
  echo $runner mr_pipe.py --threads=4 --train=$DATA/$corpus/${mode}train.mrp --${mode}dev=$DATA/$corpus/${mode}dev.mrp --bdev=$DATA/$corpus/bdev.mrp --lpss=$DATA/$corpus/lpss.mrp --test=$DATA/$corpus/test.mrp --mode=$corpus --fasttext=$FASTTEXT --bert_train=$BERT/$corpus-$cs${mode}train$size.npz --bert_${mode}dev=$BERT/$corpus-$cs${mode}dev$size.npz --bert_bdev=$BERT/$corpus-${cs}bdev$size.npz --bert_lpss=$BERT/$corpus-${cs}lpss$size.npz --bert_test=$BERT/$corpus-${cs}test$size.npz --companion_train=$CONLLU/$corpus-${mode}train.mrp --companion_${mode}dev=$CONLLU/$corpus-${mode}dev.mrp --companion_bdev=$CONLLU/$corpus-bdev.mrp --companion_lpss=$CONLLU/$corpus-lpss.mrp --companion_test=$CONLLU/$corpus-test.mrp --jamr=$JAMR $corpusargs $args #--ner_train=$NER/$corpus-train-4class.mrp --ner_dev=$NER/$corpus-dev-4class.mrp --ner_test=$NER/$corpus-test-4class.mrp --word2vec=$WORD2VEC/$corpus.txt
done >$cmdfile

case $(hostname) in
  aic|gpu-node*|cpu-node*)
    echo qsub -q "gpu.q" -pe smp 2 -l gpu=1,mem_free=24G,h_data=32G -j y -o $cmdfile.\\\$TASK_ID.log -t 1-5 -tc 5 arrayjob_runner $cmdfile;;
  *)
    echo qsub -q "gpu-ms.q@dll*" -pe smp 4 -l gpu=1,gpu_ram=11G,mem_free=24G,h_data=32G -j y -o $cmdfile.\\\$TASK_ID.log -t 1-5 -tc 5 arrayjob_runner $cmdfile;;
esac

