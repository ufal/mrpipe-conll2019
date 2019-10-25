#!/bin/bash

set -e

MTOOL=../mtool/main.py
CONLLU=../generated/companion/conllu
NER=../generated/companion/illinois
JAMR=../generated/companion/jamr.mrp

test_dataset_sample() {
  DATA=../mtool/data/sample
  corpus=$1

  echo $corpus
  ./test_dataset.py --train $DATA/$corpus/wsj.mrp --mode $corpus --companion_train=$CONLLU/$corpus-sample.mrp --jamr=$JAMR --ner_train=$NER/$corpus-sample-4class.mrp | \
  ./mtool_evaluate.py $DATA/$corpus/wsj.mrp /dev/stdin --limit_mces 100000 --limit_rrhc 4 --parallelize 4
}

test_dataset_full() {
  DATA=../generated/data_split
  corpus=$1

  echo $corpus
  ./test_dataset.py --train $DATA/$corpus/train.mrp --dev $DATA/$corpus/dev.mrp --test $DATA/$corpus/test.mrp --mode $corpus --companion_train=$CONLLU/$corpus-train.mrp --companion_dev=$CONLLU/$corpus-dev.mrp --companion_test=$CONLLU/$corpus-test.mrp --jamr=$JAMR --ner_train=$NER/$corpus-train-4class.mrp --ner_dev=$NER/$corpus-dev-4class.mrp --ner_test=$NER/$corpus-test-4class.mrp | \
  ./mtool_evaluate.py $DATA/$corpus/train.mrp /dev/stdin --limit_mces 100000 --limit_rrhc 4 --parallelize 16
}

for corpus in ${@:-dm psd eds ucca amr}; do
  test_dataset_sample $corpus
done

for corpus in ${@:-dm psd eds ucca amr}; do
  test_dataset_full $corpus
done
