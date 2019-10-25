#!/bin/bash

set -e

NE_TAGGER=../../illinois_ne_tagger/illinois-ner/
OUTPUT=../../generated/companion/illinois
DATA=../../generated/data_split
SAMPLE=../../mtool/data/sample

[ $# -ge 2 ] || { echo Usage: $0 corpus name >&2; exit 1; }

corpus=$1
name=$2

case $name in 
  sample) 
    echo $SAMPLE/$corpus/wsj.mrp
    ./make_ner_input.py $SAMPLE/$corpus/wsj.mrp > $NE_TAGGER/input/$corpus-$name.txt
    (cd $NE_TAGGER; java -Xmx4g -classpath "dist/*:lib/*:models/*" -Xmx8g edu.illinois.cs.cogcomp.ner.NerTagger -annotate input output config/ner.properties)
    ./ner_output_to_mrp.py --ner $NE_TAGGER/output/$corpus-$name.txt --mrp $SAMPLE/$corpus/wsj.mrp --mode $corpus > $OUTPUT/$corpus-$name-4class.mrp;;
  train|dev)
    echo $DATA/$corpus/$name.mrp
  ./make_ner_input.py $DATA/$corpus/$name.mrp > $NE_TAGGER/input/$corpus-$name.txt
  (cd $NE_TAGGER; java -Xmx4g -classpath "dist/*:lib/*:models/*" -Xmx8g edu.illinois.cs.cogcomp.ner.NerTagger -annotate input output config/ner.properties)
  ./ner_output_to_mrp.py --ner $NE_TAGGER/output/$corpus-$name.txt --mrp $DATA/$corpus/$name.mrp --mode $corpus > $OUTPUT/$corpus-$name-4class.mrp;;
esac
