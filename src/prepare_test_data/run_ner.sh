#!/bin/bash
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

set -e

NE_TAGGER=../../illinois_ne_tagger/illinois-ner/
OUTPUT=../../generated/companion/illinois
DATA=../../generated/data_split

[ $# -ge 1 ] || { echo Usage: $0 corpus >&2; exit 1; }

corpus=$1

echo $DATA/$corpus/test.mrp
../illinois_ne_tagger/make_ner_input.py $DATA/$corpus/test.mrp > $NE_TAGGER/input/$corpus-test.txt
(cd $NE_TAGGER; java -Xmx4g -classpath "dist/*:lib/*:models/*" -Xmx8g edu.illinois.cs.cogcomp.ner.NerTagger -annotate input output config/ner.properties)
../illinois_ne_tagger/ner_output_to_mrp.py --ner $NE_TAGGER/output/$corpus-test.txt --mrp $DATA/$corpus/test.mrp --mode $corpus > $OUTPUT/$corpus-test-4class.mrp
