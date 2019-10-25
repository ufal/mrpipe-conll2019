#!/bin/bash

set -e

COMPANION=../../data/mrp/2019/companion/udpipe.mrp
DATA=../../generated/data_split
SAMPLE=../../mtool/data/sample
OUTPUT=../../generated/companion/conllu

./split_companion_data.py --inputs $DATA/dm/train.mrp,$DATA/dm/dev.mrp,$DATA/eds/train.mrp,$DATA/eds/dev.mrp,$DATA/psd/train.mrp,$DATA/psd/dev.mrp,$DATA/ucca/train.mrp,$DATA/ucca/dev.mrp,$DATA/amr/train.mrp,$DATA/amr/dev.mrp --companion $COMPANION --outputs $OUTPUT/dm-train.mrp,$OUTPUT/dm-dev.mrp,$OUTPUT/eds-train.mrp,$OUTPUT/eds-dev.mrp,$OUTPUT/psd-train.mrp,$OUTPUT/psd-dev.mrp,$OUTPUT/ucca-train.mrp,$OUTPUT/ucca-dev.mrp,$OUTPUT/amr-train.mrp,$OUTPUT/amr-dev.mrp
#./split_companion_data.py --inputs $DATA/dm/btrain.mrp,$DATA/dm/bdev.mrp,$DATA/eds/btrain.mrp,$DATA/eds/bdev.mrp,$DATA/psd/btrain.mrp,$DATA/psd/bdev.mrp,$DATA/ucca/btrain.mrp,$DATA/ucca/bdev.mrp,$DATA/amr/btrain.mrp,$DATA/amr/bdev.mrp --companion $COMPANION --outputs $OUTPUT/dm-btrain.mrp,$OUTPUT/dm-bdev.mrp,$OUTPUT/eds-btrain.mrp,$OUTPUT/eds-bdev.mrp,$OUTPUT/psd-btrain.mrp,$OUTPUT/psd-bdev.mrp,$OUTPUT/ucca-btrain.mrp,$OUTPUT/ucca-bdev.mrp,$OUTPUT/amr-btrain.mrp,$OUTPUT/amr-bdev.mrp

./split_companion_data.py --inputs $SAMPLE/dm/wsj.mrp,$SAMPLE/eds/wsj.mrp,$SAMPLE/psd/wsj.mrp,$SAMPLE/ucca/wsj.mrp,$SAMPLE/amr/wsj.mrp --companion $COMPANION --outputs $OUTPUT/dm-sample.mrp,$OUTPUT/eds-sample.mrp,$OUTPUT/psd-sample.mrp,$OUTPUT/ucca-sample.mrp,$OUTPUT/amr-sample.mrp
