#!/bin/bash

set -e

./split_data.py --datadir ../../data/mrp/2019/training/ --splitdir=../../generated/data_split/ --ratio=0.9
