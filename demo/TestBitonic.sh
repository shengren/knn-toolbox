#!/bin/bash

set -v
./knnTest uniform11_d100_n100000_float32.bin uniform11_d100_n100_float32.bin 100000 100 128 512 0 100
./compare-knn-idx.py idx.txt sklearn-knn-idx_k512_nq100_nr100000.txt
