#!/usr/bin/env sh
set -e

/home/cpdp/lijiahui/caffe/caffe-master/.build_release/tools/caffe train \
    --solver=/home/cpdp/lijiahui/caffe/caffe-master/data/myself/solver.prototxt $@