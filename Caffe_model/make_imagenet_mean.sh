#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/home/cpdp/lijiahui/caffe/caffe-master/data/myself
DATA=/home/cpdp/lijiahui/caffe/caffe-master/data/myself
TOOLS=/home/cpdp/lijiahui/caffe/caffe-master/build/tools 

$TOOLS/compute_image_mean $EXAMPLE/imagenet_train_lmdb1\
  $DATA/imagenet_mean.binaryproto

echo "Done."
