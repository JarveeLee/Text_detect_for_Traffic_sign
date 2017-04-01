#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

EXAMPLE=/home/cpdp/lijiahui/caffe/caffe-master/data/myself  #生成的lmdb数据的存放地址
DATA=/home/cpdp/lijiahui/caffe/caffe-master/data/myself        #原始数据文件夹的存放地址
TOOLS=/home/cpdp/lijiahui/caffe/caffe-master/build/tools        #数据格式转换工具的存放地址

TRAIN_DATA_ROOT=/home/cpdp/lijiahui/caffe/caffe-master/data/myself/train   #训练数据的存放地址
VAL_DATA_ROOT=/home/cpdp/lijiahui/caffe/caffe-master/data/myself/val       #校验数据的存放地址

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=32
  RESIZE_WIDTH=32
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

rm -rf $EXAMPLE/imagenet_train_lmdb1
rm -rf $EXAMPLE/imagenet_val_lmdb1

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $EXAMPLE/imagenet_train_lmdb1

echo "Creating val lmdb..."



GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    $EXAMPLE/imagenet_val_lmdb1

echo "Done."