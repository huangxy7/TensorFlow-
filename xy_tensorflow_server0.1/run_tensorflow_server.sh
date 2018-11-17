#!/bin/bash
#A script to run the TensorFlow service.

CURRENT_DIR=`dirname "$0"`
#export TENSORFLOW_RELATED_HOME=`cd "$CURRENT_DIR/../.."; pwd`
export TENSORFLOW_RELATED_HOME=`cd "/home/"; pwd`
cd /home/pi/xy_deep_learning

PYTHON_BIN=`which python3.5`
if [ $? -ne 0 ]; then
    echo "Python 3.5 not found, quit"
    exit 1
fi


#WARM_UP_IMAGE_FILE_PATH=$TENSORFLOW_RELATED_HOME/resource/test-images/ubike.jpg
WARM_UP_IMAGE_FILE_PATH=$TENSORFLOW_RELATED_HOME/pi/xy_deep_learning/mobike.jpg

# start the service
$PYTHON_BIN $TENSORFLOW_RELATED_HOME/pi/xy_deep_learning/tensorflow_service.py --model_dir $TENSORFLOW_RELATED_HOME/pi/xy_deep_learning/ --warm_up_image_file $WARM_UP_IMAGE_FILE_PATH
