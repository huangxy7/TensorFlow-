#!/bin/bash
# A script to send request to the TensorFlow service.
sleep 5s
#if [ $# -lt 1 ]; then
#    echo "Usage: <inference-image-file-path>"
#    exit 1
#fi
python get_pic.py #take photos
#INFERENCE_IMAGE_FILE_PATH=$1
amixer cset numid=3 1 #设置3.5音频输出 amixer cset numid=3 2为hdmi接口输出
# send request to the service
#cd /home/pi/xy_deep_learning
while true
do
resu=$(curl "http://127.0.0.1:8080/?image_path=1.jpg")
#curl "http://127.0.0.1:8080/?image_path=1.jpg"
echo $resu
re=${resu:0:5}
echo $re
if [[ "$re" == "nosss" ]];then
echo "meideng"
mpg123 null.mp3
elif [[ "$re" == "green" ]];then
echo "ludeng"
mpg123 green.mp3
elif [[ "$re" == "redss" ]];then
echo "hongdeng"
mpg123 red.mp3
fi
done
