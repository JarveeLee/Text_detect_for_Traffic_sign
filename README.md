# Jarvee

# Text detect for Traffic sign.

## This repository contains two code file.

The Baidu_process.cpp is to test this program on labeled images,show the localization precision,recall and f1_score.

Data provided by Baidu.

The SVM_train is to train the SVM which is used in Baidu_process.

The label tool is used to label your own data set.

These code can be run on VS2015+OPENCV 3.1+win10,that is my own enviroment.

I have to admit that these code are bad arranged.So I must briefly explain it.

## Baidu_process.cpp run in this order:

1,MSER (to get higher cover rate I use connect_pix in the code instead) extracts lots of pitches.

2,After filtering by geometric features,the left pitches are judged by SVM by their HOG features to eventually decided which pitches contains exact one char.

## SVM_train.cpp run in this order:

1,You need a Rect_list.txt file to get img data and rect data with in which contains exact one char.

Reading process is written in Process_d() function.

For example 

file='E:\grade3\baidu_pic\plate\qualified\train\sample.jpg'

top='104' left='31' width='12' height='11'

this rect stands for a part of this pic.

2,Pre run the Baidu.cpp, the pitches which are not in the labeled rect above are regarded as negative data.

3,As you see in code,after extracting HOG features,that's how to run SVM in opencv3 and set parameters.

4,The same method to read test set,and finally exam the SVM. 

# Do you think it is easy?Ah I believe it as well

And this easy algorithm can get 0.85 pitches coverage(the result of connect_pix).The performance of this process has Train/test precision 0.87/0.73 recall 0.85/0.68.It can be promoted by more features extracted and deeper neurual network.I store code here in case of PC crashing.....
tr
## If you want to run the code ,remember to change lots of absolute path by your own,and switch train_mod in SVM_train to 2 to generate ann.xml,and switch jud_mod in baidu_process to use it.
