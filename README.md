#Jarvee
#Text detect for Traffic sign.
##This repository contains two code file.

The Baidu_process.cpp is to test this program on labeled images,show the localization precision,recall and f1_score.

Data provided by Baidu.

The SVM_train is to train the SVM which is used in Baidu_process.

These code can be run on VS2015+OPENCV 3.1+win10,that is my own enviroment.

I have to admit that these code are bad arranged.So I must briefly explain it.

##Baidu_process.cpp run in this order:

1,MSER (to get higher cover rate I use connect_pix in the code instead) extracts lots of pitches.

2,After filtering by geometric features,the left pitches are judged by SVM by their HOG features to eventually decided which pitches contains exact one char.

##SVM_train.cpp run in this order:

1,You need a data.txt file to get img data and rect data with in which contains exact one char.
For example 

file='E:\grade3\baidu_pic\plate\qualified\train\1.jpg'

top='104' left='31' width='12' height='11'

2,Pre run the Baidu.cpp, the pitches which are not in the labeled rect above are regarded as negative data.

3,As you see in code,after extracting HOG features,that's how to run SVM in opencv3 and set parameters.

4,The same method to read test set,and finally exam the SVM. 

#Do you think it is easy?Ah I believe it as well

And this easy algorithm can get 85% pitches coverage(the result of connect_pix).But the performance of SVM is very bad,recall 84 but precision 0.03 .I just store code here in case of PC crashing.....
