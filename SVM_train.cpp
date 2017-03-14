#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <iostream>  
#include <string>  
#include <fstream>  
#include <iterator>  
#include <vector>
#include <list>
#include <opencv2/opencv.hpp>
#include<stdlib.h>
#include<stdio.h>
#include <windows.h>
using namespace std;
using namespace cv;
using namespace cv::ml;

int test_bool = 1,train_bool=0;
//Mat DataMat, labelsMat,testMat;
int image_rows = 20, image_cols = 20, class_num = 2;
Ptr<SVM> model = SVM::create();
Ptr<ANN_MLP> bp = ANN_MLP::create();
FILE* fp;
FILE* fp1;
FILE* fp2;

string ann_xml = "E:/grade3/baidu_map_ann_train/ann_train/ann_data/ann_xml.xml";
char* chars_folder_ = "E:/grade3/baidu_map_ann_train/ann_train/ann";
string expIm = "E:/grade3/baidu_map_ann_train/ann_train/ann_data/digits/0/4-3.jpg";

string chaos_path = "E:/grade3/baidu_map_ann_train/ann_train/ann_data/jiejing_files";
string path_dig_folder = "E:/grade3/baidu_map_ann_train/ann_train/ann_data/digits";
string path_chi_folder = "E:/grade3/baidu_map_ann_train/ann_train/ann_data/chinese_character";
char* train_chaos_path = "E:/grade3/baidu_map_ann_train/ann_train/ann_data/chaos_train.txt";
char* test_chaos_path= "E:/grade3/baidu_map_ann_train/ann_train/ann_data/chaos_test.txt";
char* train_chi_path = "E:/grade3/baidu_map_ann_train/ann_train/ann_data/chinese_character_train.txt";
char* test_chi_path = "E:/grade3/baidu_map_ann_train/ann_train/ann_data/chinese_character_test.txt";
char* train_dig_path = "E:/grade3/baidu_map_ann_train/ann_train/ann_data/digits_train.txt";
char* test_dig_path = "E:/grade3/baidu_map_ann_train/ann_train/ann_data/digits_test.txt";

string path_data= "E:/grade3/baidu_map_ann_train/ann_train/ann_data/data.txt";
string path_data_labels = "E:/grade3/baidu_map_ann_train/ann_train/ann_data/data_labels.txt";
string path_data_test = "E:/grade3/baidu_map_ann_train/ann_train/ann_data/data_test.txt";
string path_data_labels_test = "E:/grade3/baidu_map_ann_train/ann_train/ann_data/data_labels_test.txt";


std::vector<std::string> GetListFiles(const std::string& path, const std::string & exten, bool addPath)
{
	std::vector<std::string> list;
	list.clear();
	std::string path_f = path + "/" + exten;
	WIN32_FIND_DATAA FindFileData;
	HANDLE hFind;

	hFind = FindFirstFileA((LPCSTR)path_f.c_str(), &FindFileData);
	if (hFind == INVALID_HANDLE_VALUE) {
		return list;
	}
	else {
		do {
			if (FindFileData.dwFileAttributes == FILE_ATTRIBUTE_NORMAL ||
				FindFileData.dwFileAttributes == FILE_ATTRIBUTE_ARCHIVE ||
				FindFileData.dwFileAttributes == FILE_ATTRIBUTE_HIDDEN ||
				FindFileData.dwFileAttributes == FILE_ATTRIBUTE_SYSTEM ||
				FindFileData.dwFileAttributes == FILE_ATTRIBUTE_READONLY) {
				char* fname;
				fname = FindFileData.cFileName;

				if (addPath) {
					list.push_back(path + "/" + std::string(fname));
				}
				else {
					list.push_back(std::string(fname));
				}
			}
		} while (FindNextFileA(hFind, &FindFileData));

		FindClose(hFind);
	}

	return list;
}
std::vector<std::string> GetListFolders(const std::string& path, const std::string & exten, bool addPath)
{
	std::vector<std::string> list;
	std::string path_f = path + "/" + exten;
	list.clear();

	WIN32_FIND_DATAA FindFileData;
	HANDLE hFind;

	hFind = FindFirstFileA((LPCSTR)path_f.c_str(), &FindFileData);
	if (hFind == INVALID_HANDLE_VALUE) {
		return list;
	}
	else {
		do {
			if (FindFileData.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY &&
				strcmp(FindFileData.cFileName, ".") != 0 &&
				strcmp(FindFileData.cFileName, "..") != 0) {
				char* fname;
				fname = FindFileData.cFileName;

				if (addPath) {
					list.push_back(path + "/" + std::string(fname));
				}
				else {
					list.push_back(std::string(fname));
				}
			}
		} while (FindNextFileA(hFind, &FindFileData));

		FindClose(hFind);
	}

	return list;
}
std::vector<std::string> GetListFilesR(const std::string& path, const std::string & exten, bool addPath)
{
	std::vector<std::string> list = GetListFiles(path, exten, addPath);
	std::vector<std::string> dirs = GetListFolders(path, exten, addPath);

	std::vector<std::string>::const_iterator it;
	for (it = dirs.begin(); it != dirs.end(); ++it) {
		std::vector<std::string> cl = GetListFiles(*it, exten, addPath);
		list.insert(list.end(), cl.begin(), cl.end());
	}

	return list;
}

void getHOGFeatures(const Mat& image, Mat& features) {
	//HOG descripter
	HOGDescriptor * hog = new HOGDescriptor(cvSize(128, 64), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 3);  //these parameters work well
	std::vector<float> descriptor;

	// resize input image to (128,64) for compute
	Size dsize = Size(128, 64);
	Mat trainImg = Mat(dsize, CV_32S);
	resize(image, trainImg, dsize);

	//compute descripter
	hog->compute(trainImg, descriptor, Size(8, 8));

	//copy the result
	Mat mat_featrue(descriptor);
	mat_featrue.copyTo(features);
}

Mat compare_amplify(Mat img)
{
	Mat mat = img;
	//namedWindow("original");
	//imshow("original", mat);
	Mat mergeImg;//合并后的图像  
			 //用来存储各通道图片的向量  
	vector<Mat> splitBGR(mat.channels());
	//分割通道，存储到splitBGR中  
	split(mat, splitBGR);
	//对各个通道分别进行直方图均衡化  
	for (int i = 0; i<mat.channels(); i++)
		equalizeHist(splitBGR[i], splitBGR[i]);
	//合并通道  
	merge(splitBGR, mergeImg);

	//namedWindow("equalizeHist");
	//imshow("equalizeHist", mergeImg);
	return mergeImg;
}
vector<Mat> clustering(Mat src, int ClusterNum,int mod)
{
	int row = src.rows;
	int col = src.cols;
	unsigned long int size = row*col;

	Mat clusters(size, 1, CV_32SC1);	//clustering Mat, save class label at every location;

										//convert src Mat to sample srcPoint.
	Mat srcPoint(size, 1, CV_32FC3);

	Vec3f* srcPoint_p = (Vec3f*)srcPoint.data;//////////////////////////////////////////////
	Vec3f* src_p = (Vec3f*)src.data;
	unsigned long int i;

	for (i = 0; i < size; i++)
	{
		*srcPoint_p = *src_p;
		srcPoint_p++;
		src_p++;
	}
	Mat center(ClusterNum, 1, CV_32FC3);
	double compactness;//compactness to measure the clustering center dist sum by different flag
	compactness = kmeans(srcPoint, ClusterNum, clusters,
		cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 0.1), ClusterNum,
		KMEANS_PP_CENTERS, center);

	cout << "center row:" << center.rows << " col:" << center.cols << endl;
	for (int y = 0; y < center.rows; y++)
	{
		Vec3f* imgData = center.ptr<Vec3f>(y);
		for (int x = 0; x < center.cols; x++)
		{
			cout << imgData[x].val[0] << " " << imgData[x].val[1] << " " << imgData[x].val[2] << endl;
		}
		cout << endl;
	}


	double minH, maxH;
	minMaxLoc(clusters, &minH, &maxH);			//remember must use "&"
	cout << "H-channel min:" << minH << " max:" << maxH << endl;

	int* clusters_p = (int*)clusters.data;
	//show label mat
	Mat label(src.size(), CV_32SC1);
	int* label_p = (int*)label.data;
	//assign the clusters to Mat label
	for (i = 0; i < size; i++)
	{
		*label_p = *clusters_p;
		label_p++;
		clusters_p++;
	}

	Mat label_show;
	label.convertTo(label_show, CV_8UC1);
	normalize(label_show, label_show, 255, 0, CV_MINMAX);
	//imshow("label", label_show);



	map<int, int> count;		//map<id,num>
	map<int, Vec3f> avg;		//map<id,color>

								//compute average color value of one label
	for (int y = 0; y < row; y++)
	{
		const Vec3f* imgData = src.ptr<Vec3f>(y);
		int* idx = label.ptr<int>(y);
		for (int x = 0; x < col; x++)
		{

			avg[idx[x]] += imgData[x];
			count[idx[x]] ++;
		}
	}
	//output the average value (clustering center)
	//计算所得的聚类中心与kmean函数中center的第一列一致，
	//以后可以省去后面这些繁复的计算，直接利用center,
	//但是仍然不理解center的除第一列以外的其他列所代表的意思
	double dis = INT_MAX;
	int pt = 0;
	for (i = 0; i < ClusterNum; i++)
	{
		avg[i] /= count[i];
		if (avg[i].val[0]>0 && avg[i].val[1]>0 && avg[i].val[2]>0)
		{
			cout << i << ": " << avg[i].val[0] << " " << avg[i].val[1] << " " << avg[i].val[2] << " count:" << count[i] << endl;
			double tp_dis = sqrt((avg[i].val[0] - 255)*(avg[i].val[0] - 255) +
				(avg[i].val[1] - 255)*(avg[i].val[1] - 255) +
				(avg[i].val[2])*(avg[i].val[2]));
			if (tp_dis < dis)
			{
				dis = tp_dis;
				pt = i;
			}
		}
	}
	//show the clustering img;
	vector<Mat> ans;
	ans.clear();
	Mat showImg(src.size(), CV_32FC3);
	for (int idt = 0; idt < ClusterNum; idt++)
	{
		for (int y = 0; y < row; y++)
		{
			Vec3f* imgData = showImg.ptr<Vec3f>(y);
			int* idx = label.ptr<int>(y);
			for (int x = 0; x < col; x++)
			{
				int id = idx[x];
				if (id == idt)
				{
					imgData[x].val[0] = avg[id].val[0];
					imgData[x].val[1] = avg[id].val[1];
					imgData[x].val[2] = avg[id].val[2];
				}

			}
		}
		normalize(showImg, showImg, 1, 0, CV_MINMAX);
		//imshow("show", showImg);
		//waitKey(0);
		Mat tep = showImg.clone();
		ans.push_back(tep);
	}

	//imshow("show", showImg);
	//normalize(showImg, showImg, 1, 0, CV_MINMAX);
	//imshow("show", showImg);
	//imshow("label", label_show);
	//waitKey(0);
	if (mod == 1)
	{
		//ans.clear();
		Mat t0=ans[0], t1=ans[1];
		ans.clear();
		ans.push_back(label_show);
		//if (count[0] > count[1])ans.push_back(t0);else ans.push_back(t1);
	}
	return ans;
}
void close_GetPlate(Mat srcImage, Mat ori)
{
	
	Mat operate;
	srcImage.convertTo(operate, CV_8UC3,255.0);
	cvtColor(operate, operate, CV_BGR2GRAY); // 转为灰度图像  
	threshold(operate, operate, 50, 255, CV_THRESH_BINARY);
	operate = 255 - operate;
	//imshow("operate",operate);
	//waitKey(0);
	//return operate;
	//闭操作连通块区
	Mat ret;
	int ssii = 1;
	//morphologyEx(operate, ret, MORPH_CLOSE, Mat::ones(ssii, ssii, CV_8UC1));
	morphologyEx(operate, ret, MORPH_OPEN, Mat::ones(ssii, ssii, CV_8UC1));
	imshow("ret", ret);
	vector<vector<Point> > plate_contours;
	findContours(ret, plate_contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	// 候选车牌区域判断输出  
	//imshow("blu", ret_blu);
	Mat candidates;
	//if (plate_contours.size() > 3)return;
	for (size_t i = 0; i != plate_contours.size(); ++i)
	{
		// 求解最小外界矩形  
		Rect rect = boundingRect(plate_contours[i]);
		int height = rect.height;
		int width = rect.width;
		double ratio = double(height) / width;
		double area = double(height) * width;
		//if (!(ratio > 0.3&&ratio < 10/3))continue;
		if (area < 50)continue;
		printf("ratio=%f,area=%f\n", ratio, area);
		//imshow("show", kec(rect));
		namedWindow("ori", CV_WINDOW_AUTOSIZE);
		imshow("ori", ori(rect));
		
		//judge_which(kec(rect));
		int c = waitKey(0);
		if (c -48==1)
		{

		}
		if (c-48==0)
		{
			destroyWindow("ori");
			return ;
		}
		destroyWindow("ori");
	}

	//return  candidates;
}
Mat mserFeature(cv::Mat srcImage)
{
	
	// HSV空间转换
	cv::Mat gray, gray_neg;
	cv::Mat hsi;
	cv::cvtColor(srcImage, hsi, CV_BGR2HSV);
	// 通道分离
	std::vector<cv::Mat> channels;
	cv::split(hsi, channels);
	// 提取h通道
	gray = channels[1];
	// 灰度转换 
	cv::cvtColor(srcImage, gray, CV_BGR2GRAY);
	// 取反值灰度
	gray_neg = 255 - gray;
	std::vector<std::vector<cv::Point> > regContours;
	std::vector<std::vector<cv::Point> > charContours;

	// 创建MSER对象
	cv::Ptr<cv::MSER> mesr1 = cv::MSER::create(2, 1, 300, 0.001, 0.1);
	cv::Ptr<cv::MSER> mesr2 = cv::MSER::create(2, 1, 300, 0.001, 0.1);


	std::vector<cv::Rect> bboxes1;
	std::vector<cv::Rect> bboxes2;
	// MSER+ 检测
	mesr1->detectRegions(gray, regContours, bboxes1);
	// MSER-操作
	mesr2->detectRegions(gray_neg, charContours, bboxes2);

	cv::Mat mserMapMat = cv::Mat::zeros(srcImage.size(), CV_8UC1);
	cv::Mat mserNegMapMat = cv::Mat::zeros(srcImage.size(), CV_8UC1);

	for (int i = (int)regContours.size() - 1; i >= 0; i--)
	{
		// 根据检测区域点生成mser+结果
		const std::vector<cv::Point>& r = regContours[i];
		for (int j = 0; j < (int)r.size(); j++)
		{
			cv::Point pt = r[j];
			mserMapMat.at<unsigned char>(pt) = 255;
		}
	}
	// MSER- 检测
	for (int i = (int)charContours.size() - 1; i >= 0; i--)
	{
		// 根据检测区域点生成mser-结果
		const std::vector<cv::Point>& r = charContours[i];
		for (int j = 0; j < (int)r.size(); j++)
		{
			cv::Point pt = r[j];
			mserNegMapMat.at<unsigned char>(pt) = 255;
		}
	}
	// mser结果输出
	cv::Mat mserResMat;
	// mser+与mser-位与操作
	mserResMat = mserMapMat & mserNegMapMat;
	cv::imshow("mserMapMat", mserMapMat);
	cv::imshow("mserNegMapMat", mserNegMapMat);
	cv::imshow("mserResMat", mserResMat);
	// 闭操作连接缝隙
	cv::Mat mserClosedMat;
	cv::morphologyEx(mserResMat, mserClosedMat,
		cv::MORPH_CLOSE, cv::Mat::ones(1, 1, CV_8UC1));
	return mserClosedMat;
}
/*
void ann_train(Mat trainData,Mat trainLabels,int iter)
{
	Ptr<ANN_MLP> bp = ANN_MLP::create();
	Mat samples;
	trainData.convertTo(samples, CV_32F);
	cv::Mat train_classes =
		cv::Mat::zeros((int)trainLabels.rows, class_num, CV_32F);
	cout << "converting data" << endl;
	for (int i = 0; i < train_classes.rows; ++i) {
		int tk = trainLabels.at<float>(i, 0);
		train_classes.at<float>(i, tk) = 1.f;
	}

	if (train_bool == 1)
	{
		Mat layerSizes = (Mat_<int>(1, 3) << image_rows*image_cols, 
			int(image_rows*image_cols / 2), class_num);
		bp->setLayerSizes(layerSizes);
		bp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);
		bp->setTrainMethod(cv::ml::ANN_MLP::TrainingMethods::BACKPROP);
		bp->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, iter, 0.0001));
		bp->setBackpropWeightScale(0.1);
		bp->setBackpropMomentumScale(0.1);
		Ptr<TrainData> tdata = TrainData::create(samples, 
		cv::ml::SampleTypes::ROW_SAMPLE,train_classes);
		cout << "Training ANN model, please wait..." << endl;
		bp->train(tdata);
		//long end = getTimestamp();
		cout << "Completed!" << endl;
		bp->save(ann_xml);
	}

	//system("pause");
	if (test_bool == 1)
	{
		bp = ml::ANN_MLP::load<ml::ANN_MLP>(ann_xml);
		cv::Mat output((int)trainLabels.rows, class_num, CV_32F);

		bp->predict(samples, output);
		
		float acc = 0;
		for (int i = 0; i < trainLabels.rows; i++)
		{
			
			int jud=-1, rea=-1;
			float judv = -INT_MAX, reav = -INT_MAX;
			for (int j = 0; j < class_num; j++)
			{
				if (output.at<float>(i, j) > judv)
				{
					jud = j;
					judv = output.at<float>(i, j);
					
				}
				if (train_classes.at<float>(i, j) > reav)
				{
					rea = j;
					reav = train_classes.at<float>(i, j);
				}
				cout << judv << "::" << train_classes.at<float>(i, j) << "!!";
				
			}
			//if (rea == 2)system("pause");
			cout << endl;
			if (jud == rea)acc = acc + 1;
		}
		float rate = acc / trainLabels.rows;
		cout << "acc=" << acc << " rate="<<rate<<endl;
		//for (int i = 0; i < 400; i++)cout << samples.at<float>(0, i); cout << endl;
	}

}
*/
/*
void data_oga(string path,int select)
{
	cout << path << endl;
	Mat img = imread(path, CV_GRAY2BGR);
	resize(img, img, Size(20, 20), 0, 0, CV_INTER_LINEAR);
	cvtColor(img, img, CV_BGR2GRAY);
	threshold(img, img, 130, 255, THRESH_BINARY);
	//cvtColor(img,img,gray)
	//imshow("ori", img);
	//waitKey(0);
	int ROWS = img.rows, COLS = img.cols;

	for (int i = 0; i<ROWS; i++)
	{
		for (int j = 0; j<COLS; j++)
		{
			//printf("i=%d,j=%d,data=%d\n", i, j, img.data[i, j]);
			int tk = img.data[i, j] > 0 ? 1 : 0;
			if (select == 1)fprintf(fp1, "%d ", tk);
			if (select == 2)fprintf(fp2, "%d ", tk);
		}
	}
	if (select == 1)fprintf(fp1, "\n");
	if (select == 2)fprintf(fp2, "\n");
	img.release();
}
int LoadData(string fileName, cv::Mat& matData, int matRows = 0, int matCols = 0, int matChns = 0)
{
	int retVal = 0;

	// 打开文件  
	ifstream inFile(fileName.c_str(), ios_base::in);
	if (!inFile.is_open())
	{
		cout << "读取文件失败" << endl;
		retVal = -1;
		return (retVal);
	}

	// 载入数据  
	istream_iterator<float> begin(inFile);    //按 float 格式取文件数据流的起始指针  
	istream_iterator<float> end;          //取文件流的终止位置  
	vector<float> inData(begin, end);      //将文件数据保存至 vector 中  
	cv::Mat tmpMat = cv::Mat(inData);       //将数据由 vector 转换为 cv::Mat  
	
											// 输出到命令行窗口  
											//copy(vec.begin(),vec.end(),ostream_iterator<double>(cout,"\t"));   

											// 检查设定的矩阵尺寸和通道数  
	size_t dataLength = inData.size();
	//1.通道数  
	if (matChns == 0)
	{
		matChns = 1;
	}
	//2.行列数  
	if (matRows != 0 && matCols == 0)
	{
		matCols = dataLength / matChns / matRows;
	}
	else if (matCols != 0 && matRows == 0)
	{
		matRows = dataLength / matChns / matCols;
	}
	else if (matCols == 0 && matRows == 0)
	{
		matRows = dataLength / matChns;
		matCols = 1;
	}
	//3.数据总长度  
	if (dataLength != (matRows * matCols * matChns))
	{
		cout << "读入的数据长度 不满足 设定的矩阵尺寸与通道数要求，将按默认方式输出矩阵！" << endl;
		retVal = 1;
		matChns = 1;
		matRows = dataLength;
	}

	// 将文件数据保存至输出矩阵  
	matData = tmpMat.reshape(matChns, matRows).clone();

	return (retVal);
}
*/
string num2str(double i)
{
	stringstream ss;
	ss << i;
	return ss.str();
}
/*
void train_model()
{
	test_bool = 1, train_bool = 0;
	Mat Data, Labels;
	if (train_bool == 1)
	{
		cout << "loading data...." << endl;
		LoadData(path_data_labels, Labels, 7033, 1, 0);
		LoadData(path_data, Data, 7033, 400, 0);
	}
	else
	{
		cout << "loading data...." << endl;
		LoadData(path_data_labels_test, Labels, 1900, 1, 0);
		LoadData(path_data_test, Data, 1900, 400, 0);
	}

	//cout<<trainingData.data[]

	cout << "loading completed! Start training" << endl;
	ann_train(Data, Labels, 1500);
	cout << "training complete" << endl;
}
*/
/*
void oga_data_dig_chi()
{
	string data_set;
	char *train_data=NULL, *test_data=NULL;
	int mod = 3;
	if (mod == 1)
	{
		data_set = path_dig_folder;
		train_data = train_dig_path;
		test_data = test_dig_path;
	}
	if(mod==2)
	{
		data_set = path_chi_folder;
		train_data = train_chi_path;
		test_data = test_chi_path;
	}
	if (mod == 3)
	{
		data_set = chaos_path;
		train_data = train_chaos_path;
		test_data = test_chaos_path;
	}

	string exten = "*";
	bool addPath = true; //false  

						 // 遍历指定文件夹下的所有文件，包括指定文件夹内的文件夹  
	vector<string> allfilenames = GetListFilesR(data_set, exten, addPath);

	cout << "all file names: " << endl;

	cout << allfilenames.size() << endl;

	system("pause");
	fp1 = fopen(train_data, "w");
	fp2 = fopen(test_data, "w");
	for (int i = 0; i < allfilenames.size(); i++)
		data_oga(allfilenames[i], i % 5 == 0 ? 1 : 2);
	//cout << "    " << allfilenames[i] << endl;

	fclose(fp1);
	fclose(fp2);

}
*/
Mat extractSobelFeature(Mat img)
{
	Mat img_loca;
	Mat ampli= compare_amplify(img);
	getHOGFeatures(ampli, img_loca);
	img_loca=img_loca.reshape(1, 1);
	//imshow("ori", img);
	//imshow("ampli", ampli);
	//imshow("HOG", img_loca);
	//waitKey(0);
	return img_loca;
	//img_loca = mserFeature(img);
	//img_loca = compare_amplify(img);
	//cvtColor(img_loca, img_loca, CV_RGB2GRAY);
	//threshold(img_loca, img_loca, 120, 255, 1);
	resize(img_loca, img_loca, Size(32, 32));
	int m_GaussianBlurSize = 3, binary_thres = 78;
	Mat img_blur, img_gray;
	GaussianBlur(img_loca, img_blur, 
		Size(m_GaussianBlurSize, m_GaussianBlurSize),0, 0, BORDER_DEFAULT);
	//imshow("ori", img);
	//imshow("img_loca", img_loca);
	//imshow("img_blur", img_blur);
	//waitKey(0);
	img_gray = img_blur.clone();
	//img_blur.convertTo(img_blur, CV_32FC3);
	//vector<Mat> skt = clustering(img_blur, 3, 1);
	//cvtColor(skt[0], skt[0], CV_RGB2GRAY);
	//threshold(skt[0], skt[0], 50, 255, CV_8U);
	//return skt[0];
	//img_gray = skt[0].clone();
	//img_blur.convertTo(img_blur, CV_8UC3,255.0);
	//imshow("img_blur", img_blur);
	//waitKey(0);
	//cvtColor(img_blur, img_gray, CV_RGB2GRAY);
	Mat grad[16];
	Mat merge;
	int dir_num = 2;
	for (int i = 0; i < dir_num; i++)
	{
		double kt = i / dir_num;
		Sobel(img_gray, grad[i*2], CV_8U, 1-kt, kt, 3, BORDER_DEFAULT);
		Sobel(img_gray, grad[i * 2+1], CV_8U, kt, 1-kt, 3, BORDER_DEFAULT);
		//imshow("grad1", grad[i * 2]);
		//imshow("grad2", grad[i * 2 + 1]);
		//waitKey(0);
		//threshold(grad[i*2], grad[i*2], binary_thres, 255, CV_8U);
		//threshold(grad[i * 2+1], grad[i * 2 + 1], binary_thres, 255, CV_8U);
		//convertScaleAbs(grad[i*2], grad[i*2]); 
		//convertScaleAbs(grad[i * 2+1], grad[i * 2+1]);
		//imshow(num2str(i), grad[i]);
		merge.push_back(grad[i*2]);
		merge.push_back(grad[i * 2+1]);
		//if(i==0)merge =  grad[i*2]|grad[i*2+1]; 
		//else merge = merge | grad[i * 2] | grad[i * 2+1];
		
	}
	//imshow("merge", merge);
	//waitKey(0);
	Mat sample;
	merge.reshape(1, 1).convertTo(sample,CV_32F);
	return sample;
}

void svm_data_collect()
{
	//Collect training data----------------------------------------------------------------
	string head = "E:/grade3/baidu_pic/plate/qualified/";
	float posi = 1.0, neg = -1.0;
	int rrrrt = 1;
	//int train_has=114741 ,train_no= 136374,test_has=390,test_no= 40731;
	int train_has = 12000, train_no = 12000, test_has = 390, test_no = 5000;
	int train_has_num = 11837, train_no_num = 261265, 
		test_has_num = 390, test_no_num = 40731;
	cout << "Input train_num" << endl;
	//cin >> train_num;
	//train_num = 500;
	
	Mat trainingImage;
	vector<int> trainingLabels;
	string filepath = head+"train/found";
	vector<string> allfilenames;
	
	//allfilenames = getFiles(filepath,true);
	//cout << "all file names: " << endl;
	//cout << allfilenames.size() << endl;
	
	for (int i = 0; i < train_has; i++)
	{
		string temp;
		temp = filepath + "/" + num2str(i%train_has_num) + ".jpg";
		allfilenames.push_back(temp);
	}
	//std::random_shuffle(allfilenames.begin(), allfilenames.end());
	for (int i = 0; i < allfilenames.size(); i++)
	{
		//cout << temp<<endl;
		Mat img = imread(allfilenames[i],1);
		//imshow("img_ori", img);
		img = extractSobelFeature(img);
		trainingImage.push_back(img);
		trainingLabels.push_back(posi);
	}
	cout << "Train_has_loaded" << endl;
	allfilenames.clear();
	filepath = head + "train/no";
	for (int i = 0; i < train_no; i++)
	{
		string temp;
		temp = filepath + "/" + num2str(i%train_no_num) + ".jpg";
		allfilenames.push_back(temp);
	}
	//std::random_shuffle(allfilenames.begin(), allfilenames.end());
	for (int i = 0; i < allfilenames.size(); i++)
	{
		Mat img = imread(allfilenames[i],1);
		img = extractSobelFeature(img);
		trainingImage.push_back(img);
		trainingLabels.push_back(neg);
	}
	cout << "Train_no_loaded" << endl;
	allfilenames.clear();
	Mat classes, trainingData;
	Mat(trainingImage).copyTo(trainingData);
	trainingData.convertTo(trainingData, CV_32F);
	Mat(trainingLabels).copyTo(classes);
	
	/*
	Mat trainingData,classes;
	Mat trainMat = imread(head + "trainMat.jpg", CV_32FC1);
	//imshow("data", trainingData);
	//waitKey(0);
	Mat trainLabels=imread(head + "trainLabels.jpg", CV_32FC1);
	//imshow("data", classes);
	//waitKey(0);
	Mat(trainMat).copyTo(trainingData);
	trainingData.convertTo(trainingData, CV_32FC1);
	Mat(trainLabels).copyTo(classes);
	cout << classes.at<int>(1880, 0) << endl;
	cout << "Train_data_loaded" << endl;
	*/
	//Training setting----------------------------------------------------------------------------
	//Ptr<SVM> model = SVM::create();
	//Ptr<ANN_MLP> bp = ANN_MLP::create();
	
	int train_mod = 1;//1 SVM 2 ANN
	
	if (train_mod == 1)
	{
		//Ptr<SVM> model=StatModel::load<SVM>("E:/grade3/baidu_pic/train/svm/valid/svm.xml");
		model->setType(cv::ml::SVM::C_SVC);
		model->setKernel(cv::ml::SVM::RBF);
		model->setDegree(0.1);
		// 1.4 bug fix: old 1.4 ver gamma is 1
		model->setGamma(0.1);
		model->setCoef0(0.1);
		model->setC(1);
		model->setNu(0.1);
		model->setP(0.1);
		model->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 20000, 0.0001));
		//model->SVM::load<SVM>("E:/grade3/baidu_pic/train/svm/valid/svm.xml");
		Ptr<TrainData> tdata=TrainData::create(trainingData, ROW_SAMPLE, classes);
		cout << "training please wait" << endl;
		model->train(tdata);
		
	}
	
	if (train_mod == 2)
	{
		cout << "converting data" << endl;
		Mat samples(trainingData.rows,trainingData.cols,CV_32F);
		for (int i = 0; i < samples.rows; i++)
		{
			for (int j = 0; j < samples.cols; j++)
			{
				samples.at<float>(i, j) = trainingData.at<float>(i, j);
			}
		}
		//Mat(trainingImage).convertTo(samples, CV_32FC1,1.0/255.0);
		Mat train_classes =
			Mat::zeros((int)classes.rows, 2, CV_32FC1);
		for (int i = 0; i < train_classes.rows; ++i) 
		{
			int tk = classes.at<int>(i, 0) > 0 ? 1:0 ;
			train_classes.at<float>(i, tk) = 1.f;
		}
		cout << train_classes << endl 
			<< train_classes.cols << " " << train_classes.rows << endl;
		system("pause");
		cout << trainingData.cols<<" "<<trainingData.rows<< endl;
		system("pause");
		Ptr<TrainData> tdata = TrainData::create(samples,
			ROW_SAMPLE, train_classes);
		Mat layer_sizes1(1, 3, CV_32SC1);
		layer_sizes1.at<int>(0) = trainingData.cols;
		layer_sizes1.at<int>(1) = int(trainingData.cols / 2);
		layer_sizes1.at<int>(2) = 2;
		bp->setLayerSizes(layer_sizes1);
		bp->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1, 1);
		bp->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 3, FLT_EPSILON));
		bp->setTrainMethod(ANN_MLP::BACKPROP, 0.001);
		cout << "Training ANN model, please wait..." << endl;
		bp->train(tdata);

	}
	cout << "Training completed!" << endl;
	//system("pause");
	//Checking---------------------------------------------------------------------------
	//cout << classes << endl;
	double precision = 0.0, recall = 0.0, f1_score = 0.0,
		pt_rt = 0.0, pt_rf = 0.0, pf_rt = 0.0;
	for (int i = 0; i < trainingData.rows; i++)
	{
		Mat sampleMat = trainingData.rowRange(i, i + 1).clone();
		float response;
		if (train_mod==1)response = model->predict(sampleMat);
		if (train_mod == 2)
		{
			Mat output((int)1, 2, CV_32F);
			bp->predict(sampleMat, output);
			response=output.at<float>(0, 0) > output.at<float>(0, 1) ? -1.0 : 1.0;
		}
		//cout << response << " " << classes.at<int>(i,0)<< endl;
		if (response == posi && classes.at<int>(i, 0) ==posi)
		{
			pt_rt = pt_rt + 1.0;
		}
		if (response ==neg && classes.at<int>(i, 0) == posi)
		{
			pf_rt = pf_rt + 1.0;
		}
		if (response ==posi && classes.at<int>(i, 0) == neg)
		{
			pt_rf = pt_rf + 1.0;
		}
	}
	precision = pt_rt / (pt_rt + pt_rf);
	recall= pt_rt / (pt_rt + pf_rt);
	f1_score = 2 * precision*recall / (precision + recall);
	cout << "Training Set:"<<"Precision="<<precision << " Recall=" << recall<<
		" F1_score="<< f1_score<< endl;
	trainingData.release();
	classes.release();
	allfilenames.clear();
	//system("pause");
	//Testing Set Data Collect----------------------------------------------------------------------
	cout << "Loading test data" << endl;
	Mat testImage;
	vector<int> testLabels;
	filepath = head + "test/has"; 
	for (int i = 0; i < test_has; i++)
	{
		string temp;
		temp = filepath + "/" + num2str(i%test_has_num) + ".jpg";
		allfilenames.push_back(temp);
	}
	std::random_shuffle(allfilenames.begin(), allfilenames.end());
	for (int i = 0; i < allfilenames.size(); i++)
	{
		Mat img = imread(allfilenames[i]);
		img = extractSobelFeature(img);
		testImage.push_back(img);
		testLabels.push_back(posi);
	}
	cout << "Test_has_loaded" << endl;
	allfilenames.clear();
	filepath = head + "test/no";
	for (int i = 0; i < test_no; i++)
	{
		string temp;
		temp = filepath + "/" + num2str(i%test_no_num) + ".jpg";
		allfilenames.push_back(temp);
	}
	for (int i = 0; i < allfilenames.size(); i++)
	{
		Mat img = imread(allfilenames[i],1);
		img = extractSobelFeature(img);
		testImage.push_back(img);
		testLabels.push_back(neg);
	}
	cout << "Test_no_loaded" << endl;
	allfilenames.clear();
	Mat test_classes, testData;
	Mat(testImage).copyTo(testData);
	testData.convertTo(testData, CV_32FC1);
	Mat(testLabels).copyTo(test_classes);
	//Checking-------------------------------------------
	precision = 0.0, recall = 0.0, f1_score = 0.0,
		pt_rt = 0.0, pt_rf = 0.0, pf_rt = 0.0;
	for (int i = 0; i < testData.rows; i++)
	{
		Mat sampleMat = testData.rowRange(i, i + 1).clone();
		float response;
		if (train_mod == 1)response = model->predict(sampleMat);
		if (train_mod == 2)
		{
			Mat output((int)1, class_num, CV_32F);
			bp->predict(sampleMat, output);
			response = output.at<float>(0, 0) > output.at<float>(0, 1) ? -1.0 : 1.0;
		}
		//cout << response << " " << classes.at<int>(i,0)<< endl;
		if (response == posi && test_classes.at<int>(i, 0) == posi)
		{
			pt_rt = pt_rt + 1.0;
		}
		if (response ==neg && test_classes.at<int>(i, 0) == posi)
		{
			pf_rt = pf_rt + 1.0;
		}
		if (response ==posi && test_classes.at<int>(i, 0) == neg)
		{
			pt_rf = pt_rf + 1.0;
		}
	}
	precision = pt_rt / (pt_rt + pt_rf);
	recall = pt_rt / (pt_rt + pf_rt);
	f1_score = 2 * precision*recall / (precision + recall);
	cout << "Testing Set:" << "Precision=" << precision << " Recall=" << recall <<
		" F1_score=" << f1_score << endl;
	model->save(head+"/svm.xml");
	system("pause");
}

void plate_split()
{

	string data_path = "E:/grade3/baidu_pic/train/svm/test/has";
	vector<string> allfile = GetListFilesR(data_path, "*", true);
	cout << allfile.size() << endl;
	int ite = 0;
	for (int i = 0; i < allfile.size(); i++)
	{
		string filename = allfile[i];
		Mat img = imread(filename, 1);
		string save_path = "E:/grade3/baidu_pic/train/svm/valid/test/has/"+num2str(i)+".jpg";
		imwrite(save_path, img);
		continue;
		Mat temp = img.clone();
		/*
		int  areaa = img.cols*img.rows;
		if ( areaa> 4000)
		{
			string save_path = "E:/grade3/baidu_pic/plate/valid/" + num2str(ite) + ".jpg";
			imwrite(save_path, img);
			cout << ite++ << " " << areaa << endl;
			continue;
		}
		else
		{
			continue;
		}
		*/
		GaussianBlur(img, img, Size(5, 5),5);
		imshow("ori_pic", temp);
		img.convertTo(img, CV_32FC3);
		vector<Mat>pic = clustering(img, 6,0);
		for (int j = 0; j < pic.size(); j++)
		{
			close_GetPlate(pic[j], temp);
		}
	}
	
}

void fp_char_cap()
{
	char* locate_path = "E:/grade3/baidu_pic/plate/plate_locate.txt";
	fp = fopen(locate_path, "r");
	char temp[1000];
	string temp_path;
	vector<Rect> rect_list;
	Mat img;
	int top, left, down, width, height,num_pic=0,num_rect = 0;
	double ratio = 0.0;
	while (fscanf(fp, "%s", temp) != EOF)
	{
		//printf("%s\n",temp);
		if (strcmp(temp, "<image") == 0)
		{
			//num_pic++;
			//printf("before %s\n", temp);
			fscanf(fp, " file='%s'>\n", temp);
			//printf("if %s\n", temp);
			for (int i = 0; i < strlen(temp) - 2; i++)temp_path = temp_path + temp[i];
			cout << temp_path << endl;
			img = imread(temp_path, 1);
			//imshow("img", img);
			//waitKey(0);
			//system("pause");
		}
		if (strcmp(temp, "<box") == 0)
		{
			fscanf(fp, " top='%d' left='%d' width='%d' height='%d'/>\n",
				&top, &left, &width, &height);
			
			//top = img.rows - top;
			//cout << img.cols << " " << img.rows << endl;
			if (top + height> img.rows)
			{
				height = img.rows - top;
			}
			if (top  < 0)top = 0;
			if (left + width > img.cols)width = img.cols - left;
			if (left < 0)left = 0;
			Rect rect(left, top, width, height);
			rect_list.push_back(rect);
			printf("%d %d %d %d\n", top, left, width, height);
			ratio += double(width) / height;
			//imshow("rect", img(rect));
			//waitKey(0);
		}
		if (strcmp(temp, "</image>") == 0)
		{
			if (rect_list.size() == 0)
			{
				//num_pic--;
				break;
			}
			string has_path = "E:/grade3/baidu_pic/plate/qualified/train/has/";
			string temp;
			for (int i = 0; i < rect_list.size(); i++)
			{
				cout << num_pic << endl;
				temp = has_path + num2str(num_rect++) + ".jpg";
				Mat save_pic;
				resize(img(rect_list[i]), save_pic, Size(32, 32));
				imwrite(temp, save_pic);
			}
			
			//num_rect += rect_list.size();
			
			rect_list.clear();
			temp_path.clear();
			//waitKey(0);
			//system("pause");
		}
	}
	ratio /= num_rect;
	cout << ratio << endl;
	fclose(fp);
	system("pause");
}

int main(int argc, const char* argv[]) 
{
	
	//oga_data_dig_chi();
	//imshow("1", img);
	//waitKey(0);
	//train_model();//ann
	svm_data_collect();
	//plate_split();
	//fp_char_cap();
	/*
	string pa = "E:/grade3/baidu_pic/plate/qualified/train/found";
	vector<string> file_list = GetListFilesR(pa, "*", 1);
	//int ssi = file_list.size();
	int replica = 100;
	for (int i = 0; i < 1880; i++)
	{
		string tk = pa + num2str(i) + ".jpg";
		Mat img = imread(tk, 1);
		//for (int jk = 1; jk <= replica; jk++)
		//{
			string temp = "E:/grade3/baidu_pic/plate/qualified/train/found/" + num2str(i+1440) + ".jpg";
			resize(img, img, Size(32, 32));
			imwrite(temp, img);
		//}
		cout << i << endl;
	}
	*/
	system("pause");
	return 0;
}