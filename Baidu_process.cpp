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
//#define ClusterNum (3)

using namespace std;
using namespace cv;
using namespace cv::ml;

using namespace cv;
using namespace std;

//int threshval = 120;
FILE* fp;
FILE* fp1;

string filename = "E:/grade3/baidu_pic/9.jpg";
string ann_xml_path = "E:/grade3/baidu_map_ann_train/ann_train/ann_data/ann_xml.xml";
Ptr<SVM> model = StatModel::load<SVM>("E:/grade3/baidu_pic/plate/qualified/svm.xml");
Ptr<ANN_MLP> bp = StatModel::load<ANN_MLP>("E:/grade3/baidu_pic/plate/qualified/ann.xml");
int jud_mod = 2;
int loc_model = 1;
double pt_rt = 0.0, pf_rt = 0.0, pt_rf = 0.0, pf_rf = 0.0,
total_found = 0.0, total_plate = 0.0, total_reco_rate = 0.0, total_pic_num = 0.0,
success = 0.0, eighty_success = 0.0, total_found_rate=0.0;
int no_ite = 0, has_ite = 3320;
int step_l = 10;
double thres_s = 0.3;
double found_statistic[100];

void getHOGFeatures(const Mat& image, Mat& features) {
	//HOG descripter
	HOGDescriptor * hog = new HOGDescriptor(cvSize(32, 32), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 3);  //these parameters work well
	std::vector<float> descriptor;

	// resize input image to (128,64) for compute
	Size dsize = Size(32, 32);
	Mat trainImg = Mat(dsize, CV_32S);
	resize(image, trainImg, dsize);

	//compute descripter
	hog->compute(trainImg, descriptor, Size(8, 8));

	//copy the result
	Mat mat_featrue(descriptor);
	mat_featrue.copyTo(features);
}

string num2str(double i)
{
	stringstream ss;
	ss << i;
	return ss.str();
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
Mat extractSobelFeature(Mat img)
{
	Mat img_loca;
	Mat ampli = compare_amplify(img);
	getHOGFeatures(ampli, img_loca);
	img_loca = img_loca.reshape(1, 1);
	//imshow("ori", img);
	//imshow("ampli", ampli);
	//imshow("HOG", img_loca);
	//waitKey(0);
	return img_loca;
	//img_loca = mserFeature(img);
	img_loca = compare_amplify(img);
	cvtColor(img_loca, img_loca, CV_RGB2GRAY);
	threshold(img_loca, img_loca, 120, 255, 1);
	resize(img_loca, img_loca, Size(32, 32));
	int m_GaussianBlurSize = 1, binary_thres = 78;
	Mat img_blur, img_gray;
	GaussianBlur(img_loca, img_blur, Size(m_GaussianBlurSize, m_GaussianBlurSize), 0, 0, BORDER_DEFAULT);
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
		Sobel(img_gray, grad[i * 2], CV_8U, 1 - kt, kt, 1, BORDER_DEFAULT);
		Sobel(img_gray, grad[i * 2 + 1], CV_8U, kt, 1 - kt, 1, BORDER_DEFAULT);
		//imshow("grad1", grad[i * 2]);
		//imshow("grad2", grad[i * 2 + 1]);
		//waitKey(0);
		//threshold(grad[i*2], grad[i*2], binary_thres, 255, CV_8U);
		//threshold(grad[i * 2+1], grad[i * 2 + 1], binary_thres, 255, CV_8U);
		//convertScaleAbs(grad[i*2], grad[i*2]); 
		//convertScaleAbs(grad[i * 2+1], grad[i * 2+1]);
		//imshow(num2str(i), grad[i]);
		merge.push_back(grad[i * 2]);
		merge.push_back(grad[i * 2 + 1]);
		//if(i==0)merge =  grad[i*2]|grad[i*2+1]; 
		//else merge = merge | grad[i * 2] | grad[i * 2+1];

	}
	//imshow("merge", merge);
	//waitKey(0);
	merge = merge.reshape(1, 1);
	return merge;
}
float judge_which(Mat src)
{

	Mat src3 = src.clone();
	//resize(src, src3, Size(200, 200));
	//imshow("src3", src3);
	src3 = extractSobelFeature(src3);
	//imshow("resize", src3);
	//waitKey(0);
	src3 = src3.reshape(1, 1);
	src3.convertTo(src3, CV_32FC1);
	//Ptr<ANN_MLP> bp = ANN_MLP::create();
	//bp = ml::ANN_MLP::load<ml::ANN_MLP>(ann_xml_path);
	float jud = -1;
	if (jud_mod == 1)jud=model->predict(src3);
	if (jud_mod == 2)
	{
		Mat output((int)1, 2, CV_32F);
		bp->predict(src3, output);
		jud = output.at<float>(0, 0) > output.at<float>(0, 1) ? -1.0 : 1.0;
	}
	//cout << "judge is" << jud << " " << endl;
	//waitKey(0);
	return jud;
}
Mat reverse(Mat src)
{

	Mat dst = src<100;

	return dst;
}

Mat connect_pix(Mat img, int threshval)
{
	Mat img_loca;
	cvtColor(img, img_loca, CV_RGB2GRAY);
	Mat bw = threshval < 128 ? (img_loca < threshval) : (img_loca > threshval);
	Mat labelImage(img_loca.size(), CV_32S);
	int nLabels = connectedComponents(bw, labelImage, 8);
	std::vector<Vec3b> colors(nLabels);
	colors[0] = Vec3b(0, 0, 0);//background
	for (int label = 1; label < nLabels; ++label) {
		colors[label] = Vec3b((rand() & 255), (rand() & 255), (rand() & 255));
	}
	Mat dst(img.size(), CV_8UC3);
	for (int r = 0; r < dst.rows; ++r) {
		for (int c = 0; c < dst.cols; ++c) {
			int label = labelImage.at<int>(r, c);
			Vec3b &pixel = dst.at<Vec3b>(r, c);
			if(label!=0)pixel = Vec3b(255, 255, 255);
			else pixel = colors[0];
		}
	}
	return dst;
}

void close_GetPlate(Mat srcImage,vector<Rect>& pitches_list)
{
	Mat operate;
	
	cvtColor(srcImage, operate, CV_BGR2GRAY); // 转为灰度图像 
	//imshow("operate", operate);
	Mat ret;
	//threshold(operate, ret, 20, 255, CV_THRESH_BINARY);
	//int ssii = 1;morphologyEx(operate, ret, MORPH_CLOSE, Mat::ones(ssii, ssii, CV_8UC1));
	operate.convertTo(ret, CV_8UC1);
	//imshow("ret", ret);
	//waitKey(0);
	vector<vector<Point>> plate_contours;
	findContours(ret, plate_contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	Mat candidates;
	for (size_t i = 0; i != plate_contours.size(); ++i)
	{
		// 求解最小外界矩形  
		Rect rect = boundingRect(plate_contours[i]);
		int height = rect.height;
		int width = rect.width;
		double ratio = double(width) / height;
		double area = double(height) * width;
		if (ratio < 0.1)continue;
		if (area < 20)continue;
		//Mat cp = srcImage(rect).clone();
		pitches_list.push_back(rect);
		//imshow("pitch", srcImage(rect));
		//waitKey(0);
	}
	//return ans;
	//return  candidates;
}

std::vector<cv::Rect> mserGetPlate(cv::Mat srcImage)
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
	cv::Ptr<cv::MSER> mesr1 = cv::MSER::create(step_l, 20, 600, thres_s, 0.3);
	cv::Ptr<cv::MSER> mesr2 = cv::MSER::create(step_l, 20, 600, thres_s, 0.3);


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
		cv::MORPH_CLOSE, cv::Mat::ones(3, 1, CV_8UC1));
	cv::imshow("mserClosedMat", mserClosedMat);
	// 寻找外部轮廓
	std::vector<std::vector<cv::Point> > plate_contours;
	cv::findContours(mserClosedMat, plate_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	// 候选车牌区域判断输出
	std::vector<cv::Rect> candidates;
	for (size_t i = 0; i != plate_contours.size(); ++i)
	{
		// 求解最小外界矩形
		cv::Rect rect = cv::boundingRect(plate_contours[i]);
		// 宽高比例
		double wh_ratio = rect.width / double(rect.height);
		double areaa = rect.width * double(rect.height);

		// 不符合尺寸条件判断
		if (wh_ratio > 0.2&&areaa>20)
			candidates.push_back(rect);

	}
	return  candidates;
}

void judge_cal(Mat ori,Rect rect, vector<Rect> rect_list)
{
	float jud = judge_which(ori(rect));
	
	
	
	int found = 0;
	//cout<<"judge is "<<jud<<endl;
	for (int jk = 0; jk < rect_list.size(); jk++)
	{
		Rect r1 = rect, r2 = rect_list[jk];
		Rect r3 = r1 & r2;
		double exact = 0.7;
		double ratio1 = double(r3.area()) / r2.area();
		double ratio2 = double(r3.area()) / r1.area();
		if (ratio1 >= exact && ratio2>= exact)
		 {
	
			if (found_statistic[jk] == 0)
			{
				//imshow("ori", ori);
				//imshow("cap", ori(rect));
				//waitKey(0);
				found_statistic[jk] = 1.0;
			}
			if (jud == 1.00)pt_rt++;
			if (jud == -1.00)pf_rt++;
			found = 1;					//break;
			//string has_path = "E:/grade3/baidu_pic/plate/qualified/train/found/";
			//has_path = has_path + num2str(has_ite++) + ".jpg";
			//imwrite(has_path, ori(rect));
		}
		//break;
	}
	/*
	for (int tp = 0; tp + ceil(height*rrt) <width; tp++)
	{
		Rect rect_tp(rect.x + tp, rect.y, int(height*0.75), height);
		
	}
	*/
	if (found == 0)
	{
		if (jud == 1.00)pt_rf++;
		if (jud == -1.00)pf_rf++;
		string no_path = "E:/grade3/baidu_pic/plate/qualified/train/no/";
		no_path = no_path + num2str(no_ite++) + ".jpg";
		imwrite(no_path, ori(rect));
	}
}

void img_calculate(Mat img, vector<Rect> rect_list)
{
	for (int i = 0; i < rect_list.size(); i++)found_statistic[i] = 0.0;
	vector<Mat> img_list;
	vector<Rect> pitches_list;
	Mat img_loca = img.clone();
	for (int val = 30; val < 225; val = val + 10)
	{
		img_list.push_back(connect_pix(img, val));
	}
	for (int i = 0; i < img_list.size(); i++)
	{
		close_GetPlate(img_list[i], pitches_list);
	}
	//pitches_list = mserGetPlate(img_loca);
	for (int i = 0; i < pitches_list.size(); i++)
	{
		//cout << pitches_list.size() << endl;
		Rect rect = pitches_list[i];
		//judge_cal(img, rect, rect_list);
		/*
		namedWindow("pitches",1);
		imshow("pitches", img(rect));
		imshow("ori", img);
		int c=waitKey(0);
		if (c == 48)break;
		destroyAllWindows();
		*/
		/*
		int height = rect.height, width = rect.width;
		double ratio = double(width) / height;
		double rrt = 0.7;
		if (ratio > rrt&& ratio<5)
		{
			//cout << "a" <<width<<" "<<height<< endl;
			for (int tp = 0; tp + height*rrt <width; tp++)
			{
				Rect rect_tp(rect.x + tp, rect.y, int(height*0.7), height);
				//imshow("rect", img(rect_tp));
				//waitKey(0);
				//cout << rect.x + tp<<" "<<rect.y<<" "<<int(height*0.7)<<" "<< height << endl;
				judge_cal(img,rect_tp, rect_list);
			}
		}
		if (ratio > 1.0 && ratio < 5)
		{
			for (int tp = 0; tp + height * 1 <width; tp++)
			{
				Rect rect_tp(rect.x + tp, rect.y, int(height * 1.0), height);
				//imshow("rect", img(rect_tp));
				//waitKey(0);
				//cout << rect.x + tp<<" "<<rect.y<<" "<<int(height*0.7)<<" "<< height << endl;
				judge_cal(img, rect_tp, rect_list);
			}
			continue;
		}
		//cout << "b" << endl;
		*/
		judge_cal(img,rect, rect_list);
	}
	double tcap = 0.0;
	for (int i = 0; i < rect_list.size(); i++)
	{
		total_found += found_statistic[i];
		tcap += found_statistic[i];
	}
	tcap = tcap / rect_list.size();
	if (tcap == 1.00)success++;
	if (tcap >= 0.8)eighty_success++;
	cout << "tcap==" << tcap << endl;
	//cout << "success==" << success/total_pic_num << endl;
	//system("pause");
	total_plate += rect_list.size();
	total_reco_rate += tcap;
	//waitKey(0);
}

void process_d()
{
	char* locate_path = "E:/grade3/baidu_pic/plate/plate_locate.txt";
	fp = fopen(locate_path, "r");
	char temp[1000];
	string temp_path;
	vector<Rect> rect_list;
	Mat img;
	int top, left, down, width, height;
	while (fscanf(fp, "%s", temp) != EOF)
	{
		//printf("%s\n",temp);
		if (strcmp(temp, "<image") == 0)
		{
			total_pic_num++;
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
			//printf("%d %d %d %d\n", top, left, width, height);
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
			//imshow("rect", img(rect));
			//waitKey(0);
		}
		if (strcmp(temp, "</image>") == 0)
		{
			if (rect_list.size() == 0)
			{
				total_pic_num--;
				break;
			}

			img_calculate(img, rect_list);

			double precision, recall, f1_score;
			precision = double(pt_rt) / (pt_rt + pt_rf);
			recall = double(pt_rt) / (pt_rt + pf_rt);
			f1_score = 2 * precision*recall / (precision + recall);
			total_found_rate = total_found / total_plate;
			//total_reco_rate /= total_pic_num;
			cout << "precison=" << precision << " "
				<< "recall==" << recall << " " <<
				"f1_score==" << f1_score << endl;
			cout << "total_found_rate=" << total_found_rate << endl;
			cout << "total_reco_rate=" << total_reco_rate / total_pic_num << endl;
			cout << "success=" << success / total_pic_num << endl;
			cout << "eighty_success=" << eighty_success / total_pic_num << endl;
			rect_list.clear();
			temp_path.clear();
			//waitKey(0);
			//system("pause");
		}
	}
	fclose(fp);
}
int main(int argc, const char** argv)
{
	/*
	double best = 0, best_s = 0, best_l = 0;
	for (int i = 1; i < 11; i++)
	{
		for (double j = 0.1; j < 0.6; j=j+0.1)
		{
			step_l = i;
			thres_s = j;
			process_d();
			if (total_found_rate > best)
			{
				best = total_found_rate;
				best_s = thres_s;
				best_l = step_l;
			}
			//cout << best << " " << best_s << " " << best_l << endl;
			//system("pause");
		}
		
	}
	cout << best << " " << best_s << " " << best_l << endl;*/
	process_d();
	system("pause");

	//system("pause");
	return 0;
}
