#include "ER.h"
#include "read_data.h"
#include "common.h"
using namespace std;

string data_path = "/home/cpdp/lijiahui/svm/img/";
string caffe_path = "/home/cpdp/lijiahui/caffe/caffe-master/data/myself/";
char * label_path = "/home/cpdp/lijiahui/svm/img/label.txt";
char * train_txt = "/home/cpdp/lijiahui/caffe/caffe-master/data/myself/train.txt";
char * test_txt = "/home/cpdp/lijiahui/caffe/caffe-master/data/myself/val.txt";
int train_has_ite = 0, train_no_ite = 0, test_has_ite = 0, test_no_ite = 0,train_num=945;
FILE * fp1;
FILE * fp_train = fopen(train_txt,"w");
FILE * fp_val = fopen(test_txt,"w");
FILE * fp_label = fopen(label_path,"w");
string num2str(double i)
{
	stringstream ss;
	ss << i;
	return ss.str();
}

void img_write(int ite,int total,double rate,Mat img,int tr)
{
	string temp;
	if (tr == 1 && ite < total*rate)
	{
		temp="has/"+ num2str(train_has_ite) + ".jpg";
		imwrite(caffe_path+"train/" +temp, img);
		fprintf(fp_train , "%s %d\n" , temp.data(),tr );
		train_has_ite++;
	}
	
	if (tr == 1 && ite >= total*rate)
	{
		temp="has/"+ num2str(test_has_ite) + ".jpg";
		imwrite(caffe_path+"val/" +temp, img);
		fprintf(fp_val , "%s %d\n" , temp.data(),tr );
		test_has_ite++;
	}
	if (tr == 0 && ite >= total*rate)
	{
		temp="no/"+ num2str(test_no_ite) + ".jpg";
		imwrite(caffe_path+"val/" +temp, img);
		fprintf(fp_val , "%s %d\n" , temp.data(),tr );
		test_no_ite++;
	}
	if (tr == 0 && ite < total*rate)
	{
		temp="no/"+ num2str(train_no_ite) + ".jpg";
		imwrite(caffe_path+"train/" +temp, img);
		fprintf(fp_train , "%s %d\n" , temp.data(),tr );
		train_no_ite++;
	}

}

int predict(Mat img)
{
	string temp_path = "/home/cpdp/lijiahui/caffe/caffe-master/data/myself/temp.jpg";
	imwrite(temp_path,img);
	string command = "/home/cpdp/lijiahui/caffe/caffe-master/build/examples/cpp_classification/classification1.bin \
  /home/cpdp/lijiahui/caffe/caffe-master/data/myself/deploy.prototxt \
  /home/cpdp/lijiahui/caffe/caffe-master/data/myself/snapshot_iter_1000.caffemodel \
  /home/cpdp/lijiahui/caffe/caffe-master/data/myself/imagenet_mean.binaryproto \
  /home/cpdp/lijiahui/caffe/caffe-master/data/myself/word.txt \ ";
    command+=temp_path;
	char cddd[5000];
	for(int i=0;i<command.size();i++)cddd[i]=command[i];
	char buffer[80];
	fp1=popen(cddd,"r");
	fgets(buffer,sizeof(buffer),fp1);
	int jud=buffer[0]-48;
	//printf("%d",jud);
	pclose(fp1);
	return jud;
}

int main(int argc, const char* argv[]) {
	
	//Mat ck=imread("/home/cpdp/lijiahui/caffe/caffe-master/data/myself/val/has/5.jpg");
	//cout<<predict(ck)<<endl;
	vector<ER> ers;
	vector<Rect> candidates;
	vector<label_img> img_list;
	Mat pic;
	vector<Rect> rect_list;
	//detect_ers(rgb_img,ers);
	//extractRect(rgb_img, ers, candidates);
	//show_ers(rgb_img, ers);
	read_rect_list(img_list,50);
	int found[1000],check[1000],found_num=0,total_label=0;
	float pt_rt=0, pt_rf=0, pf_rt=0, pf_rf=0;
        cout<<"check"<<endl;
	for (int i = 0; i < img_list.size(); i++)
	{
		if(i==train_num)
		{
			pt_rt=0, pt_rf=0, pf_rt=0, pf_rf=0;
		}
		pic = img_list[i].img;
		fprintf(fp_label,"<image file= '%s%s.jpg'>\n",data_path.data(),num2str(i).data());
		//Mat featuress;
		rect_list = img_list[i].label;
		ers.clear();
		candidates.clear();
        //cout<<"Extracting ers"<<endl;
		extractRect(pic, ers, candidates);
        //cout<<"Got ers,checking"<<endl;
		cout << "i= " << i <<" "<< rect_list.size()<<" "<< candidates.size()<<endl;
		for (int j = 0; j < rect_list.size(); j++)found[j] = 0,check[j]=0;
		for (int k = 0; k < candidates.size(); k++)
		{
			Rect r2 = candidates[k];
			int ok = 0;
			int p=0,r=0;
			p=predict(pic(r2));
			for (int j = 0; j < rect_list.size(); j++)
			{
				//if(k==0)img_write(i, img_list.size(), 0.7, pic(rect_list[j]), 1);
				//int p = 0, r = 0;
				Rect r1 = rect_list[j];
				Rect r3 = r1 & r2;
				double exact = 0.7;
				double ratio1 = double(r3.area()) / r1.area();
				double ratio2 = double(r3.area()) / r2.area();
				double w1 = r1.width, w2 = r2.width, h1 = r1.height, h2 = r2.height;
				if ((ratio1 >= 0.7 && ratio2 >= 0.7)
					|| (ratio2 >=0.8)
					||(ratio1 >= 0.9&&h1/h2>=0.7&&h1 / h2<=1.3
						&&w1 / w2 <= 8))
				{
					found[j] = 1;
					r = 1;
					ok = 1;
				}
				//rectangle(pic, r2, Scalar(255, 0, 0));
				//rectangle(pic, r1, Scalar(0, 0, 255));
				//int jud = judge_which(pic(r2),0.5);
				//p = jud == 1 ? 1 : 0;
				//
				//if(p==1)rectangle(pic, r2, Scalar(0, 255, 0));	
			}
			if (p == 1 && r == 1)pt_rt++;
			if (p == 1 && r == 0)pt_rf++;
			if (p == 0 && r == 1)pf_rt++;
			if (r == 1)img_write(i, img_list.size(), 0.7, pic(r2), 1);
			if (r == 0)img_write(i, img_list.size(), 0.7, pic(r2), 0);
			if (p == 1)
			{
				fprintf(fp_label," <box top='%d' left='%d' width='%d' height='%d'/>\n",
                        r2.y,r2.x,r2.width,r2.height);
			}
		}
		total_label += rect_list.size();
		int cur_found = 0;
		for (int j = 0; j < rect_list.size(); j++)cur_found +=found[j];
		found_num += cur_found;
		cout << double(cur_found)/ rect_list.size() <<" "<< double(found_num) / total_label << endl;
		float precision = pt_rt / (pt_rt + pt_rf), recall = pt_rt / (pt_rt + pf_rt);
		cout << "precision="<<precision << "  recall=" << recall << endl;
		fprintf(fp_label,"</image>\n");
	}
	fclose(fp_label);
	fclose(fp_train);
	fclose(fp_val);
	system("pause");
	return 0;
}
