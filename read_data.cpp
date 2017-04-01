#include "read_data.h"

void read_rect_list(vector<label_img> &img_list,
	int num)
{
	/*
	pt_rt = 0.0, pf_rt = 0.0, pt_rf = 0.0, pf_rf = 0.0,
	total_found = 0.0, total_plate = 0.0, total_reco_rate = 0.0, total_pic_num = 0.0,
	success = 0.0, eighty_success = 0.0, total_found_rate = 0.0;
	*/
    
	FILE *fp = fopen("/home/cpdp/lijiahui/svm/img/label_img_data.txt", "r");
	//cout<<"fopen"<<endl;
	char temp[1000];
	string temp_path;
	Mat img;
	label_img temp_img;
	int top, left, down, width, height;
	cout<<"begin read img_list"<<endl;
        //fscanf(fp, "%s", temp);
        //printf("%s\n",temp);

	while (fscanf(fp, "%s", temp))
	{
		//printf("%s\n",temp);

		if (strcmp(temp, "<image") == 0)
		{
			fscanf(fp, " file='%s'>\n", temp);
			temp_path.clear();
			for (int i = 0; i < strlen(temp) - 2; i++)temp_path = temp_path + temp[i];
			cout << temp_path << endl;
			img = imread(temp_path, 1);
			//resize(img, img, Size(600, 600));
			temp_img.img = img.clone();
		}
		if (strcmp(temp, "<box") == 0)
		{
			fscanf(fp, " top='%d' left='%d' width='%d' height='%d'/>\n",
				&top, &left, &width, &height);
			if (top + height> img.rows)
			{
				height = img.rows - top;
			}
			if (top  < 0)top = 0;
			if (left + width > img.cols)width = img.cols - left;
			if (left < 0)left = 0;
			Rect rect(left, top, width, height);
			temp_img.label.push_back(rect);
			//imshow("rect", img(rect));
			//waitKey(0);
		}
		if (strcmp(temp, "</image>") == 0)
		{
			img_list.push_back(temp_img);
			temp_img.label.clear();
			if (img_list.size() > num)break;
		}
	}
	fclose(fp);
}