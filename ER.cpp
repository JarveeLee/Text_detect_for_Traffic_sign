#include "ER.h"

ER::ER() {
	area = 0;
	// perimeter = 0;
}

bool is_border_point(int x, int y, Mat label_mat) {
        //cout<<"border"<<endl;
	int label = label_mat.at<int>(x, y);
	for (int i = 0; i < 8; ++i) {
		int tmp_x = x + dx[i];
		int tmp_y = y + dy[i];
		if (!(tmp_x >= 0 && tmp_x < label_mat.rows && tmp_y >= 0 && tmp_y < label_mat.cols)) return true;
		int other_label = label_mat.at<int>(tmp_x, tmp_y);
		if (label != other_label) return true;
	}
	return false;
}

void detect_ers_in_binary(Mat binary, vector<ER> &ers) {
        //cout<<"in binary"<<endl;
	Mat label_mat;
	int label_num = connectedComponents(binary, label_mat);

	vector<vector<Point> > components(label_num);
	vector<double> area(label_num);
	vector<double> perimeter(label_num);
	for (int i = 0; i < label_mat.rows; ++i) {
		for (int j = 0; j < label_mat.cols; ++j) {
			int label = label_mat.at<int>(i, j);
			area[label] += 1;
			components[label].push_back(Point(j, i));
			/*
			if (is_border_point(i, j, label_mat)) {
			perimeter[label] += 1;
			components[label].push_back(Point(j, i));
			}
			*/
		}
	}

	for (int i = 0; i < components.size(); ++i) {
		if (area[i] < 60 || area[i] > 40000) continue;
		ER er;

		er.area = area[i];
		// er.perimeter = perimeter[i];
		er.bounding_rect = boundingRect(components[i]);
		ers.push_back(er);
	}
}

void detect_ers_in_single_channel(Mat image, vector<ER> &ers) {
        //cout<<"single channel"<<endl;
	Mat gray_image;
	cvtColor(image, gray_image, COLOR_BGR2GRAY);

	for (int theta = 0; theta < 256; theta += 10) {
                //cout<<"theta= "<<theta<<endl;
		Mat binary, binary_inv;
		threshold(gray_image, binary, theta, 255, THRESH_BINARY);
		detect_ers_in_binary(binary, ers);

		threshold(gray_image, binary_inv, theta, 255, THRESH_BINARY_INV);
		detect_ers_in_binary(binary_inv, ers);
	}
}

vector<ER> filter_ers(vector<ER> ers) {
        //cout<<"filtering"<<endl;
	vector<ER> filtered_ers0;
        ER er;
	for (int i=0;i<ers.size();i++) {
                er=ers[i];
		double w = er.bounding_rect.width;
		double h = er.bounding_rect.height;

		if (w / h > 0.1 && w / h < 10) {
			filtered_ers0.push_back(er);
		}
	}

	vector<ER> filtered_ers1;
	for (int i = 0; i < filtered_ers0.size(); ++i) {
		bool flag0 = true;
		for (int j = 0; j < filtered_ers0.size(); ++j) {
			if (i == j) continue;
			Rect a = filtered_ers0[i].bounding_rect;
			Rect b = filtered_ers0[j].bounding_rect;
			Rect inter_rect = a & b;
			Rect union_rect = a | b;
			if (inter_rect.area() >= union_rect.area() * 0.95 && (a.area() < b.area() ||
				(a.area() == b.area() && i > j))) {
				flag0 = false;
				break;
			}
		}
		if (flag0) {
			filtered_ers1.push_back(filtered_ers0[i]);
		}
	}

	return filtered_ers1;
}

void detect_ers(Mat image, vector<ER> &ers) {
	vector<Mat> channels;
	//split(image, channels);
        //cout<<channels.size()<<endl;
        //int i=0;
	//for (Mat channel : channels) {
		detect_ers_in_single_channel(image, ers);
                //cout<<i++<<endl;
	//}

	ers = filter_ers(ers);
}

void show_ers(Mat image, vector<ER> ers) {
        ER er;
	for (int i=0;i<ers.size();i++) {
                er=ers[i];
		rectangle(image, er.bounding_rect, Scalar(255, 0, 0));
	}

	imshow("img", image);
	waitKey(0);
}

void extractRect(Mat rgb_img, vector<ER> ers, vector<Rect> &candidates)
{
	candidates.clear();
	detect_ers(rgb_img, ers);
	for (int i = 0; i < ers.size(); i++)
	{
		candidates.push_back(ers[i].bounding_rect);
	}
}