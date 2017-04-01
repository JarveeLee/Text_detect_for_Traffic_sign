#include "common.h"

struct label_img
{
	Mat img;
	vector<Rect> label;
};

void read_rect_list(vector<label_img> &img_list,
	int num = 10);
