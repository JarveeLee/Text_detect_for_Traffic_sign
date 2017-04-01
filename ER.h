#ifndef ER_H
#define ER_H

#include "common.h"
const int dx[] = { 1, -1, 1, -1, 0, 1, 0, -1 };
const int dy[] = { 1, -1, -1, 1, 1, 0, -1, 0 };

class ER {
public:
	ER();
	double area;
	// double perimeter;
	Rect bounding_rect;
};

void detect_ers(Mat image, vector<ER> &ers);
void show_ers(Mat image, vector<ER> ers);
void extractRect(Mat rgb_img, vector<ER> ers, vector<Rect> &candidates);
#endif