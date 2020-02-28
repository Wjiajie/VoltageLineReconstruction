#pragma once

#include "Head.h"


class LineExtract
{
public:
	LineExtract();
	void InitLineHead(Mat gray_image, vector<Point2i> & initpoint);
	void FindCenterPoint(Mat gray_image, Mat & center_point);
	void FindInitLineHead(Mat center_point, vector<Point2i> & initpoint);
	void ExtractLine(Mat gray_image, vector<Point2i> initpoint,Mat & final_line);
	void ExtractLineSingle(Mat gray_image, Point2i init_point_index,Mat & final_line);
	~LineExtract();

};