#pragma once
#include "head.h"

#define COUTENDL(x) std::cout << x << std::endl

//高压线的提取（仅提取图片顶上为端点的线的部分,而且仅针对水平梯度的直线）
class LineExtractor
{
public:
	Mat image;
	Mat gray_img;
	uint height, width;
	Mat binary_Edge;

	struct PointsInLine
	{
		uint lineIdx;
		std::vector<Eigen::Vector2i> vec_points;
		Eigen::Matrix3d lineParams;

		uint NumberofNOTConsecutiveUpdates = 0; //非连续更新，如果连着几次没有更新就不在更新这个直线（防止有部分断点的直线）
		Eigen::Vector3i color;
		PointsInLine()
		{

		}
		PointsInLine(uint idx)
		{
			lineIdx = idx;
			color = Eigen::Vector3i(rand() % 255 + 1, rand() % 255 + 1, rand() % 255 + 1);
		}

		inline void AddPoint(Eigen::Vector2i u)
		{
			this->vec_points.emplace_back(u);
		}

	};
	std::vector<PointsInLine> vec_lines;

	LineExtractor() {}
	LineExtractor(Mat _m, Rect ROI = Rect(-1, -1, -1, -1))
	{
		/*if (ROI.x < 0)
			image = _m;
		else
			image = _m(ROI);
		cvtColor(image, gray_img, CV_BGR2GRAY);
		height = image.rows;
		width = image.cols;*/
	}

	bool CalEdge();
	void InitFindLine(); //初始化找曲线 也就是从第一行找到几个点，就作为需要找的点
	void FindLines(); //根据初始的点，不断找曲线
	inline int findNearestPoint(Eigen::Vector2i p); //查找和这个像素最近的点，被认为是线的下一个迭代点，返回这条线的Index
	int findBeginColIndex(uint aim_row, int startRow, uint lineIdx = 0); //找下一次迭代需要的列号
	bool updateParameter(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& A, uint lineIdx);
	bool findInitPoints(int startRow, uint lineIdx, uint& lastRow);

	bool findLinesOnlyByBinaryImg();

};

//[-1,0,1]求梯度(宽度至少是三个像素)
inline double CalHorizontalGradient(double x1, double x2, double x3)
{
	double r = x3 - x1;
	return r > 0 ? r : -1 * r;
}

//直接比较像素差
inline double CalHorizontalGradient(double x_l, double x_r)
{
	double r = x_r - x_l;
	return r > 0 ? r : -1 * r;
}