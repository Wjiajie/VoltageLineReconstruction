#pragma once
#include "head.h"

#define COUTENDL(x) std::cout << x << std::endl

//��ѹ�ߵ���ȡ������ȡͼƬ����Ϊ�˵���ߵĲ���,���ҽ����ˮƽ�ݶȵ�ֱ�ߣ�
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

		uint NumberofNOTConsecutiveUpdates = 0; //���������£�������ż���û�и��¾Ͳ��ڸ������ֱ�ߣ���ֹ�в��ֶϵ��ֱ�ߣ�
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
	void InitFindLine(); //��ʼ�������� Ҳ���Ǵӵ�һ���ҵ������㣬����Ϊ��Ҫ�ҵĵ�
	void FindLines(); //���ݳ�ʼ�ĵ㣬����������
	inline int findNearestPoint(Eigen::Vector2i p); //���Һ������������ĵ㣬����Ϊ���ߵ���һ�������㣬���������ߵ�Index
	int findBeginColIndex(uint aim_row, int startRow, uint lineIdx = 0); //����һ�ε�����Ҫ���к�
	bool updateParameter(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& A, uint lineIdx);
	bool findInitPoints(int startRow, uint lineIdx, uint& lastRow);

	bool findLinesOnlyByBinaryImg();

};

//[-1,0,1]���ݶ�(�����������������)
inline double CalHorizontalGradient(double x1, double x2, double x3)
{
	double r = x3 - x1;
	return r > 0 ? r : -1 * r;
}

//ֱ�ӱȽ����ز�
inline double CalHorizontalGradient(double x_l, double x_r)
{
	double r = x_r - x_l;
	return r > 0 ? r : -1 * r;
}