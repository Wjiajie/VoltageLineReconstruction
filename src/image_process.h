#pragma once

#include "head.h"
#include "reconstruction.h"

class ImageProcess
{
public:
	ImageProcess() {};

	//彩色图转灰度图
	void rgb2gray(Mat src , Mat & dst);
	//去畸变
	void ImageDedistortion(vector<Mat> src, vector<Mat> & dst , vector<string> path);
	//锐化 对比度比较低的图像才需要
	void Sharpen(Mat src, Mat & dst);

	//read data
	void ReadDataFromFile(string src_image_path, vector<Mat>& src_image);
	void LoadFeatureAndMatchFile(vector<LinePoint> & feature_s, vector<LinePoint> & feature_t, vector<Point2d> & match_s_t, string save_path , string pattern);
	//从txt文件中读取数据到vector中
	void ConvertTxt2Vector(string file_path, vector<Point2f> & vector_point, string pattern);
	
	//save data
	//将特征点由 Mat 转为 txt 格式保存
	void ConvertImage2Txt(Mat image, string save_path);
	void SaveFeatureAndMatchFile(vector<LinePoint> feature_s, vector<LinePoint> feature_t, vector<Point2f> match_s_t, string save_path);
	void ConvertVector2Txt(string file_path, vector<Point2f> & vector_point);

	~ImageProcess() {};

};