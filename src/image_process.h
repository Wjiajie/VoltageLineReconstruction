#pragma once

#include "head.h"
#include "reconstruction.h"

class ImageProcess
{
public:
	ImageProcess() {};

	//��ɫͼת�Ҷ�ͼ
	void rgb2gray(Mat src , Mat & dst);
	//ȥ����
	void ImageDedistortion(vector<Mat> src, vector<Mat> & dst , vector<string> path);
	//�� �ԱȶȱȽϵ͵�ͼ�����Ҫ
	void Sharpen(Mat src, Mat & dst);

	//read data
	void ReadDataFromFile(string src_image_path, vector<Mat>& src_image);
	void LoadFeatureAndMatchFile(vector<LinePoint> & feature_s, vector<LinePoint> & feature_t, vector<Point2d> & match_s_t, string save_path , string pattern);
	//��txt�ļ��ж�ȡ���ݵ�vector��
	void ConvertTxt2Vector(string file_path, vector<Point2f> & vector_point, string pattern);
	
	//save data
	//���������� Mat תΪ txt ��ʽ����
	void ConvertImage2Txt(Mat image, string save_path);
	void SaveFeatureAndMatchFile(vector<LinePoint> feature_s, vector<LinePoint> feature_t, vector<Point2f> match_s_t, string save_path);
	void ConvertVector2Txt(string file_path, vector<Point2f> & vector_point);

	~ImageProcess() {};

};