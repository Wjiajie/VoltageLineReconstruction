#include "image_process.h"

//转换成灰度图
void ImageProcess::rgb2gray(Mat src , Mat & dst)
{
	cvtColor(src, dst, CV_BGR2GRAY);
}

//去畸变
void ImageProcess::ImageDedistortion(vector<Mat> src, vector<Mat> & dst , vector<string> path)
{
	for (int i = 0; i < src.size(); ++i)
	{
		Mat output;
		Mat	cameraMatrix, distCoeffs;
		Size imageSize;
		Mat map1, map2;

		// 读取相机内参
		bool FSflag = false;
		FileStorage readfs;

		FSflag = readfs.open(path[i], FileStorage::READ);
		if (FSflag == false) cout << "Cannot open the file" << endl;
		readfs["camera_matrix"] >> cameraMatrix;
		readfs["distortion_coefficients"] >> distCoeffs;
		readfs["image_width"] >> imageSize.width;
		readfs["image_height"] >> imageSize.height;

		cout << "cameraMatrix" << cameraMatrix << endl << "distCoeffs" << distCoeffs << endl << "imageSize" << imageSize << endl;

		readfs.release();

		//去畸变	

		Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 0);

		undistort(src[i], output, cameraMatrix, distCoeffs);
		dst.emplace_back(output);
	}
	
}

void ImageProcess::Sharpen(Mat src, Mat & dst)
{
	//图像锐化
	Mat h_kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	filter2D(src, dst, CV_16UC1, h_kernel);
	convertScaleAbs(dst, dst);
	
}

void ImageProcess::ReadDataFromFile(string src_image_path, vector<Mat>& src_image)
{
	vector<cv::String> src_files;
	glob(src_image_path, src_files, false);     //读取文件夹下所有符合要求的文件路径

	for (int i = 0; i < src_files.size(); ++i)
	{
		Mat image = imread(src_files[i]);
		src_image.emplace_back(image);
	}
}


void ImageProcess::SaveFeatureAndMatchFile(vector<LinePoint> feature_s, vector<LinePoint> feature_t, vector<Point2f> match_s_t ,string save_path)
{
	// 按照line_id保存匹配点 ，湖州4线

	ofstream ofs_l,ofs_m,ofs_l_m;
	string ofs_l_path = save_path + "\\feature_s.txt";
	string ofs_m_path = save_path + "\\feature_t.txt";
	string ofs_l_m_path = save_path + "\\match_s_t.txt";
	ofs_l.open(ofs_l_path);
	ofs_m.open(ofs_m_path);
	ofs_l_m.open(ofs_l_m_path);

	if (!ofs_l.is_open() || !ofs_m.is_open() ||!ofs_l_m.is_open())
	{
		cout << "TXT文件不存在" << endl;		
	}
	
	else
	{
		for (int i = 0; i < feature_s.size(); i++)
		{
			ofs_l << feature_s[i].point.x << " " << feature_s[i].point.y <<" "<< feature_s[i].line_id1 << " " << feature_s[i].line_id2 << endl;
		}
		for (int i = 0; i < feature_t.size(); i++)
		{
			ofs_m << feature_t[i].point.x << " " << feature_t[i].point.y <<" "<< feature_t[i].line_id1 << " " << feature_t[i].line_id2 << endl;
		}
		for (int i = 0; i < match_s_t.size(); i++)
		{
			ofs_l_m << match_s_t[i].x << " " << match_s_t[i].y << endl;
		}

	}
	ofs_l.close();
	ofs_m.close();
	ofs_l_m.close();

}

void ImageProcess::LoadFeatureAndMatchFile(vector<LinePoint> & feature_s, vector<LinePoint> & feature_t, vector<Point2d> & match_s_t, string save_path , string pattern)
{
	ifstream ifs_l, ifs_m, ifs_l_m;
	string ifs_l_path = save_path + "\\feature_s.txt";
	string ifs_m_path = save_path + "\\feature_t.txt";
	string ifs_l_m_path = save_path + "\\match_s_t.txt";
	ifs_l.open(ifs_l_path);
	ifs_m.open(ifs_m_path);
	ifs_l_m.open(ifs_l_m_path);

	if (!ifs_l.is_open() || !ifs_m.is_open() || !ifs_l_m.is_open())
	{
		cout << "TXT文件不存在" << endl;
	}
	else
	{
		string s1,s2,s3;
		while (getline(ifs_l, s1))
		{
			int point_index = 0;
			vector<string> ret;
			if (!pattern.empty())
			{
				size_t start = 0, index = s1.find_first_of(pattern, 0);
				while (index != s1.npos)
				{
					if (start != index)
						ret.push_back(s1.substr(start, index - start));
					start = index + 1;
					index = s1.find_first_of(pattern, start);
				}
				if (!s1.substr(start).empty())
					ret.push_back(s1.substr(start));
			}
			//(x,y)
			feature_s.emplace_back(LinePoint(Point3f(atof(ret[0].c_str()), atof(ret[1].c_str()), 1), point_index++, atof(ret[2].c_str()), atof(ret[3].c_str())));			
		}

		while (getline(ifs_m, s2))
		{
			int point_index = 0;
			vector<string> ret;
			if (!pattern.empty())
			{
				size_t start = 0, index = s2.find_first_of(pattern, 0);
				while (index != s2.npos)
				{
					if (start != index)
						ret.push_back(s2.substr(start, index - start));
					start = index + 1;
					index = s2.find_first_of(pattern, start);
				}
				if (!s2.substr(start).empty())
					ret.push_back(s2.substr(start));
			}
			//(x,y)
			feature_t.emplace_back(LinePoint(Point3f(atof(ret[0].c_str()), atof(ret[1].c_str()), 1), point_index++, atof(ret[2].c_str()), atof(ret[3].c_str())));
		}

		while (getline(ifs_l_m, s3))
		{
			int point_index = 0;
			vector<string> ret;
			if (!pattern.empty())
			{
				size_t start = 0, index = s3.find_first_of(pattern, 0);
				while (index != s3.npos)
				{
					if (start != index)
						ret.push_back(s3.substr(start, index - start));
					start = index + 1;
					index = s3.find_first_of(pattern, start);
				}
				if (!s3.substr(start).empty())
					ret.push_back(s3.substr(start));
			}
			//(x,y)
			match_s_t.emplace_back(Point2f(atof(ret[0].c_str()), atof(ret[1].c_str())));
		}
	}
}

//将特征点由 png 转为 txt 格式保存
void ImageProcess::ConvertImage2Txt(Mat image, string save_path)
{
	ofstream ofs;
	ofs.open(save_path);
	if (!ofs.is_open())
		cout << "TXT文件不存在" << endl;
	else
		for (int i = 0; i < image.rows; ++i)
		{
			for (int j = 0; j < image.cols; ++j)
			{
				if (*(image.data + i * image.step[0] + j * image.step[1]) == 255)
					ofs << j << " " << i << endl;
			}
		}	
}

void ImageProcess::ConvertTxt2Vector(string file_path, vector<Point2f> & vector_point, string pattern)
{
	ifstream ifs;
	ifs.open(file_path);
	if (!ifs.is_open())
		cout << "Txt文件不存在" << endl;
	else
	{
		string s;
		while (getline(ifs, s))
		{
			vector<string> ret;
			if (!pattern.empty())
			{
				size_t start = 0, index = s.find_first_of(pattern, 0);
				while (index != s.npos)
				{
					if (start != index)
						ret.push_back(s.substr(start, index - start));
					start = index + 1;
					index = s.find_first_of(pattern, start);
				}
				if (!s.substr(start).empty())
					ret.push_back(s.substr(start));
			}
			//(x,y)
			vector_point.push_back(Point2f(atof(ret[0].c_str()), atof(ret[1].c_str())));
					
		}
	}
	
}

void ImageProcess::ConvertVector2Txt(string file_path, vector<Point2f> & vector_point)
{
	ofstream ofs;
	ofs.open(file_path);
	if (!ofs.is_open())
		cout << "TXT文件不存在" << endl;
	else
		for (int i = 0; i < vector_point.size(); ++i)
		{
			ofs << vector_point[i].x << " " << vector_point[i].y << endl;
		}
	ofs.close();
}