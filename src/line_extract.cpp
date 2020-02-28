#include "line_extract.h"

#define outlier_threshold 3 //二值化的参数
#define init_point_row 10 //初始化的行数
#define move_threshold 1 //卷积核左右移动的大小

////可修改调整

#define template_size 7 //值模板大小 7*7
#define end_size 3 //中值模板大小 3*3
#define end_threshold 1 //中断模板的方差小于该阈值就退出
#define update_threshold 10 //每隔10格判断是否更新一次 值模板   
#define value_threshold 5  //图像二值化的阈值 
#define value_threshold 5  //图像二值化的阈值 
#define vector_template_size 100 // 存储正确索引的模板长度

#define end_y_index 1480 // 1480 1390 //undistort_image.rows - vector_template_size - 1


LineExtract::LineExtract()
{
	cout << "class create!" << endl;
}

LineExtract::~LineExtract()
{
	cout << "class destory!" << endl;
}

void LineExtract::InitLineHead(Mat gray_image, vector<Point2i> & init_point)
{	//Mat center_point = Mat::ones(gray_image.rows, gray_image.cols, CV_8UC1);  2020-2-25
	Mat center_point = Mat::zeros(gray_image.rows, gray_image.cols, CV_8UC1);
	// 提取电力线的质心
	FindCenterPoint(gray_image, center_point);
	FindInitLineHead(center_point, init_point);
}

void LineExtract::FindCenterPoint(Mat gray_image, Mat & center_point)
{	
	Mat mask = Mat::zeros(gray_image.rows, gray_image.cols, CV_8UC1);
	Mat mask_process = Mat::zeros(gray_image.rows, gray_image.cols, CV_8UC1);
	/*Canny(gray_image, mask, 20, 150);*/
	
	for (int i = 0; i < gray_image.rows; ++i)
	{
		for (int j = 0; j < gray_image.cols - 1; ++j)
		{
			uchar value_1 = *(gray_image.data + i * gray_image.step[0] + j * gray_image.step[1]);
			uchar value_2 = *(gray_image.data + i * gray_image.step[0] + (j + 1)*gray_image.step[1]);

			//线头左侧为(i,j+1)
			if ((value_1 - value_2) > value_threshold)
			{
				*(mask.data + i * mask.step[0] + (j + 1)*mask.step[1]) = 255;
			}
			//线头右侧为(i,j)
			else if ((value_2 - value_1) > value_threshold)
			{
				*(mask.data + i * mask.step[0] + j * mask.step[1]) = 255;
			}
		}
	}

	int first_index = init_point_row, last_index = mask.cols - init_point_row;
	int first_count = 0, last_count = 0;

	//要求拍摄时，需要保留的电力线在图像最上边
	for (int i = 1; i < mask.rows; ++i)
	{
		for (int j = first_index; j < last_index; ++j)
		{

			if (*(mask.data + i * mask.step[0] + j * mask.step[1]) == 255)
			{
				*(mask_process.data + i * mask_process.step[0] + j * mask_process.step[1]) = 255;
				last_count = j;

				if (first_count == 0)
				{
					first_count = j;
				}
			}
		}

		first_index = (first_count - init_point_row) > 0 ? (first_count - init_point_row) : 0;

		first_count = 0;
		last_index = (last_count + init_point_row) < mask.cols ? (last_count + init_point_row) : mask.cols;
		last_count = 0;


	}
	Mat dst = mask_process.clone();

	//先腐蚀去除部分离群点，再膨胀，使六轴线连在一起
	// 获得结构元
	Mat se = getStructuringElement(MORPH_RECT, Size(1, 1));
	Mat se1 = getStructuringElement(MORPH_RECT, Size(1, 1));

	//dilate(dst, dst, se1);
	erode(dst, dst, se);

	//只保留电力线的“质心”位置的像素
	vector<int> vec_first, vec_last;

	for (int i = 0; i < dst.rows; ++i)
	{
		for (int j = 0; j < dst.cols - 1; ++j)
		{

			if (*(dst.data + i * dst.step[0] + j * dst.step[1]) == 0 && *(dst.data + i * dst.step[0] + (j + 1)*dst.step[1]) == 255)
			{
				vec_first.push_back(j + 1);
			}
			if (*(dst.data + i * dst.step[0] + j * dst.step[1]) == 255 && *(dst.data + i * dst.step[0] + (j + 1)*dst.step[1]) == 0)
			{
				vec_last.push_back(j);

			}
		}

		if (vec_first.size() == vec_last.size())
		{
			for (int k = 0; k < vec_first.size(); ++k)
			{
				int index = ceil((vec_first[k] + vec_last[k]) / 2.f);
				*(center_point.data + i * center_point.step[0] + index * center_point.step[1]) = 255;

			}

		}

		vec_first.clear();
		vec_last.clear();
	}
}
void LineExtract::FindInitLineHead(Mat center_point, vector<Point2i> & init_point)
{
	// 初始化好点,不选择开始的几行
	for (int i = 0; i < center_point.cols; ++i)
	{
		if (*(center_point.data + init_point_row * center_point.step[0] + i * center_point.step[1]) == 255)
		{
			// 要排除离群点的情况
			int winsize = init_point_row <= outlier_threshold ? init_point_row : outlier_threshold;
			int point_num = 0;
			for (int j = init_point_row - winsize; j <= init_point_row + winsize; ++j)
			{
				for (int k = i - winsize; k <= i + winsize; ++k)
				{
					if (*(center_point.data + j * center_point.step[0] + k * center_point.step[1]) == 255)
						point_num++;
				}
			}
			if (point_num >= outlier_threshold)
				init_point.push_back(Point2i(init_point_row, i));
		}
	}
}

void LineExtract::ExtractLine(Mat gray_image, vector<Point2i> initpoint, Mat & final_line)
{
	for (auto init_point_index : initpoint)
	{
		ExtractLineSingle(gray_image, init_point_index, final_line);
	}
}

void LineExtract::ExtractLineSingle(Mat undistort_image, Point2i init_point_index, Mat & final_line)
{
	cout << "START EXTRACT LINE" << endl;
	vector<int> status_vector;
	// 首先根据线头，构建一个 7*7 的像素值模板，它代表了电力线的值特征
	int template_radius = (template_size - 1) / 2;
	Mat image_template = Mat::zeros(template_size, template_size, CV_8UC1);
	image_template = undistort_image(Rect(init_point_index.y - template_radius, init_point_index.x - template_radius, template_size, template_size));

	int sum_value = 0, med_value = 0, max_value = -1, min_value = 999;
	// 找出模板内元素的中值
	for (int i = 0; i < image_template.rows; ++i)
	{
		for (int j = 0; j < image_template.cols; ++j)
		{
			if (*(image_template.data + i * image_template.step[0] + j * image_template.step[1]) > max_value)
				max_value = *(image_template.data + i * image_template.step[0] + j * image_template.step[1]);

			if (*(image_template.data + i * image_template.step[0] + j * image_template.step[1]) < min_value)
				min_value = *(image_template.data + i * image_template.step[0] + j * image_template.step[1]);
		}
	}

	med_value = (int)((max_value - min_value) / 2 + min_value);

	// 根据med_value构建卷积核

	Mat template_mask = Mat::zeros(template_size, template_size, CV_8UC1);

	for (int i = 0; i < image_template.rows; ++i)
	{
		for (int j = 0; j < image_template.cols; ++j)
		{
			if (*(image_template.data + i * image_template.step[0] + j * image_template.step[1]) <= med_value)
			{
				*(template_mask.data + i * template_mask.step[0] + j * template_mask.step[1]) = 1;
			}
		}
	}


	vector<Point2i> vector_template;
	int x_index = init_point_index.x;
	int y_index = init_point_index.y;


	//初始化第一个索引模板
	for (int i = 0; i < vector_template_size; ++i)
	{
		int min_x_index = 0, min_y_index = 0, min_sum_value = 9999;

		for (int bias = -1 * move_threshold; bias <= move_threshold; ++bias)
		{
			int local_x_index = x_index + 1;
			int local_y_index = y_index + bias;
			int local_sum_value = 0;

			int tem_x = 0, tem_y = 0;

			// 当前值模板与值模板差值，再卷积核卷积，求出最小值对应的模板中心索引
			for (int j = local_x_index - template_radius; j <= local_x_index + template_radius; ++j)
			{
				for (int k = local_y_index - template_radius; k <= local_y_index + template_radius; ++k)
				{
					int mask_value = *(template_mask.data + tem_x * template_mask.step[0] + tem_y * template_mask.step[1]);
					int global_tem_value = *(image_template.data + tem_x * image_template.step[0] + tem_y * image_template.step[1]);
					int local_tem_value = *(undistort_image.data + j * undistort_image.step[0] + k * undistort_image.step[1]);
					local_sum_value += abs(global_tem_value - local_tem_value)*mask_value;
					tem_y++;
				}
				tem_y = 0;
				tem_x++;
			}

			if (local_sum_value < min_sum_value)
			{
				min_sum_value = local_sum_value;
				min_x_index = local_x_index;
				min_y_index = local_y_index;
			}

		}
		//存储到vector_template中		
		vector_template.push_back(Point2i(min_x_index, min_y_index));
		*(final_line.data + min_x_index * final_line.step[0] + min_y_index * final_line.step[1]) = 255;

		x_index = min_x_index;
		y_index = min_y_index;

	}

	
	Mat matrix_template = Mat(vector_template_size, 3, CV_64FC1);
	for (int j = 0; j < vector_template_size; ++j)
	{		
		*((double*)(matrix_template.data + j * matrix_template.step[0] + 0 * matrix_template.step[1])) = vector_template[j].x;//
		*((double*)(matrix_template.data + j * matrix_template.step[0] + 1 * matrix_template.step[1])) = vector_template[j].y;//
		*((double*)(matrix_template.data + j * matrix_template.step[0] + 2 * matrix_template.step[1])) = 1.0;
	}
	
	//初始化 V和D
	Mat mat_square = matrix_template.t()*matrix_template;
	Mat D, V;
	eigen(mat_square, D, V);

	// 根据后续数据，更新当前的索引模板，判断遇到环的条件，并特殊处理
	int template_size_vector = vector_template_size;

	for (int i = init_point_row + vector_template_size + 1; i < end_y_index; ++i)
	{
		//定间距减小 索引模板的长度 ，为了让电线曲率更容易改变  可调整
		if (i % (undistort_image.rows/100) == 0)
		{
			// 更新索引模板，将正确索引写入final_line中
			for (int j = 0; j < template_size_vector - 1; ++j)
			{
				vector_template[j] = vector_template[j + 1];
			}
			vector_template.pop_back();
			//cout << "vector_template.size(): " << vector_template.size() << endl;
			template_size_vector--;
			//cout << "template_size_vector: " << template_size_vector << endl;
		}

		int min_x_index = 0, min_y_index = 0, min_sum_value = 9999;

		for (int bias = -1 * move_threshold; bias <= move_threshold; ++bias)
		{
			int local_x_index = x_index + 1;
			int local_y_index = y_index + bias;
			int local_sum_value = 0;

			int tem_x = 0, tem_y = 0;

			// 当前值模板与值模板差值，再卷积核卷积，求出最小值对应的模板中心索引
			for (int j = local_x_index - template_radius; j <= local_x_index + template_radius; ++j)
			{
				for (int k = local_y_index - template_radius; k <= local_y_index + template_radius; ++k)
				{
					int mask_value = *(template_mask.data + tem_x * template_mask.step[0] + tem_y * template_mask.step[1]);
					int global_tem_value = *(image_template.data + tem_x * image_template.step[0] + tem_y * image_template.step[1]);
					int local_tem_value = *(undistort_image.data + j * undistort_image.step[0] + k * undistort_image.step[1]);
					local_sum_value += abs(global_tem_value - local_tem_value)*mask_value;
					tem_y++;
				}
				tem_y = 0;
				tem_x++;
			}



			if (local_sum_value < min_sum_value)
			{
				min_sum_value = local_sum_value;
				min_x_index = local_x_index;
				min_y_index = local_y_index;
			}
		}

	
		// 由索引模板，判断是否遇到环		
		int pre_pre_y_index = vector_template[template_size_vector - 2].y;
		int pre_y_index = vector_template[template_size_vector - 1].y;

		double min_eigenvalues = 1e20;
		int min_index = -1;
		for (int j = 0; j < D.rows; ++j)
		{
			if (*((double*)(D.data + j * D.step[0] + 0 * D.step[1])) <= min_eigenvalues)
			{
				min_index = j;
				min_eigenvalues = *((double*)(D.data + j * D.step[0] + 0 * D.step[1]));
			}
		}
		double min_eigen_vector_value1 = *((double*)(V.data + min_index * V.step[0] + 0 * V.step[1]));
		double min_eigen_vector_value2 = *((double*)(V.data + min_index * V.step[0] + 1 * V.step[1]));
		double min_eigen_vector_value3 = *((double*)(V.data + min_index * V.step[0] + 2 * V.step[1]));


		double value1 = abs((min_eigen_vector_value1*min_x_index + min_eigen_vector_value2 * min_y_index + min_eigen_vector_value3 * 1) / sqrt(pow(min_eigen_vector_value1, 2) + pow(min_eigen_vector_value2, 2)));

		//正常
		if (value1 < 1)
		{
			status_vector.push_back(-1);
			//cout << value1 << endl;
			// 更新索引模板，将正确索引写入final_line中
			for (int j = 0; j < template_size_vector - 1; ++j)
			{
				vector_template[j] = vector_template[j + 1];
			}

			vector_template[template_size_vector - 1] = Point2i(min_x_index, min_y_index);

			*(final_line.data + min_x_index * final_line.step[0] + min_y_index * final_line.step[1]) = 255;

			x_index = min_x_index;
			y_index = min_y_index;

		}
		//修正
		else
		{
			status_vector.push_back(1);

			int value2 = static_cast<int>(-(min_eigen_vector_value1*min_x_index + min_eigen_vector_value3) / min_eigen_vector_value2);

			if (value2 >= template_radius && value2 < undistort_image.cols - template_radius)
			{
				// 更新索引模板，将正确索引写入final_line中
				for (int j = 0; j < template_size_vector - 1; ++j)
				{
					vector_template[j] = vector_template[j + 1];
				}

				vector_template[template_size_vector - 1] = Point2i(min_x_index, value2);

				*(final_line.data + min_x_index * final_line.step[0] + value2 * final_line.step[1]) = 255;

				x_index = min_x_index;
				y_index = value2;

			}
		}

		// 是否更新7*7 值模板，更新的话可以使值模板更符合当前的值。  可调整，不需要就注释掉
		if ((i - (init_point_row + vector_template_size + 1)) % update_threshold == (update_threshold - 1))
		{
			int status_sum = 0;
			for (int j = 0; j < value_threshold; ++j)
			{
				status_sum += abs(status_vector[i - (init_point_row + vector_template_size + 1) - j] - status_vector[i - (init_point_row + vector_template_size + 1) - j - 1]);
			}

			if (status_sum > 0)
			{
				image_template.empty();
				image_template = undistort_image(Rect(y_index - template_radius, x_index - template_radius, template_size, template_size));
			}
		}

		// 更新 V和D

		Mat matrix_template = Mat(template_size_vector, 3, CV_64FC1);
		for (int j = 0; j < template_size_vector; ++j)
		{
			*((double*)(matrix_template.data + j * matrix_template.step[0] + 0 * matrix_template.step[1])) = vector_template[j].x;
			*((double*)(matrix_template.data + j * matrix_template.step[0] + 1 * matrix_template.step[1])) = vector_template[j].y;
			*((double*)(matrix_template.data + j * matrix_template.step[0] + 2 * matrix_template.step[1])) = 1.0;
		}

		mat_square.empty();
		mat_square = matrix_template.t()*matrix_template;
		D.empty();
		V.empty();
		eigen(mat_square, D, V);
		
	}
}