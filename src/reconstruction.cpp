#include "Reconstruction.h"

#define IMAGEH 3000
#define IMAGEW 4096

#define init_point_row 11 //初始化的行数
#define vector_template_size 100

#define MATCH_THRESHOLD 1 //对极线点匹配最小阈值
#define DISTANCE_THRESHOLD 1  //点云匹配距离阈值

#define MAX_DEPTH 100 //点云滤波参数
#define MIN_DEPTH 60

#define TRIM_DISTANCE 0.5 //截断图像，终止点找的不好

void Reconstruction::pose_estimation_2d2d(vector<Point2f> points1, vector<Point2f> points2, vector<Point2f> match)
{
	//转换成Mat 格式
	int size = match.size();
	Mat p1(size, 2, CV_32FC1);
	Mat p2(size, 2, CV_32FC1);

	//计算F
	Mat fundamental_matrix;
	vector<uchar> m_RANSACStatus;
	fundamental_matrix = findFundamentalMat(points2, points1, m_RANSACStatus, FM_RANSAC);
	std::cout << "opencv 算的基础矩阵F： " << endl << fundamental_matrix << endl;

	/*Mat image_l = imread(image_path_l);
	Mat image_r = imread(image_path_r);
	DrawEpiLines(image_r, image_l, points2, points1, fundamental_matrix);*/

	// 计算野点个数
	int OutlinerCount = 0;
	for (int i = 0; i < size; i++)
	{
		if (m_RANSACStatus[i] == 0) // 状态为0表示野点
		{
			OutlinerCount++;
		}
	}

	std::cout << "total size: " << size << " F  OutlinerCount" << OutlinerCount << endl;


	////计算误差
	//CaculateError(points2, points1, fundamental_matrix);


	// 相机内参
	/*Eigen::Matrix3d I = Eigen::Matrix3d::Identity();*/

	Mat K_l1 = (Mat_<double>(3, 3) << 3.7612906067774788e+003, 0., 2.0457995013785016e+003, 0.,
		3.7159736130516289e+003, 1.4796241830921265e+003, 0., 0., 1.);

	/*Mat K_m1 = (Mat_<double>(3, 3) << 3.7516261809333914e+003, 0., 2.0321668349416393e+003, 0.,
	3.7075460080696898e+003, 1.4757973862050546e+003, 0., 0., 1.);*/

	Mat K_r1 = (Mat_<double>(3, 3) << 3.7604405383652875e+003, 0., 2.0415438075955501e+003, 0.,
		3.7160779331549415e+003, 1.4845370511494227e+003, 0., 0., 1.);


	Mat essential_matrix = K_l1.t() * fundamental_matrix * K_r1;

	std::cout << "本质矩阵E: " << endl << essential_matrix << endl;


	Mat R, t;

	//-- 从本质矩阵中恢复旋转和平移信息.
	recoverPose(essential_matrix, points2, points1, K_l1, R, t);


	std::cout << "R : " << endl << R << endl;
	std::cout << "t : " << endl << t << endl;


	//利用 R,t计算 F

	//-- 验证E=t^R*scale
	Mat t_x = (Mat_<double>(3, 3) <<
		0, -t.at<double>(2, 0), t.at<double>(1, 0),
		t.at<double>(2, 0), 0, -t.at<double>(0, 0),
		-t.at<double>(1.0), t.at<double>(0, 0), 0);

	std::cout << "t^R=" << endl << t_x * R << endl;

	cv::Mat mat;
	recoverPose(essential_matrix, points2, points1, K_l1, R, t, 99999, cv::noArray(), mat);

	//SavePLYFile_PointCloud("C:/Users/1/Desktop/data-total/test.ply", mat);

}

void Reconstruction::solve_pnp_2d_3d_ceres(string ctrp_path)
{

}

void Reconstruction::CalculateStructure_Init_DLT(std::vector<View> view, std::vector<Observation> obs , vector<Structure> & P )
{
	std::cout << "START INIT POINTCLOUD" << endl;
	for (int i = 0; i < 2; ++i)
	{
		for (int j = 0; j < obs.size(); j++)
		{
			Eigen::Matrix<double, 3, 4> P1, P2;

			int host_camera_id = (i == 0 ? obs[j].host_camera_id : obs[j].neighbor_camera_id);
			int neighbor_camera_id = (i == 0 ? obs[j].neighbor_camera_id : obs[j].host_camera_id);

			P1.block(0, 0, 3, 3) = view[host_camera_id].rotation;
			P1.block(0, 3, 3, 1) = -view[host_camera_id].rotation * view[host_camera_id].center;
			P1 = view[host_camera_id].K * P1;

			P2.block(0, 0, 3, 3) = view[neighbor_camera_id].rotation;
			P2.block(0, 3, 3, 1) = -view[neighbor_camera_id].rotation * view[neighbor_camera_id].center;
			P2 = view[neighbor_camera_id].K * P2;

			Eigen::Vector2d pixel = (i == 0 ? obs[j].pixel : obs[j].match_pixel);
			Eigen::Vector2d match_pixel = (i == 0 ? obs[j].match_pixel : obs[j].pixel);

			Eigen::Vector3d X_init;
			Eigen::Matrix<double, 4, 4> A;
			A.block(0, 0, 1, 4) =
				(pixel.x()) * P1.row(2) - P1.row(0);
			A.block(1, 0, 1, 4) =
				(pixel.y()) * P1.row(2) - P1.row(1);
			A.block(2, 0, 1, 4) =
				(match_pixel.x()) * P2.row(2) - P2.row(0);
			A.block(3, 0, 1, 4) =
				(match_pixel.y()) * P2.row(2) - P2.row(1);

			Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinV);
			Eigen::Vector4d X_ = svd.matrixV().col(3);
			X_init << X_.x() / X_.w(), X_.y() / X_.w(), X_.z() / X_.w();
			if (X_init.z() < MIN_DEPTH || X_init.z() > MAX_DEPTH)
			{
				P.emplace_back(Structure(X_init,i*obs.size()+j, false));
			}
			else
				P.emplace_back(Structure(X_init, i*obs.size() + j, true));
		}
	}
	std::cout << "FINISH INIT POINTCLOUD" << endl;
}

void Reconstruction::CalculateStructure_Ceres(std::vector<View> view, std::vector<Observation> obs, vector<Structure> & output_struct)
{
	std::cout << "SATRT  BA" << endl;
	ceres::Problem problem;
	ceres::LossFunction* lossFunc = new ceres::HuberLoss(2.0f);

	for (uint i = 0; i < output_struct.size(); ++i)
	{
		if (!output_struct[i].isvalid_structure)
			continue;
		
		int current_index = i / obs.size();
		Eigen::Matrix<double, 3, 4> P;

		P.block(0, 0, 3, 3) = view[current_index].rotation;
		P.block(0, 3, 3, 1) = -view[current_index].rotation * view[current_index].center;
		P = view[current_index].K * P;
		Eigen::Vector2d pixel = (current_index == 0 ? obs[i].pixel : obs[i - obs.size()].match_pixel);

		/*ceres::CostFunction* cost_function =
			new ceres::AutoDiffCostFunction<Ceres_Triangulate, 2, 3>(new Ceres_Triangulate(pixel, P));*/
		ceres::CostFunction* cost_function =
			new ceres::AutoDiffCostFunction<Ceres_Triangulate_AdjustRt, 2, 3, 3, 3>(new Ceres_Triangulate_AdjustRt(pixel, view[current_index].K ));
		problem.AddResidualBlock(cost_function, lossFunc, output_struct[i].positions_array, view[current_index].rotation_array, view[current_index].translatrion_array);
	}

	ceres::Solver::Options options;
	options.max_num_iterations = 20;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	std::cout << "FINISH CERES BA" << endl;;
}

void Reconstruction::DrawEpiLines(const Mat& img_1, const Mat& img_2, vector<Point2f>points1, vector<Point2f>points2, cv::Mat F,string save_path)
{
	std::vector<cv::Vec<float, 3>> epilines;
	cv::computeCorrespondEpilines(points1, 1, F, epilines);//计算对应点的外极线epilines是一个三元组(a,b,c)，表示点在另一视图中对应的外极线ax+by+c=0;
														   //将图片转换为RGB图，画图的时候外极线用彩色绘制
	cv::Mat img1, img2;
	if (img_1.type() == CV_8UC3)
	{
		img_1.copyTo(img1);
		img_2.copyTo(img2);
	}
	else if (img_1.type() == CV_8UC1)
	{
		cvtColor(img_1, img1, COLOR_GRAY2BGR);
		cvtColor(img_2, img2, COLOR_GRAY2BGR);
	}
	else
	{
		std::cout << "unknow img type\n" << endl;
		exit(0);
	}

	cv::RNG& rng = theRNG();
	for (int i = 0; i < points1.size(); i += 100)
	{
		Scalar color = Scalar(rng(256), rng(256), rng(256));//随机产生颜色
		circle(img1, points1[i], 5, color, 1);//在视图1中把关键点用圆圈画出来
		circle(img2, points2[i], 5, color, 1);//在视图2中把关键点用圆圈画出来，然后再绘制在对应点处的外极线
		line(img2, Point(0, -epilines[i][2] / epilines[i][1]), Point(img2.cols, -(epilines[i][2] + epilines[i][0] * img2.cols) / epilines[i][1]), color);
	}
	string img_pointLines = save_path + "\\img_pointLines1.png";
	string img_point = save_path + "\\img_point1.png";
	imwrite(img_point, img1);
	imwrite(img_pointLines, img2);
}

void Reconstruction::LinePointInit(Mat points_source, Mat points_target, vector<LinePoint> &source_points, vector<LinePoint> & target_points)
{
	//为源图和目标图创建电线的索引，每个点都找到归属的电线id
	/*CreateLineHash(points_source, source_points);
	CreateLineHash(points_target, target_points);	*/	

	CreateLineHashDelectly(points_source, source_points);
	CreateLineHashDelectly(points_target, target_points);
}

//x2.t()*F*x1 == 0  x1:points1  , x2:points2
void Reconstruction::FindMatchPoint(vector<LinePoint> &source_points, vector<LinePoint> & target_points, vector<LinePoint> & feature_source, vector<LinePoint> & feature_target, vector<Point2f> & match_s_t , Mat F_s_2_t,bool is_create_match_file /*, Mat F_t_2_s*/)
{
	std::cout << "START FIND MATCH POINT" << endl;
	vector<Vec<float, 3>> epilines_target ;
	vector<Point2f> points_source, points_target;
	for (int i = 0; i < source_points.size(); ++i)
	{
		points_source.push_back(Point2f(source_points[i].point.x, source_points[i].point.y));
	}
	for (int i = 0; i < target_points.size(); ++i)
	{
		points_target.push_back(Point2f(target_points[i].point.x, target_points[i].point.y));
	}
	//Epilines = F_s_2_t * p , p:points_source
	computeCorrespondEpilines(points_source, 1, F_s_2_t, epilines_target);//计算对应点的外极线epilines是一个三元组(a,b,c)，表示点在另一视图中对应的外极线ax+by+c=0;
														//将图片转换为RGB图，画图的时候外极线用彩色绘制
	//目标图对源图反算一遍对极线
	//computeCorrespondEpilines(points_target, 1, F_t_2_s, epilines_source);//计算对应点的外极线epilines是一个三元组(a,b,c)，表示点在另一视图中对应的外极线ax+by+c=0;
														//将图片转换为RGB图，画图的时候外极线用彩色绘制

	//如果不创建match file 默认feature file的点按顺序对应匹配
	if (!is_create_match_file)
	{
		for (int i = 0; i < source_points.size(); i++)
		{
			/*vector<LinePoint> output_points_in_target;
			vector<LinePoint> output_points_in_source*/;
			// 在目标图中找出过对极线的点
			//FindCorrespondPointInEpilines(source_points[i],target_points, match_source, match_target,epilines_target[i]);
			//epiline (a,b,c)  ax+by+c = 0 (a,b,c)*(x,y,1).t()
		int y_begin, y_end;
		if (epilines_target[i][1] != 0)
		{
			y_begin = ((-(epilines_target[i][2]) / epilines_target[i][1]) >= 0 ? (-(epilines_target[i][2]) / epilines_target[i][1]) : 0);
			y_end = ((-(epilines_target[i][0] * IMAGEW + epilines_target[i][2]) / epilines_target[i][1]) <= IMAGEH ? (-(epilines_target[i][0] * IMAGEW + epilines_target[i][2]) / epilines_target[i][1]) : IMAGEH);
		}
		else
		{
			y_begin = 0;
			y_end = IMAGEH;
		}
		
		for (int j = 0; j < target_points.size(); ++j)
		{

			if (!target_points[j].is_matched && y_begin <= target_points[j].point.y && target_points[j].point.y <= y_end)
			{
				
				if (abs(epilines_target[i][0] * target_points[j].point.x + epilines_target[i][1] * target_points[j].point.y + epilines_target[i][2]) < MATCH_THRESHOLD)
				{
					if (source_points[i].line_id1 == target_points[j].line_id1 || source_points[i].line_id1 == target_points[j].line_id2 || source_points[i].line_id2 == target_points[j].line_id1 || source_points[i].line_id2 == target_points[j].line_id2)
					{
						feature_source.emplace_back(source_points[i]);
						feature_target.emplace_back(target_points[j]);						
						break;

					}
				}
			}
		}
		if (i % 500 == 0)
			std::cout << "find matched :: " << (float)(1.0*i/ source_points.size()) << endl;
		}
		std::cout << "feature_source.size()" << feature_source.size() << endl;
		std::cout << "feature_target.size()" << feature_target.size() << endl;
		
	}

	else
	{
		for (int i = 0; i < source_points.size()/2; i++)
		{		
			int y_begin, y_end;
			if (epilines_target[i][1] != 0)
			{
				y_begin = ((-(epilines_target[i][2]) / epilines_target[i][1]) >= 0 ? (-(epilines_target[i][2]) / epilines_target[i][1]) : 0);
				y_end = ((-(epilines_target[i][0] * IMAGEW + epilines_target[i][2]) / epilines_target[i][1]) <= IMAGEH ? (-(epilines_target[i][0] * IMAGEW + epilines_target[i][2]) / epilines_target[i][1]) : IMAGEH);
			}
			else
			{
				y_begin = 0;
				y_end = IMAGEH;
			}
			/*std::cout << "y_begin : " << y_begin << "y_end : " << y_end << endl;*/
			for (int j = 0; j < target_points.size()/2; ++j)
			{

				if (!target_points[j].is_matched && y_begin <= target_points[j].point.y && target_points[j].point.y <= y_end)
				{
					//std::cout << epiline[0] * target_points[i].x + epiline[1] * target_points[i].y + epiline[2] << endl;
					if (abs(epilines_target[i][0] * target_points[j].point.x + epilines_target[i][1] * target_points[j].point.y + epilines_target[i][2]) < MATCH_THRESHOLD)
					{
						if (source_points[i].line_id1 == target_points[j].line_id1 || source_points[i].line_id1 == target_points[j].line_id2 || source_points[i].line_id2 == target_points[j].line_id1 || source_points[i].line_id2 == target_points[j].line_id2)
						{
							match_s_t.push_back(Point2i(i, j));
							break;
						}
					}
				}
			}
			if (i % 500 == 0)
				std::cout << "find matched :: " << (float)(1.0*i / source_points.size()) << endl;
		}

		for (int i = 0; i < source_points.size(); ++i)
		{
			feature_source.emplace_back(source_points[i]);
		}
		for (int i = 0; i < target_points.size(); ++i)
		{
			feature_target.emplace_back(target_points[i]);
		}
		
		std::cout << "feature_source.size()" << feature_source.size() << endl;
		std::cout << "feature_target.size()" << feature_target.size() << endl;
	}
}

void Reconstruction::FindMatchPointByProjection(vector<LinePoint> &source_points, vector<LinePoint> & target_points, vector<LinePoint> source_points_center, vector<LinePoint> target_points_center ,vector<LinePoint> & feature_source, vector<LinePoint> & feature_target, vector<Point2f> & match_s_t, Mat F_s_2_t, vector<View>  view, bool is_create_match_file /*, Mat F_t_2_s*/)
{
	std::cout << "START FIND MATCH POINT" << endl;
	Eigen::Matrix<double, 3, 4> P1, P2;

	P1.block(0, 0, 3, 3) = view[0].rotation;
	P1.block(0, 3, 3, 1) = -view[0].rotation * view[0].center;
	P1 = view[0].K * P1;

	P2.block(0, 0, 3, 3) = view[1].rotation;
	P2.block(0, 3, 3, 1) = -view[1].rotation * view[1].center;
	P2 = view[1].K * P2;


	vector<Vec<float, 3>> epilines_target, epilines_target_center;
	vector<Point2f> points_source, points_target, points_source_center, points_target_center;
	for (int i = 0; i < source_points.size(); ++i)
	{
		points_source.push_back(Point2f(source_points[i].point.x, source_points[i].point.y));
	}
	for (int i = 0; i < target_points.size(); ++i)
	{
		points_target.push_back(Point2f(target_points[i].point.x, target_points[i].point.y));
	}
	for (int i = 0; i < source_points_center.size(); ++i)
	{
		points_source_center.push_back(Point2f(source_points_center[i].point.x, source_points_center[i].point.y));
	}
	for (int i = 0; i < target_points_center.size(); ++i)
	{
		points_target_center.push_back(Point2f(target_points_center[i].point.x, target_points_center[i].point.y));
	}



	//Epilines = F_s_2_t * p , p:points_source
	computeCorrespondEpilines(points_source, 1, F_s_2_t, epilines_target);//计算对应点的外极线epilines是一个三元组(a,b,c)，表示点在另一视图中对应的外极线ax+by+c=0;
														//将图片转换为RGB图，画图的时候外极线用彩色绘制
	//Epilines = F_s_2_t * p , p:points_source
	computeCorrespondEpilines(points_source_center, 1, F_s_2_t, epilines_target_center);//计算对应点的外极线epilines是一个三元组(a,b,c)，表示点在另一视图中对应的外极线ax+by+c=0;
														//将图片转换为RGB图，画图的时候外极线用彩色绘制

			
#pragma omp parallel for  
	for (int i = 0; i < source_points.size(); i++)
	{
		int y_begin, y_end;
		if (epilines_target[i][1] != 0)
		{
			y_begin = ((-(epilines_target[i][2]) / epilines_target[i][1]) >= 0 ? (-(epilines_target[i][2]) / epilines_target[i][1]) : 0);
			y_end = ((-(epilines_target[i][0] * IMAGEW + epilines_target[i][2]) / epilines_target[i][1]) <= IMAGEH ? (-(epilines_target[i][0] * IMAGEW + epilines_target[i][2]) / epilines_target[i][1]) : IMAGEH);
		}
		else
		{
			y_begin = 0;
			y_end = IMAGEH;
		}

		//找到该点与源图中心线哪一点最近，求得中心线最近点的3D点
		int min_x_distance = 9999;
		int min_x_index = -1;
		int search_start_index = 0; //为了使用多线程，将这个全局变量起到局部变量，同时下面的循环从0开始而不是从search_start_index开始


		for (int j = 0; j < points_source_center.size(); ++j)
		{
			if (points_source[i].y == points_source_center[j].y)
			{
				if (abs(points_source[i].x - points_source_center[j].x) < min_x_distance)
				{
					min_x_distance = abs(points_source[i].x - points_source_center[j].x);
					min_x_index = points_source_center[j].x;
					search_start_index = j;
				}
			}
		}

		//std::cout << "search_start_index" << search_start_index << endl;
		//std::cout << "points_source[i] : " << points_source[i] << "   " << min_x_index << "  " << points_source[i].y <<"  "<< points_source_center[search_start_index].x<<"  "<< points_source_center[search_start_index].y<< endl;
		//记录该中心点的坐标
		Eigen::Vector2d p_s = Eigen::Vector2d(min_x_index, points_source[i].y);
		Eigen::Vector2d p_t = Eigen::Vector2d(0, 0);
		//现在在目标中心线上找出对应点
		bool is_valid_value = false;
		for (int j = 0; j < target_points_center.size(); ++j)
		{

			if (!target_points_center[j].is_matched && y_begin <= target_points_center[j].point.y && target_points_center[j].point.y <= y_end)
			{

				if (abs(epilines_target_center[search_start_index][0] * target_points_center[j].point.x + epilines_target_center[search_start_index][1] * target_points_center[j].point.y + epilines_target_center[search_start_index][2]) < MATCH_THRESHOLD)
				{

					if (source_points_center[search_start_index].line_id1 == target_points_center[j].line_id1)
					{
						p_t = Eigen::Vector2d(target_points_center[j].point.x, target_points_center[j].point.y);
						is_valid_value = true;
						//std::cout << "test pt" << p_t << endl;
						break;
					}
				}
			}
		}

		if (is_valid_value)
		{
			Eigen::Vector3d X_center;
			Eigen::Matrix<double, 4, 4> A;
			A.block(0, 0, 1, 4) =
				(p_s.x()) * P1.row(2) - P1.row(0);
			A.block(1, 0, 1, 4) =
				(p_s.y()) * P1.row(2) - P1.row(1);
			A.block(2, 0, 1, 4) =
				(p_t.x()) * P2.row(2) - P2.row(0);
			A.block(3, 0, 1, 4) =
				(p_t.y()) * P2.row(2) - P2.row(1);

			Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinV);
			//求得中心线对应的3D点
			Eigen::Vector4d X_ = svd.matrixV().col(3);
			X_center << X_.x() / X_.w(), X_.y() / X_.w(), X_.z() / X_.w();
			//std::cout << "X_center坐标是： " << X_center << endl;
			//std::cout << "源点坐标是： " << points_source[i] << " 中心点坐标是： " << min_x_index << " ," << points_source[i].y << endl;

			//找到中心点的3D坐标后，利用该3D点筛选匹配点

			Eigen::Vector2d p_s_c = Eigen::Vector2d(source_points[i].point.x, source_points[i].point.y);//原图像中的待匹配点
			double min_point_distance = 99999.0;
			int min_point_index = -1;
			Eigen::Vector3d min_X_candidate;

			for (int j = 0; j < target_points.size(); ++j)
			{

				if (!target_points[j].is_matched && y_begin <= target_points[j].point.y && target_points[j].point.y <= y_end)
				{
					//std::cout << epiline[0] * target_points[i].x + epiline[1] * target_points[i].y + epiline[2] << endl;
					if (abs(epilines_target[i][0] * target_points[j].point.x + epilines_target[i][1] * target_points[j].point.y + epilines_target[i][2]) < MATCH_THRESHOLD)
					{
						//生成3D候选点，然后选择与中心点最近的候选点作为匹配点
						Eigen::Vector3d X_candidate;
						Eigen::Matrix<double, 4, 4> A_c;
						Eigen::Vector2d p_t_c = Eigen::Vector2d(target_points[j].point.x, target_points[j].point.y);//目标图像中的候选匹配点
						A_c.block(0, 0, 1, 4) =
							(p_s_c.x()) * P1.row(2) - P1.row(0);
						A_c.block(1, 0, 1, 4) =
							(p_s_c.y()) * P1.row(2) - P1.row(1);
						A_c.block(2, 0, 1, 4) =
							(p_t_c.x()) * P2.row(2) - P2.row(0);
						A_c.block(3, 0, 1, 4) =
							(p_t_c.y()) * P2.row(2) - P2.row(1);

						Eigen::JacobiSVD<Eigen::MatrixXd> svd(A_c, Eigen::ComputeThinV);
						//求得中心线对应的3D点
						Eigen::Vector4d X_c_ = svd.matrixV().col(3);
						X_candidate << X_c_.x() / X_c_.w(), X_c_.y() / X_c_.w(), X_c_.z() / X_c_.w();
						//cout << "X_candidate is: " << X_candidate << endl;

						double current_distance = pow((X_candidate.x() - X_center.x()), 2) + pow((X_candidate.y() - X_center.y()), 2) + pow((X_candidate.z() - X_center.z()), 2);
						if (current_distance < min_point_distance)
						{
							min_point_distance = current_distance;
							min_point_index = j;
							min_X_candidate = X_candidate;
						}
					}
				}
			}
			//cout << "min_point_distance is" << min_point_distance << endl;
			//cout << " min_X_candidate is:  " << min_X_candidate << endl;

			//找到有效的匹配点了

			if (is_create_match_file)
			{

#pragma omp critical  //同一时间该代码只能被单一线程执行
				{
					if (min_point_index != -1 && min_point_distance < DISTANCE_THRESHOLD)
					{

						match_s_t.push_back(Point2i(i, min_point_index));
					}
				}
			}

			else
			{
#pragma omp critical  //同一时间该代码只能被单一线程执行
				{
					//如果不创建match file 默认feature file的点按顺序对应匹配
					if (min_point_index != -1 && min_point_distance < DISTANCE_THRESHOLD)
					{
						feature_source.emplace_back(source_points[i]);
						feature_target.emplace_back(target_points[min_point_index]);
					}
				}
			}
		}

		if (i % 500 == 0)
			std::cout << "find matched :: " << (float)(1.0*i / source_points.size()) << endl;
	}

		std::cout << "match_s_t size: " << match_s_t.size() << endl;

		if (is_create_match_file)
		{
			for (int i = 0; i < source_points.size(); ++i)
			{
				feature_source.emplace_back(source_points[i]);
			}
			for (int i = 0; i < target_points.size(); ++i)
			{
				feature_target.emplace_back(target_points[i]);
			}
		}

		std::cout << "feature_source.size()" << feature_source.size() << endl;
		std::cout << "feature_target.size()" << feature_target.size() << endl;
	
}

void Reconstruction::EraseInvalidStructure(std::vector<Structure> & structure , std::vector<Observation>& observation)
{	
	std::cout << "START REASE INVALID STRUCTURE" << endl;
	std::vector<Structure>::iterator iter = structure.begin();
	for (std::vector<Observation>::iterator iter_o = observation.begin(); iter_o != observation.end();)
	{
		if (!iter->isvalid_structure)
		{
			iter = structure.erase(iter);
			iter_o = observation.erase(iter_o);
			if (iter_o == observation.end())
				break;
		}
		else
		{
			iter++;
			iter_o++;
		}
	}

	for (std::vector<Structure>::iterator iter_s = structure.begin(); iter_s != structure.end();)
	{
		if (!iter_s->isvalid_structure)
		{
			iter_s = structure.erase(iter_s);
			if (iter_s == structure.end())
				break;
		}
		else
		{
			iter_s++;			
		}
	}

}

void Reconstruction::SavePLYFile_PointCloud(std::string filePath, std::vector<Structure> structure)
{
	ofstream ofs(filePath);
	if (!ofs)
	{
		std::cout << "err in create ply!" << endl;;
		return;
	}
	else
	{
		std::cout << "BEGIN SAVE PLY" << endl;
		ofs << "ply " << endl << "format ascii 1.0" << endl;
		//ofs << "element vertex " << this->structures.size() + this->Views.size() << endl;//old
		ofs << "element vertex " << structure.size()  << endl;
		ofs << "property float x" << endl << "property float y" << endl << "property float z" << endl;
		ofs << "property uchar blue"
			<< endl << "property uchar green"
			<< endl << "property uchar red" << endl;
		ofs << "end_header" << endl;

		
		for (uint i = 0; i < structure.size(); i++)
		{
			ofs << structure[i].positions_array[0] << " " << structure[i].positions_array[1] << " " << structure[i].positions_array[2]
				<< " 255 255 255" << endl;					
		}
	}
	ofs.close();
	ofs.flush();
	std::cout << "FINISH SAVE PLY" << endl;
}

void FindCorrespondPointInEpilines(LinePoint source_point, vector<LinePoint>target_points, vector<Point2f> & match_source, vector<Point2f> & match_target, Vec<float, 3> epiline)
{
	//epiline (a,b,c)  ax+by+c = 0 (a,b,c)*(x,y,1).t()
	int y_begin, y_end;
	if (epiline[1] != 0)
	{
		y_begin = ((-(epiline[2]) / epiline[1]) >= 0 ? (-(epiline[2]) / epiline[1]) : 0);
		y_end = ((-(epiline[0] * IMAGEW + epiline[2]) / epiline[1]) <= IMAGEH ? (-(epiline[0] * IMAGEW + epiline[2]) / epiline[1]) : IMAGEH);
	}
	else
	{
		y_begin = 0;
		y_end = IMAGEH;
	}
	//std::cout << "y_begin : " << y_begin << "y_end : " << y_end << endl;
	for (int i = 0; i < target_points.size(); ++i)
	{
		
		if (y_begin <= target_points[i].point.y && target_points[i].point.y <= y_end)
		{		
			//std::cout << epiline[0] * target_points[i].x + epiline[1] * target_points[i].y + epiline[2] << endl;
			if (abs(epiline[0] * target_points[i].point.x + epiline[1] * target_points[i].point.y + epiline[2]) < 0.3)
			{
				if (source_point.line_id1 == target_points[i].line_id1 || source_point.line_id1 == target_points[i].line_id2 || source_point.line_id2 == target_points[i].line_id1 || source_point.line_id2 == target_points[i].line_id2)
				{
					match_source.push_back(Point2f(source_point.point.x, source_point.point.y));
					match_target.push_back(Point2f(target_points[i].point.x, target_points[i].point.y));
					break;
				}
			}
		}
	}

	
}

//重载
void FindCorrespondPointInEpilines(LinePoint source_point, LinePoint Inspection_point, vector<LinePoint> & output_points, Vec<float, 3> epiline)
{
	if (abs(epiline[0] * source_point.point.x + epiline[1] * source_point.point.y + epiline[2]) < 0.3)
		output_points.emplace_back(Inspection_point);
}

void FindMatchHash(vector<float> x_index_l, Mat points_source,vector<LinePoint> &  source_points, int left_size,bool is_left_side)
{
	//针对大连的中心线的特点
	
	//if (x_index_l.size() == 2 || x_index_l.size()==3)
	//{
	//	for (int i = 0; i < points_source.rows; ++i)
	//	{
	//		int line_count = 0;
	//		int left_line_count = 0;
	//		for (int j = 0; j < points_source.cols && line_count < x_index_l.size(); ++j)
	//		{
	//			if (*(points_source.data + i * points_source.step[0] + j * points_source.step[1]) != 0)
	//			{
	//				if (is_left_side)
	//				{
	//					source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, line_count + 1, line_count + 1));
	//					line_count++;
	//				}
	//				//如果是右侧线，line_id 需要加上左侧线的数目
	//				else
	//				{
	//					if (left_line_count++ < left_size)
	//						continue;
	//					source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, x_index_l.size() +line_count + 1, x_index_l.size() +line_count + 1));
	//					line_count++;
	//				}
	//			}
	//		}
	//	}
	//}

	//else
	//{
	//	std::cout << "数据不符合预设" << endl;
	//}


	// 针对湖州数据的电线特点 
	if (x_index_l.size() == 7)
	{
		for (int i = 0; i < points_source.rows; ++i)
		{
			int line_count = 0;
			int left_line_count = 0;
			for (int j = 0; j < points_source.cols && line_count < 7; ++j)
			{
				if (*(points_source.data + i * points_source.step[0] + j * points_source.step[1]) != 0)
				{
					if (is_left_side)
					{
						source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, line_count + 1, line_count + 1));
						line_count++;
					}
					//如果是右侧线，line_id 需要加上左侧线的数目
					else
					{
						if (left_line_count++ < left_size)
							continue;
						source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+line_count + 1, 7+line_count + 1));
						line_count++;
					}
				}
			}
		}
	}

	else if (x_index_l.size() == 4)
	{
		for (int i = 0; i < points_source.rows; ++i)
		{
			int line_count = 0;
			int left_line_count = 0;
			for (int j = 0; j < points_source.cols && line_count < 4; ++j)
			{
				if (*(points_source.data + i * points_source.step[0] + j * points_source.step[1]) != 0)
				{
					if (is_left_side)
					{
						if (line_count != 3)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, (line_count + 1) * 2 - 1, (line_count + 1) * 2));
						
						}
						else
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, (line_count + 1) * 2 - 1, (line_count + 1) * 2 - 1));
							
						}
						line_count++;
					}
					else
					{
						if (left_line_count++ < left_size)
							continue;

						if (line_count != 3)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+(line_count + 1) * 2 - 1, 7+(line_count + 1) * 2));
							
						}
						if (line_count == 3)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+7 , 7+7));
							
						}
						line_count++;
					}

				}
			}
		}
	}
	//这是用提取的中心线完成匹配，6+1 + 1+6 -> 1+1 + 1+1
	else if (x_index_l.size() == 2)
	{
		for (int i = 0; i < points_source.rows; ++i)
		{
			int line_count = 0;
			int left_line_count = 0;
			for (int j = 0; j < points_source.cols && line_count < 2; ++j)
			{
				if (*(points_source.data + i * points_source.step[0] + j * points_source.step[1]) != 0)
				{
					if (is_left_side)
					{
						source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, line_count + 1, line_count + 1));
						line_count++;
					}
					//如果是右侧线，line_id 需要加上左侧线的数目
					else
					{
						if (left_line_count++ < left_size)
							continue;
						source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 2 + line_count + 1, 2 + line_count + 1));
						line_count++;
					}
				}
			}
		}
	}

	//1+1+2+1  需要判断哪两条线不重合
	else if (x_index_l.size() == 5)
	{
		int min_distance = 9999;
		int min_index_l = -1, min_index_r = -1;
		for (int i = 0; i < x_index_l.size() - 1; ++i)
		{
			if ((x_index_l[i + 1] - x_index_l[i]) < min_distance)
			{
				min_index_l = i;
				min_index_r = i + 1;
				min_distance = x_index_l[i + 1] - x_index_l[i];
			}
		}
		//1,2不重
		if (min_index_l == 0)
		{
			for (int i = 0; i < points_source.rows; ++i)
			{
				int line_count = 0;
				int left_line_count = 0; //当是右边的线时，左边的线不读取
				for (int j = 0; j < points_source.cols && line_count < 5; ++j)
				{
					if (*(points_source.data + i * points_source.step[0] + j * points_source.step[1]) != 0)
					{
						if (is_left_side)
						{
							if (line_count == 0 || line_count == 1)
							{
								source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, line_count + 1, line_count + 1));
								
							}
							if (line_count == 2 || line_count == 3)
							{
								source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 2 * line_count - 1, 2 * line_count));
								
							}
							if (line_count == 4)
							{
								source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7, 7));								
							}
							line_count++;
						}

						else
						{
							if (left_line_count++ < left_size)
								continue;

							if (line_count == 0 || line_count == 1)
							{
								source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+ line_count + 1, 7+line_count + 1));
								
							}
							if (line_count == 2 || line_count == 3)
							{
								source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+ 2 * line_count - 1, 7+ 2 * line_count));
								
							}
							if (line_count == 4)
							{
								source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+7, 7+7));
								
							}
							line_count++;
						}
					}
				}
			}

		}

		//2,3不重
		if (min_index_l == 1)
		{
			for (int i = 0; i < points_source.rows; ++i)
			{
				int line_count = 0;
				int left_line_count = 0;
				for (int j = 0; j < points_source.cols && line_count < 5; ++j)
				{
					if (*(points_source.data + i * points_source.step[0] + j * points_source.step[1]) != 0)
					{
						if (is_left_side)
						{
							if (line_count == 0)
							{
								source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 1, 2));
								
							}
							if (line_count == 1 || line_count == 2)
							{
								source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, line_count + 2, line_count + 2));
								
							}
							if (line_count == 3)
							{
								source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 5, 6));
								
							}
							if (line_count == 4)
							{
								source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7, 7));								
							}
							line_count++;
						}

						else
						{
							if (left_line_count++ < left_size)
								continue;

							if (line_count == 0)
							{
								source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+ 1, 7+ 2));
								
							}
							if (line_count == 1 || line_count == 2)
							{
								source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+line_count + 2, 7+ line_count + 2));
								
							}
							if (line_count == 3)
							{
								source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+ 5, 7+ 6));
								
							}
							if (line_count == 4)
							{
								source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+ 7, 7+7));								
							}
							line_count++;
						}
					}
				}
			}

		}

		//3,4不重
		if (min_index_l == 2)
		{
			for (int i = 0; i < points_source.rows; ++i)
			{
				int line_count = 0;
				int left_line_count = 0;
				for (int j = 0; j < points_source.cols && line_count < 5; ++j)
				{
					if (*(points_source.data + i * points_source.step[0] + j * points_source.step[1]) != 0)
					{
						if (is_left_side)
						{
							if (line_count == 0 || line_count == 1)
							{
								source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 2 * (line_count + 1) - 1, 2 * (line_count + 1)));							
								
							}
							if (line_count == 2 || line_count == 3)
							{
								source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, line_count + 3, line_count + 3));							
								
							}
							if (line_count == 4)
							{
								source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7, 7));
								
							}
							line_count++;
						}

						else
						{
							if (left_line_count++ < left_size)
								continue;

							if (line_count == 0 || line_count == 1)
							{
								source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+2 * (line_count + 1) - 1, 7+2 * (line_count + 1)));
								
							}
							if (line_count == 2 || line_count == 3)
							{
								source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+line_count + 3, 7+line_count + 3));
								
							}
							if (line_count == 4)
							{
								source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+ 7, 7+ 7));
								
							}
							line_count++;
						}

					}
				}
			}

		}
	}

	//1+2+2+1  需要判断哪两条线重合了
	else if (x_index_l.size() == 6)
	{
		int first_min_distance = 9999, second_min_distance = 9999;
		int first_min_index_l = -1, first_min_index_r = -1, second_min_index_l = -1, second_min_index_r = -1;
		//找出第一小的距离
		for (int i = 0; i < x_index_l.size() - 1; ++i)
		{
			if ((x_index_l[i + 1] - x_index_l[i]) < first_min_distance)
			{
				first_min_index_l = i;
				first_min_index_r = i + 1;
				first_min_distance = x_index_l[i + 1] - x_index_l[i];
			}
		}

		//找出第二小的距离
		for (int i = 0; i < x_index_l.size() - 1; ++i)
		{
			if (i != first_min_index_l && (x_index_l[i + 1] - x_index_l[i]) < second_min_distance)
			{
				second_min_index_l = i;
				second_min_index_r = i + 1;
				second_min_distance = x_index_l[i + 1] - x_index_l[i];
			}
		}

		//第一根重合了
		if (0 != first_min_index_l && 0 != first_min_index_r && 0 != second_min_index_l && 0 != second_min_index_r)
		{
			for (int i = 0; i < points_source.rows; ++i)
			{
				int line_count = 0;
				int left_line_count = 0;
				for (int j = 0; j < points_source.cols && line_count < 6; ++j)
				{
					if (is_left_side)
					{
						if (line_count == 0)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 1, 2));
							
						}
						if (line_count == 1 || line_count == 2 || line_count == 3 || line_count == 4)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, line_count + 2, line_count + 2));
							
						}
						if (line_count == 5)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7, 7));
							
						}
						line_count++;
					}
					else
					{
						if (left_line_count++ < left_size)
							continue;

						if (line_count == 0)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+1, 7+ 2));
							
						}
						if (line_count == 1 || line_count == 2 || line_count == 3 || line_count == 4)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+line_count + 2, 7+line_count + 2));
							
						}
						if (line_count == 5)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+ 7, 7+7));
							
						}
						line_count++;
					}
				}
			}
		}

		//第三根重合了
		if (0 != first_min_index_l && 0 != first_min_index_r && 0 != second_min_index_l && 0 != second_min_index_r)
		{
			for (int i = 0; i < points_source.rows; ++i)
			{
				int line_count = 0;
				int left_line_count = 0;
				for (int j = 0; j < points_source.cols && line_count < 6; ++j)
				{
					if (is_left_side)
					{
						if (line_count == 0 || line_count == 1)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, line_count + 1, line_count + 1));
							
						}
						if (line_count == 2)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 3, 4));
							
						}
						if (line_count == 3 || line_count == 4)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, line_count + 2, line_count + 2));
							
						}
						if (line_count == 5)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7, 7));
							
						}
						line_count++;
					}
					else
					{
						if (left_line_count++ < left_size)
							continue;

						if (line_count == 0 || line_count == 1)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+ line_count + 1, 7+ line_count + 1));
							
						}
						if (line_count == 2)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+ 3, 7+ 4));
							
						}
						if (line_count == 3 || line_count == 4)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+line_count + 2, 7+line_count + 2));
							
						}
						if (line_count == 5)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+7, 7+7));
							
						}
						line_count++;
					}
				}
			}
		}

		//第五根重合了
		if (0 != first_min_index_l && 0 != first_min_index_r && 0 != second_min_index_l && 0 != second_min_index_r)
		{
			for (int i = 0; i < points_source.rows; ++i)
			{
				int line_count = 0;
				int left_line_count = 0;
				for (int j = 0; j < points_source.cols && line_count < 6; ++j)
				{
					if (is_left_side)
					{
						if (line_count == 0 || line_count == 1 || line_count == 2 || line_count == 3)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, line_count + 1, line_count + 1));
							
						}
						if (line_count == 4)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 5, 6));
							
						}

						if (line_count == 5)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7, 7));
							
						}
						line_count++;
					}

					else
					{
						if (left_line_count++ < left_size)
							continue;

						if (line_count == 0 || line_count == 1 || line_count == 2 || line_count == 3)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+line_count + 1, 7+line_count + 1));
							
						}
						if (line_count == 4)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+5, 7+6));
							
						}

						if (line_count == 5)
						{
							source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, 7+7, 7+7));
							
						}
						line_count++;
					}
				}
			}
		}
	}

	else
	{
		std::cout << "输入数据不符合预设" << endl;
		std::cout << "x_index_l.size()" << x_index_l.size() << endl;
	}
}

void CreateLineHash(Mat points_source, vector<LinePoint> & source_points)
{
	std::cout << "INIT LINE POINT" << endl;
	vector<float> x_index, x_index_l, x_index_r;
	for (int j = 0; j < points_source.cols; ++j)
	{
		if (*(points_source.data + init_point_row * points_source.step[0] + j * points_source.step[1]) != 0)
		{
			x_index.push_back(j);
		}
	}
	for (int i = 0; i < x_index.size(); ++i)
	{
		float med_index = (x_index[x_index.size() - 1] + x_index[0]) / 2;
		if (x_index[i] <= med_index)
		{
			x_index_l.push_back(x_index[i]);
			
		}
		else
		{
			x_index_r.push_back(x_index[i]);
		}
	}
	vector<LinePoint> point_line_left, point_line_right;	
	bool is_left_side = true;
	FindMatchHash(x_index_l, points_source, point_line_left, x_index_l.size(), is_left_side);


	is_left_side = false;
	FindMatchHash(x_index_r, points_source, point_line_right, x_index_l.size(), is_left_side);


	//合并结果
	int local_end_index_l = 0, local_end_index_r = 0;
	for (int i = init_point_row; i < points_source.rows; ++i)
	{
		for (int j = local_end_index_l; j < point_line_left.size(); ++j)
		{
			if (point_line_left[j].point.y == i)
			{
				source_points.emplace_back(point_line_left[j]);
				local_end_index_l++;
			}

			else if (point_line_left[j].point.y > i)
				break;
		}

		for (int j = local_end_index_r; j < point_line_right.size(); ++j)
		{
			if (point_line_right[j].point.y == i)
			{
				source_points.emplace_back(point_line_right[j]);
				local_end_index_r++;
			}

			else if (point_line_right[j].point.y > i)
				break;
		}
	}
}

void CreateLineHashDelectly(Mat points_source, vector<LinePoint> & source_points)
{
	/*for (int i = 0; i < points_source.rows *TRIM_DISTANCE; ++i)
	{
		int line_count = 0;		
		for (int j = 0; j < points_source.cols ; ++j)
		{
			if (*(points_source.data + i * points_source.step[0] + j * points_source.step[1]) != 0)
			{
				source_points.emplace_back(LinePoint(Point3f(j, i, 1), i*points_source.cols + j, line_count + 1, line_count + 1));
				line_count++;
			}
		}		
	}*/

	//按照直线顺序保存
	int line_count = 0;
	for (int i = 0; i < points_source.rows *TRIM_DISTANCE; ++i)
	{
		int max_line_count = 0;
		for (int j = 0; j < points_source.cols; ++j)
		{
			if (*(points_source.data + i * points_source.step[0] + j * points_source.step[1]) != 0 )
			{
				max_line_count++;				
			}
		}
		if(line_count< max_line_count)
			line_count = max_line_count;	
	}

	for (int i = 0; i < line_count; ++i)
	{
		for (int j = 0; j < points_source.rows *TRIM_DISTANCE; ++j)
		{
			int current_count = 0;
			for (int k = 0; k < points_source.cols; ++k)
			{
				if (*(points_source.data + j * points_source.step[0] + k * points_source.step[1]) != 0 && current_count++ == i)
				{
					source_points.emplace_back(LinePoint(Point3f(k, j, 1), j*points_source.cols + k, i + 1, i + 1));
				}
			}
		}
	}
}