#include "LineExtract.h"
#include "ImageProcess.h"
#include "Reconstruction.h"

string src_image_path = "E:\\BaiduNetdiskDownload\\RecoverStruct\\InputData\\RawImage\\*.png"; //ԭʼͼƬ��·��
string IntrinsicsPath_l = "E:\\BaiduNetdiskDownload\\RecoverStruct\\InputData\\camera_para\\out_camera_data_190530L.yml"; //�������
string IntrinsicsPath_m = "E:\\BaiduNetdiskDownload\\RecoverStruct\\InputData\\camera_para\\out_camera_data_190530M.yml";
string save_path = "E:\\BaiduNetdiskDownload\\RecoverStruct\\OutputData"; //����ļ���
string pc_init_path = "E:\\BaiduNetdiskDownload\\RecoverStruct\\OutputData\\Init_dlt_cloud.ply"; //����ĵ���
string pc_final_path = "E:\\BaiduNetdiskDownload\\RecoverStruct\\OutputData\\ceres_cloud.ply";

#define search_distance  10
#define threshold 5   //Խ��Խ�ϸ�
#define init_point_row 10 //��ʼ��������

void FindCenterPointInFinalLine(Mat mask, Mat & mask_center)
{
	Mat se = getStructuringElement(MORPH_RECT, Size(2, 2));
	Mat se1 = getStructuringElement(MORPH_RECT, Size(22, 22));
	Mat se2 = getStructuringElement(MORPH_RECT, Size(10, 10));
	Mat dst = Mat::zeros(mask.size(), CV_8UC1);
	//erode(mask, dst, se);
	dilate(mask, dst, se1);
	//erode(dst, dst, se2);
	//imwrite("E:\\BaiduNetdiskDownload\\shuangmushuju\\dalianwafangdian\\test\\dst.png", dst);
	
	vector<int> vec_first, vec_last;

	for (int i = init_point_row; i < dst.rows; ++i)
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
				*(mask_center.data + i * mask_center.step[0] + index * mask_center.step[1]) = 255;

			}

		}

		vec_first.clear();
		vec_last.clear();
	}

}


int main()
{
	vector<Mat> src_image, undistort_image;

	ImageProcess image_process;
	image_process.ReadDataFromFile(src_image_path, src_image);
	
	vector<std::string> IntrinsicsPath;
	IntrinsicsPath.emplace_back(IntrinsicsPath_l);
	IntrinsicsPath.emplace_back(IntrinsicsPath_m);

	image_process.ImageDedistortion(src_image, undistort_image, IntrinsicsPath);
	
	
	vector<Mat> extract_line, extract_center_line;
	for (int i = 0; i < undistort_image.size(); ++i)
	{
		Mat gray_image,gray_image_sharpen;
		image_process.rgb2gray(undistort_image[i], gray_image);
		image_process.Sharpen(gray_image, gray_image_sharpen); //�ɵ�����ͼ��Աȶȵ͵�ʱ�������Ч�����Щ
		//�ҵ���ͷ
		vector<Point2i> init_point;
		LineExtract line;
		line.InitLineHead(gray_image, init_point);	
		//��ȡ��
		Mat final_line = Mat::zeros(gray_image.rows, gray_image.cols, CV_8UC1);
		line.ExtractLine(gray_image_sharpen, init_point, final_line);

		//��������
		Mat center_point = Mat::zeros(undistort_image[i].rows, undistort_image[i].cols, CV_8UC1);
		// ��ȡ�����ߵ�����
		FindCenterPointInFinalLine(final_line, center_point);
		
		extract_line.emplace_back(final_line);
		extract_center_line.emplace_back(center_point);
	}

		
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//����֪�������������ؽ�
	//F��R,t ��һЩ��ͳ��������ģ���֪2d��2d�㣬�ɶԼ�Լ���õ���2d��3d����pnp�õ�����3d��3d�㣬��icp�õ���
	Eigen::Matrix3d R0;
	R0 << 0.775071, -0.631781, -0.0108743,
		0.0352285, 0.0603884, -0.997553,
		0.630892, 0.772791, 0.069062;    //world_pnp huzhou

	Eigen::Vector3d c0;
	c0 << 943.42, 1019.13, 54.7952; //world pnp huzhou

	Eigen::Matrix3d K0;
	K0 << 3.7612906067774788e+003, 0., 2.0457995013785016e+003, 0.,
		3.7159736130516289e+003, 1.4796241830921265e+003, 0., 0., 1.;  //huzhou

	std::vector<double> distort0{ -7.5267404892772435e-002, 1.2710728604878160e-001,
	   -2.5774289715705485e-003, -3.6861402083794338e-005,
	   2.1573505064134832e-001 };  //huzhou

	View view0(R0, c0, K0, distort0);

	Eigen::Matrix3d R1;
	R1 << 0.785548, -0.618716, -0.010268,
		0.0190466, 0.0407612, -0.998987,
		0.618508, 0.784557, 0.0438043;  //world pnp huzhou

	Eigen::Vector3d c1;
	c1 << 947.823, 1016.18, 54.5612; //world pnp huzhou

	Eigen::Matrix3d K1;
	K1 << 3.7516261809333914e+003, 0., 2.0321668349416393e+003, 0.,
		3.7075460080696898e+003, 1.4757973862050546e+003, 0., 0., 1.; //HUZHOU

	std::vector<double> distort1{ -8.9706636166420994e-002, 2.2503707166823739e-001,
	   -2.7655590432857850e-003, 1.1201754344822839e-004,
	   -8.2725767124464909e-003 }; //huzhou

	View view1(R1, c1, K1, distort1);

	vector<View> view{ view0,view1 };
	
	//Ѱ��ֱ��ƥ���
	
	Mat line_l, line_m;
	Mat center_line_l, center_line_m;
	Mat image_l, image_m;

	line_l = extract_line[0].clone();
	line_m = extract_line[1].clone();
	

	image_l = undistort_image[0].clone();
	image_m = undistort_image[1].clone();

	center_line_l = extract_center_line[0].clone();
	center_line_m = extract_center_line[1].clone();

	
	//F��R,t ��һЩ��ͳ��������ģ���֪2d��2d�㣬�ɶԼ�Լ���õ���2d��3d����pnp�õ�����3d��3d�㣬��icp�õ���
	//Reconstruction::pose_estimation_2d2d()�������F,R,t
	//huzhou
	Mat F_l_2_m_mat = (Mat_<double>(3, 3) << 1.56085e-09, 1.90842e-07, -0.000659478,
		-2.36214e-07, 6.59792e-08, 0.0100108,
		0.00070144, -0.0101018, 1);

		
	Reconstruction rec;

	//Ѱ�ҵ���ƥ���ϵ	
	vector<LinePoint> source_points, target_points;
	vector<LinePoint> source_points_center, target_points_center;
	rec.LinePointInit(line_l, line_m, source_points, target_points);
	rec.LinePointInit(center_line_l, center_line_m, source_points_center, target_points_center);
	vector<LinePoint> feature_l, feature_m;
	vector<Point2f> match_l_m;
	//rec.FindMatchPoint(source_points_center, target_points_center, feature_l, feature_m, match_l_m, F_l_2_m_mat , true);
	rec.FindMatchPointByProjection(source_points, target_points, source_points_center, target_points_center, feature_l, feature_m, match_l_m, F_l_2_m_mat, view, true);
	vector<Point2f> feature_l_2d, feature_m_2d;

	for (int i = 0; i < match_l_m.size(); ++i)
	{
		feature_l_2d.push_back(Point2f(feature_l[match_l_m[i].x].point.x, feature_l[match_l_m[i].x].point.y));
		feature_m_2d.push_back(Point2f(feature_m[match_l_m[i].y].point.x, feature_m[match_l_m[i].y].point.y));
	}
	
	rec.DrawEpiLines(image_l, image_m, feature_l_2d, feature_m_2d, F_l_2_m_mat, save_path);
	
	//image_process.SaveFeatureAndMatchFile(feature_l, feature_m, match_l_m, save_path);

	/*vector<Point2d> feature_l_by_lineid, feature_m_by_lineid;
	vector<Point2d> match_l_m_by_lineid;
	int current_lineid = 0;
	for (int i = 0; i < match_l_m.size(); ++i)
	{
		if(feature_l[i].line_id1)
	}*/

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	//����ṩfeature���ݺ�match���ݣ�ֱ���������������ؽ�
	//��������򵥵�����������������������ļ����Ͷ�Ӧ��ƥ���ļ�����ͼ��ƥ�䲻����
	//vector<LinePoint>  feature_s, feature_t;
	//vector<Point2d>  match_s_t;
	//image_process.LoadFeatureAndMatchFile(feature_s, feature_t, match_s_t, save_path, " ");

	vector<vector<LinePoint>> feature_file;

	feature_file.emplace_back(feature_l);
	feature_file.emplace_back(feature_m);


	vector<Observation> obs;
	
	for (int j = 0; j < match_l_m.size(); ++j)
	{
		/*double feature_col_index = (i == 0 ? match_s_t[j].x : match_s_t[j].y);
		double match_col_index = (i == 0 ? match_s_t[j].y : match_s_t[j].x);*/

		Eigen::Vector2d point = Eigen::Vector2d(feature_file[0][match_l_m[j].x].point.x, feature_file[0][match_l_m[j].x].point.y);
		Eigen::Vector2d match_point = Eigen::Vector2d(feature_file[1][match_l_m[j].y].point.x, feature_file[1][match_l_m[j].y].point.y);
		int l_id1 = feature_file[0][match_l_m[j].x].line_id1;
		int l_id2 = feature_file[0][match_l_m[j].x].line_id2;
		obs.emplace_back(Observation(point, match_point, 0 , 1 , l_id1, l_id2));
	}
		
	vector<Structure> P; //������ά����
	rec.CalculateStructure_Init_DLT(view, obs, P); //������������͹۲�㣬�������
	rec.EraseInvalidStructure(P, obs);
	rec.SavePLYFile_PointCloud(pc_init_path, P);
	rec.CalculateStructure_Ceres(view, obs, P);
	rec.SavePLYFile_PointCloud(pc_final_path, P);	
	
	return 1;

}