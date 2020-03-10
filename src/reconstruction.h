#pragma once

#include "Head.h"

#define IMG_WIDTH 4096
#define IMG_HEIGHT 3000

struct LinePoint
{
	Point3f point;
	int point_id; //点索引
	int line_id1; //属于哪条线
	int line_id2; //属于哪条线  有时候会出现两条线重合的情况，即一个点属于两条线
	bool is_matched = false; //是否已匹配

	LinePoint() {};
	LinePoint(Point3f p, int p_id  = -1 ,int l_id1 = -1 ,int l_id2 = -1 )
	{
		point = p;
		point_id = p_id;
		line_id1 = l_id1;
		line_id2 = l_id2;
	}

};

struct View
{
	//poses
	Eigen::Matrix3d rotation;
	Eigen::Vector3d center;
	Eigen::Matrix3d K;
	Eigen::Matrix3d K_Inv;
	std::vector<double> distortionParams;
	Eigen::Matrix3d t_;
	Eigen::Vector3d t;

	double* rotation_array; //轴角
	double* translatrion_array;
	double scale;
	View() {}
	View(Eigen::Matrix3d r, Eigen::Vector3d c, Eigen::Matrix3d _K, std::vector<double> _distort)
	{	
		this->rotation = r;
		this->center = c;
		this->K = _K;
		this->distortionParams = _distort;

		this->K_Inv = this->K.inverse();
		t = -r * c;
		t_ << 0, -t(2), t(1),
			t(2), 0, -t(0),
			-t(1), t(0), 0;

		this->rotation_array = new double[3];
		Eigen::AngleAxisd aa(this->rotation);
		Eigen::Vector3d v = aa.angle() * aa.axis();
		rotation_array[0] = v.x();
		rotation_array[1] = v.y();
		rotation_array[2] = v.z();

		this->translatrion_array = new double[3];
		translatrion_array[0] = t.x();
		translatrion_array[1] = t.y();
		translatrion_array[2] = t.z();
		this->scale = 1.0f;   //和world的尺度
	}
};

struct Observation
{
	Eigen::Vector2d pixel;
	Eigen::Vector2d pixel_norm;
	Eigen::Vector2d match_pixel;
	Eigen::Vector2d match_pixel_norm;
	int host_camera_id; //属于哪一个camera
	int neighbor_camera_id; //与之匹配的camera，用来寻找匹配点
	int line_id1; //属于哪一条线
	int line_id2;

	Observation() {}
	Observation(Eigen::Vector2d p, Eigen::Vector2d m_p,int h_camera_id_ = -1, int n_camera_id_ = -1, int l_id1_ = -1,int l_id2_ = -1)
	{
		pixel = p;
		pixel_norm = Eigen::Vector2d(p.x() / IMG_WIDTH, p.y() / IMG_HEIGHT);
		match_pixel = m_p;
		match_pixel_norm = Eigen::Vector2d(m_p.x() / IMG_WIDTH, m_p.y() / IMG_HEIGHT);
		host_camera_id = h_camera_id_;
		neighbor_camera_id = n_camera_id_;
		line_id1 = l_id1_;
		line_id2 = l_id2_;
	}
};

struct Structure
{
	Eigen::Vector3d position;
	Eigen::Vector3d colors; //rgb	
	uint structure_index;
	double* positions_array;
	bool isvalid_structure;
	
	Structure() {}
	Structure(Eigen::Vector3d _p , uint structure_index_,bool isvalid_structure_ = true)
	{
		position = _p;
		colors = Eigen::Vector3d::Zero();	
		positions_array = new double[3];
		positions_array[0] = _p.x();
		positions_array[1] = _p.y();
		positions_array[2] = _p.z();

		structure_index = structure_index_;

		isvalid_structure = isvalid_structure_;
	}
};

//BA
struct Ceres_Triangulate
{
	//2d 点
	const Eigen::Vector2d x;
	const Eigen::Matrix<double, 3, 4> P;  //投影矩阵

	Ceres_Triangulate(Eigen::Vector2d x_, Eigen::Matrix<double, 3, 4> P_) :x(x_), P(P_) {}

	template<typename T>
	bool operator()(const T* const ceres_X, T* residual) const
	{

		T PX0 = T(P(0, 0)) * ceres_X[0] + T(P(0, 1)) * ceres_X[1] + T(P(0, 2)) * ceres_X[2] + T(P(0, 3));
		T PX1 = T(P(1, 0)) * ceres_X[0] + T(P(1, 1)) * ceres_X[1] + T(P(1, 2)) * ceres_X[2] + T(P(1, 3));
		T PX2 = T(P(2, 0)) * ceres_X[0] + T(P(2, 1)) * ceres_X[1] + T(P(2, 2)) * ceres_X[2] + T(P(2, 3));

		PX0 = PX0 / PX2;
		PX1 = PX1 / PX2;

		residual[0] = T(x.x()) - PX0;
		residual[1] = T(x.y()) - PX1;

		return true;
	}

};

struct Ceres_Triangulate_AdjustRt
{
	const Eigen::Vector2d x;
	const Eigen::Matrix<double, 3, 3> K;  //投影矩阵


	Ceres_Triangulate_AdjustRt(Eigen::Vector2d x_, Eigen::Matrix<double, 3, 3> K_ ) :x(x_), K(K_){}

	template<typename T>
	bool operator()(const T* const ceres_X, const T* const ceres_angleAxis, const T* const ceres_t, T* residual) const
	{
		T PX[3];
		PX[0] = ceres_X[0];
		PX[1] = ceres_X[1];
		PX[2] = ceres_X[2];

		T PX_r[3];
		ceres::AngleAxisRotatePoint(ceres_angleAxis, PX, PX_r);

		T PX0 = T(K(0, 0)) * (PX_r[0] + ceres_t[0]) + T(K(0, 2)) * (PX_r[2] + ceres_t[2]);
		T PX1 = T(K(1, 1)) * (PX_r[1] + ceres_t[1]) + T(K(1, 2)) * (PX_r[2] + ceres_t[2]);
		T PX2 = (PX_r[2] + ceres_t[2]);

		PX0 = PX0 / PX2;
		PX1 = PX1 / PX2;


		residual[0] = T(x.x()) -  PX0;
		residual[1] = T(x.y()) -  PX1;
		return true;
	}

};


class Reconstruction
{
public:

	std::vector<View> view_;
	std::vector<Observation> obs_;
	std::vector<Structure> P_;
	
	Reconstruction() {};
	void pose_estimation_2d2d(vector<Point2f> points1, vector<Point2f> points2, vector<Point2f> match);
	void solve_pnp_2d_3d_ceres(string ctrp_path);
	void CalculateStructure_Init_DLT(std::vector<View> view, std::vector<Observation> obs , vector<Structure> & output_struct);
	void CalculateStructure_Ceres(std::vector<View> view, std::vector<Observation> obs, vector<Structure> & output_struct);
	void DrawEpiLines(const Mat& img_1, const Mat& img_2, vector<Point2f>points1, vector<Point2f>points2, Mat F, string save_path);
	void FindMatchPoint(vector<LinePoint> &source_points, vector<LinePoint> & target_points, vector<LinePoint> & match1, vector<LinePoint> & match2, vector<Point2f> & match_1_2 ,Mat F_s_2_t , bool is_create_match_file = false/*,Mat F_t_2_s*/);
	void FindMatchPointByProjection(vector<LinePoint> &source_points, vector<LinePoint> & target_points, vector<LinePoint> source_points_center, vector<LinePoint> target_points_center, vector<LinePoint> & feature_source, vector<LinePoint> & feature_target, vector<Point2f> & match_s_t, Mat F_s_2_t, vector<View>  view, bool is_create_match_file /*, Mat F_t_2_s*/);
	void LinePointInit(Mat point_source, Mat point_target, vector<LinePoint> & source_points, vector<LinePoint> & target_points);
	void SavePLYFile_PointCloud(std::string save_path, std::vector<Structure> structure);
	void EraseInvalidStructure(std::vector<Structure> & structure ,std::vector<Observation>& observation);
	~Reconstruction() {};
};

//返回在target_points中找到过epilines的对应点
void FindCorrespondPointInEpilines(LinePoint source_point,vector<LinePoint>target_points, vector<Point2f> & match_source, vector<Point2f> & match_target, Vec<float, 3> epiline);
void FindCorrespondPointInEpilines(LinePoint source_point, LinePoint Inspection_point, vector<LinePoint> & output_points, Vec<float, 3> epiline);

void FindMatchHash(vector<float> x_index_l,Mat points_source, vector<LinePoint> & source_points, int left_size ,bool is_left_side);
void CreateLineHash(Mat points_source, vector<LinePoint> & source_points);
void CreateLineHashDelectly(Mat points_source, vector<LinePoint> & source_points);
