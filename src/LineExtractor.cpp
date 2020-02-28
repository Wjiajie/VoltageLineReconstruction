#include "LineExtractor.h"

#define EDGE_THREASHOLD 20
#define MAXDSTOFLINE 0.05

bool LineExtractor::CalEdge()
{
	cv::Mat binaryImg;
	//cv::threshold(this->gray_img, binaryImg, 110, 255, CV_THRESH_BINARY_INV);
	binaryImg = this->gray_img;

	binary_Edge = Mat(image.rows, image.cols, CV_8UC1, Scalar(0));
	//没有在图像边缘的像素的梯度（没有padding）	
	//初始 粗筛选
	for (uint i = 0; i < binary_Edge.rows; i++)
	{
		for (uint j = 1; j < binary_Edge.cols; j++)
		{
			double g = CalHorizontalGradient((double)binaryImg.ptr<uchar>(i)[j - 1], (double)binaryImg.ptr<uchar>(i)[j]);
			binary_Edge.ptr<uchar>(i)[j] = g > EDGE_THREASHOLD ? g : 0;
			/*if (g > EDGE_THREASHOLD)
			{
				if (!flag)
				{
					previousEdgeColIdx = j;
				}
				else
				{

				}

			}*/

		}
	}

	//nms
	for (uint i = 0; i < binary_Edge.rows; i++)
	{
		for (uint j = 2; j < binary_Edge.cols - 2; j++)
		{
			if (binary_Edge.ptr<uchar>(i)[j] == 0)
				continue;
			if (binary_Edge.ptr<uchar>(i)[j] < binary_Edge.ptr<uchar>(i)[j - 1]
				|| binary_Edge.ptr<uchar>(i)[j] < binary_Edge.ptr<uchar>(i)[j + 1])
			{
				binary_Edge.ptr<uchar>(i)[j] = 0;
			}
			else
			{
				binary_Edge.ptr<uchar>(i)[j] = 255;
			}
		}
	}
	imwrite("E:\\ProjectsDuringPostgraduate\\HighVoltageLine\\images\\001_m_lines_binary_nms.jpg", this->binary_Edge);


	//再将内外边合并成一条边
	for (uint i = 0; i < binary_Edge.rows; i++)
	{
		bool flag = false;
		uint previousEdgeColIdx = 0;
		for (uint j = 1; j < binary_Edge.cols; j++)
		{
			if (binary_Edge.ptr<uchar>(i)[j] > 100)
			{
				binary_Edge.ptr<uchar>(i)[j] = 0;
				//入边
				if (!flag)
				{
					previousEdgeColIdx = j;
					flag = true;
				}
				else //出边
				{
					//防止刚开始就将出边误认为入边（最开始那个出边就不要了）
					if (j - previousEdgeColIdx > 20)
					{
						previousEdgeColIdx = j;
						flag = true;
						continue;
					}
					uint edgeIdx = std::round((j + previousEdgeColIdx) * 0.5f);
					binary_Edge.ptr<uchar>(i)[edgeIdx] = 255;
					flag = false;
				}
			}
		}
	}



	return true;
}

void LineExtractor::InitFindLine()
{
	uint cnt = 0;
	for (uint j = 1; j < this->binary_Edge.cols; j++)
	{
		if (this->binary_Edge.ptr<uchar>(1)[j] > 200) //不会准确=0
		{
			PointsInLine pl(cnt);
			pl.AddPoint(Eigen::Vector2i(j, 1));
			this->vec_lines.emplace_back(pl);
			cnt++;
		}
	}
	COUTENDL("number lines to process: " << cnt);

}

void LineExtractor::FindLines()
{
	bool allHasInit = false;
	uint rowInterval = 2;
	std::vector< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> > vec_A;  //方程系数矩阵
	vec_A.resize(vec_lines.size());
	uint fileFineFirstColCnt = 0;  //计数器
	uint preRowIdx = 0;

	//按行遍历
	/*
	for (uint i = 6; i < this->binary_Edge.rows - 3; i += rowInterval)
	{
		//每条直线的下一个迭代col
		bool breakFlag = true;  //如果没有修改这个就说明都Continue了

		for (uint k = 0; k < this->vec_lines.size(); k++)
		{
			if (vec_lines[k].NumberofNOTConsecutiveUpdates > 5)
			{
				//这个直线被视作不更新直线
				continue;
			}
			int procceedCol = findBeginColIndex(i, preRowIdx+1, k);
			if (procceedCol == -1)
			{
				vec_lines[k].NumberofNOTConsecutiveUpdates++;
				breakFlag = false;
				continue;
			}
			else
			{
				vec_lines[k].NumberofNOTConsecutiveUpdates = 0;;
			}

			//找到这个点是某条直线的下一个点
			if (vec_lines[k].vec_points.size() > 6)
			{
				//是否满足曲线方程
				Eigen::Vector3d u(procceedCol, i, 1);
				Eigen::Vector3d v1 = vec_lines[k].lineParams * u;
				double dst = u.dot(v1);
				if (dst < MAXDSTOFLINE)
				{
					this->vec_lines[k].vec_points.push_back(Eigen::Vector2i(procceedCol, i));
				}
			}
			else
			{
				//初值
				this->vec_lines[k].vec_points.emplace_back(Eigen::Vector2i(procceedCol, i));
			}

			//六个点更新一次参数即可
			if (vec_lines[k].vec_points.size() == 6)
			{
				updateParameter(vec_A[k], k);
			}
			breakFlag = false;
		}

		if (breakFlag)
		{
			COUTENDL("break flag, all lines break!");
			break;
		}

		if (i > 50 && i % 50 == 0)
		{
			rowInterval = 2;

			for (uint k = 0; k < this->vec_lines.size(); k++)
			{
				updateParameter(vec_A[k], k);
			}

		}
		preRowIdx = i;
	}
	*/

	//按线做吧
	for (uint k = 0; k < this->vec_lines.size(); k++)
	{
		//break到下一个线条件： 多次没有找到合适的下一个点或者 行找完了
		uint lastRow = 0;
		if (!findInitPoints(3, k, lastRow))
		{
			COUTENDL("drop line: " << k);
			continue;
		}

		updateParameter(vec_A[k], k);

		//查找后续的点
		uint failureTime = 0;
		for (uint i = lastRow + rowInterval; i < this->height; i += rowInterval)
		{
			Eigen::Vector2i p = *(this->vec_lines[k].vec_points.end() - 1);
			Eigen::Vector2i q = *(this->vec_lines[k].vec_points.end() - 1);
			Eigen::Vector2d dir = Eigen::Vector2d((double)(p - q).x(), (double)(p - q).y());
			dir.normalize();
			uint col_center = p.x();
			//左右共5个区间计算某个点是否符合曲线方程
			double minDst = 99999;
			int colSelect = -1;
			for (int m = -2; m < 3; m++)
			{
				int c = col_center + m;
				if (c < this->width && c > 0)
				{
					if (this->binary_Edge.ptr<uchar>(i)[c] > 200)
					{
						Eigen::Vector3d x(c / (float)(this->width), i / (float)(this->height), 1);
						Eigen::Vector3d tmp = this->vec_lines[k].lineParams * x;
						double dst = x.dot(tmp);
						if (dst < MAXDSTOFLINE && dst < minDst)
						{
							minDst = dst;
							colSelect = c;
						}
					}
				}
			}
			if (colSelect < 0)
			{
				failureTime++;
				if (failureTime > 5)
				{
					break;
				}
			}

			else
			{
				Eigen::Vector2i x(colSelect, i);
				Eigen::Vector2d dir_current = Eigen::Vector2d((x - p).x(), (x - p).y());
				dir_current.normalize();
				//if (dir_current.dot(dir) < 0.85)
				//{
				//	//夹角太大
				//	failureTime++;
				//	rowInterval = 1;
				//	continue;
				//}
				failureTime = 0;
				this->vec_lines[k].vec_points.push_back(Eigen::Vector2i(colSelect, i));
				rowInterval = 3;
			}

			if (this->vec_lines[k].vec_points.size() % 20 == 0)
			{
				updateParameter(vec_A[k], k);
			}

		}
		COUTENDL(k << "th line finish find points");
	}

	COUTENDL("finish first find line, delete invalid line");
	/*for (std::vector<PointsInLine>::iterator iter = this->vec_lines.begin(); iter != this->vec_lines.end(); iter++)
	{
		if (iter->vec_points.size() < 0.2 * height)
		{
			iter = this->vec_lines.erase(iter);
		}
	}*/
	for (uint k = 0; k < this->vec_lines.size(); k++)
	{
		this->vec_lines[k].lineIdx = k;
		for (uint j = 0; j < this->vec_lines[k].vec_points.size(); j++)
		{
			Eigen::Vector2i p = this->vec_lines[k].vec_points[j];
			image.ptr<uchar>(p.y())[3 * p.x()] = vec_lines[k].color.z();
			image.ptr<uchar>(p.y())[3 * p.x() + 1] = vec_lines[k].color.x();
			image.ptr<uchar>(p.y())[3 * p.x() + 2] = vec_lines[k].color.y();

		}

	}
	COUTENDL("VALID lines number: " << this->vec_lines.size());
}


int LineExtractor::findBeginColIndex(uint aim_row, int startRow, uint lineIdx)
{
	Eigen::Vector2i q = vec_lines[lineIdx].vec_points[vec_lines[lineIdx].vec_points.size() - 1];
	int leftColIdx = q.x();
	//从这个行列出发，沿着线 找到目标行的位置
	for (uint i = startRow; i < aim_row; i++)
	{
		if (this->binary_Edge.ptr<uchar>(i)[leftColIdx] > 250)
		{

		}
		else if (this->binary_Edge.ptr<uchar>(i)[leftColIdx - 1] > 250)
		{
			leftColIdx--;
		}
		else if (this->binary_Edge.ptr<uchar>(i)[leftColIdx + 1] > 250)
		{
			leftColIdx++;
		}
		else
		{

			return -1;
		}

	}

	return leftColIdx;
}

bool LineExtractor::updateParameter(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& A, uint lineIdx)
{
	uint pre_pointsNum = A.rows();

	if (pre_pointsNum > 0)
	{
		Eigen::Matrix<float, Eigen::Dynamic, 6> m = A;
		A.resize(vec_lines[lineIdx].vec_points.size(), 6);
		A.block(0, 0, pre_pointsNum, 6) = m;
	}
	else
	{
		A.resize(vec_lines[lineIdx].vec_points.size(), 6);
	}

	for (uint n = pre_pointsNum; n < vec_lines[lineIdx].vec_points.size(); n++)
	{
		//归一化
		float x = (vec_lines[lineIdx].vec_points[n].x()) / (float)this->width;
		float y = (vec_lines[lineIdx].vec_points[n].y()) / (float)this->height;
		A.block(n, 0, 1, 6) << x * x, 2 * x* y, 2 * x, y* y, 2 * y, 1;
	}
	//更新
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullV);
	Eigen::VectorXf vn = svd.matrixV().block(0, 5, 6, 1);
	vn /= vn(5, 0);
	vec_lines[lineIdx].lineParams
		<< vn(0, 0), vn(1, 0), vn(2, 0),
		vn(1, 0), vn(3, 0), vn(4, 0),
		vn(2, 0), vn(4, 0), 1;

	return true;
}

bool LineExtractor::findInitPoints(int startRow, uint lineIdx, uint& lastRow)
{
	Eigen::Vector2i q = vec_lines[lineIdx].vec_points[vec_lines[lineIdx].vec_points.size() - 1];
	int rowInterval = 1;
	int maxFailIterationTime = 3;
	int failureIterationTime = 0;
	int leftColIdx = q.x();
	//从这个行列出发，沿着线 找到目标行的位置
	while (true)
	{
		q = vec_lines[lineIdx].vec_points[vec_lines[lineIdx].vec_points.size() - 1];
		//隔五行找一个点
		for (uint i = 0; i < rowInterval; i++)
		{
			if (startRow >= this->height)
			{
				return false;
			}
			//沿直线搜索
			if (this->binary_Edge.ptr<uchar>(startRow)[leftColIdx] > 200)
			{
				failureIterationTime = 0;
			}
			else if (this->binary_Edge.ptr<uchar>(startRow)[leftColIdx - 1] > 200)
			{
				leftColIdx--;
				failureIterationTime = 0;
			}
			else if (this->binary_Edge.ptr<uchar>(startRow)[leftColIdx + 1] > 200)
			{
				leftColIdx++;
				failureIterationTime = 0;
			}
			else
			{
				if (vec_lines[lineIdx].vec_points.size() > 1)
				{
					Eigen::Vector2i q2 = vec_lines[lineIdx].vec_points[vec_lines[lineIdx].vec_points.size() - 2];
					Eigen::Vector2f dir = Eigen::Vector2f((float)(q - q2).x(), (float)(q - q2).y());
					dir.normalize();
					//下个迭代点选择这个方向
					int deltax = round(dir.x());
					int deltay = round(dir.y());
					leftColIdx += deltax;
					startRow += deltay;
					failureIterationTime++;
					continue;
				}
				else
				{
					//第二个点都找不到 直接丢掉这个直线
					return false;
				}
				failureIterationTime++;
			}
			q = Eigen::Vector2i(leftColIdx, startRow);  //上一次确定的迭代点
			startRow++;
		}
		if (failureIterationTime >= maxFailIterationTime)
		{
			return false;
		}
		this->vec_lines[lineIdx].vec_points.emplace_back(Eigen::Vector2i(leftColIdx, startRow));
		if (this->vec_lines[lineIdx].vec_points.size() >= 10)
		{
			COUTENDL("finish init points");
			lastRow = startRow;
			return true;
		}
		rowInterval = 5;
	}

	return false;
}

bool LineExtractor::findLinesOnlyByBinaryImg()
{
	Mat binaryImg_copy;
	this->binary_Edge.copyTo(binaryImg_copy);
	if (binaryImg_copy.channels() == 3)
	{
		cvtColor(binaryImg_copy, binaryImg_copy, CV_BGR2GRAY);
	}

	vec_lines.clear();
	uint linePointCnt = 0;
	int lineIdx = 0;
	do
	{
		linePointCnt = 0;
		PointsInLine pointsInLine(lineIdx);
		for (uint i = 0; i < (binaryImg_copy.rows / 2); i++)
		{
			bool hasFindLinePoint = false;
			for (uint j = 0; j < binaryImg_copy.cols; j++)
			{
				if (binaryImg_copy.ptr<uchar>(i)[j] > 128)
				{
					Eigen::Vector2i p;
					p << j, i;
					pointsInLine.vec_points.emplace_back(p);
					linePointCnt++;
					binaryImg_copy.ptr<uchar>(i)[j] = 0;
					hasFindLinePoint = true;
					break;
				}
			}
			if (hasFindLinePoint)
				continue;
		}

		if (linePointCnt > 0)
		{
			vec_lines.emplace_back(pointsInLine);
			lineIdx++;
		}

	} while (linePointCnt > 0);

	return true;
}
