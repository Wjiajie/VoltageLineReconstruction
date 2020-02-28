#include "LineExtractor.h"

#define EDGE_THREASHOLD 20
#define MAXDSTOFLINE 0.05

bool LineExtractor::CalEdge()
{
	cv::Mat binaryImg;
	//cv::threshold(this->gray_img, binaryImg, 110, 255, CV_THRESH_BINARY_INV);
	binaryImg = this->gray_img;

	binary_Edge = Mat(image.rows, image.cols, CV_8UC1, Scalar(0));
	//û����ͼ���Ե�����ص��ݶȣ�û��padding��	
	//��ʼ ��ɸѡ
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


	//�ٽ�����ߺϲ���һ����
	for (uint i = 0; i < binary_Edge.rows; i++)
	{
		bool flag = false;
		uint previousEdgeColIdx = 0;
		for (uint j = 1; j < binary_Edge.cols; j++)
		{
			if (binary_Edge.ptr<uchar>(i)[j] > 100)
			{
				binary_Edge.ptr<uchar>(i)[j] = 0;
				//���
				if (!flag)
				{
					previousEdgeColIdx = j;
					flag = true;
				}
				else //����
				{
					//��ֹ�տ�ʼ�ͽ���������Ϊ��ߣ��ʼ�Ǹ����߾Ͳ�Ҫ�ˣ�
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
		if (this->binary_Edge.ptr<uchar>(1)[j] > 200) //����׼ȷ=0
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
	std::vector< Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> > vec_A;  //����ϵ������
	vec_A.resize(vec_lines.size());
	uint fileFineFirstColCnt = 0;  //������
	uint preRowIdx = 0;

	//���б���
	/*
	for (uint i = 6; i < this->binary_Edge.rows - 3; i += rowInterval)
	{
		//ÿ��ֱ�ߵ���һ������col
		bool breakFlag = true;  //���û���޸������˵����Continue��

		for (uint k = 0; k < this->vec_lines.size(); k++)
		{
			if (vec_lines[k].NumberofNOTConsecutiveUpdates > 5)
			{
				//���ֱ�߱�����������ֱ��
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

			//�ҵ��������ĳ��ֱ�ߵ���һ����
			if (vec_lines[k].vec_points.size() > 6)
			{
				//�Ƿ��������߷���
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
				//��ֵ
				this->vec_lines[k].vec_points.emplace_back(Eigen::Vector2i(procceedCol, i));
			}

			//���������һ�β�������
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

	//��������
	for (uint k = 0; k < this->vec_lines.size(); k++)
	{
		//break����һ���������� ���û���ҵ����ʵ���һ������� ��������
		uint lastRow = 0;
		if (!findInitPoints(3, k, lastRow))
		{
			COUTENDL("drop line: " << k);
			continue;
		}

		updateParameter(vec_A[k], k);

		//���Һ����ĵ�
		uint failureTime = 0;
		for (uint i = lastRow + rowInterval; i < this->height; i += rowInterval)
		{
			Eigen::Vector2i p = *(this->vec_lines[k].vec_points.end() - 1);
			Eigen::Vector2i q = *(this->vec_lines[k].vec_points.end() - 1);
			Eigen::Vector2d dir = Eigen::Vector2d((double)(p - q).x(), (double)(p - q).y());
			dir.normalize();
			uint col_center = p.x();
			//���ҹ�5���������ĳ�����Ƿ�������߷���
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
				//	//�н�̫��
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
	//��������г����������� �ҵ�Ŀ���е�λ��
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
		//��һ��
		float x = (vec_lines[lineIdx].vec_points[n].x()) / (float)this->width;
		float y = (vec_lines[lineIdx].vec_points[n].y()) / (float)this->height;
		A.block(n, 0, 1, 6) << x * x, 2 * x* y, 2 * x, y* y, 2 * y, 1;
	}
	//����
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
	//��������г����������� �ҵ�Ŀ���е�λ��
	while (true)
	{
		q = vec_lines[lineIdx].vec_points[vec_lines[lineIdx].vec_points.size() - 1];
		//��������һ����
		for (uint i = 0; i < rowInterval; i++)
		{
			if (startRow >= this->height)
			{
				return false;
			}
			//��ֱ������
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
					//�¸�������ѡ���������
					int deltax = round(dir.x());
					int deltay = round(dir.y());
					leftColIdx += deltax;
					startRow += deltay;
					failureIterationTime++;
					continue;
				}
				else
				{
					//�ڶ����㶼�Ҳ��� ֱ�Ӷ������ֱ��
					return false;
				}
				failureIterationTime++;
			}
			q = Eigen::Vector2i(leftColIdx, startRow);  //��һ��ȷ���ĵ�����
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
