#pragma once
#ifndef _HEAD_H_
#define _HEAD_H_

//OpencvDirTraverse.cpp : Defines the entry point for the console application.
#define _CRT_SECURE_NO_DEPRECATE

#include <cstdio>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include <Eigen/SVD>
#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/rotation.h>


#include "opencv\cv.h"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>



using namespace std;
using namespace cv;

#endif // !_HEAD_H_

