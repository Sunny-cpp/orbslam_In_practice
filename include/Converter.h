#ifndef CONVERTER_H
#define CONVERTER_H

#include <Eigen/Core>

#include <opencv2/core/core.hpp>

namespace ORBSlam
{
	Eigen::Matrix3d MatcamKtoEigen(const cv::Mat& camK);
}
#endif
