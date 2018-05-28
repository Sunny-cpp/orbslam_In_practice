#include "Converter.h"

namespace ORBSlam
{
Eigen::Matrix3d MatcamKtoEigen(const cv::Mat& camK)
{
	Eigen::Matrix3d K;
	K.setZero();
	if (camK.cols != 3 && camK.rows != 3)
		return K;

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			K(i, j) = camK.at<float>(i, j);
	return K;
}

}