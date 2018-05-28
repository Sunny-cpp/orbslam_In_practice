#ifndef SYSTEM_H
#define SYSTEM_H

#include <string>

#include <opencv2/core/core.hpp>
#include <Eigen/Core>

#include "Tracking.h"
namespace Eigen {}

namespace ORBSlam
{
class SlamSystem
{
public:
	SlamSystem(const std::string& configName, const std::string& bowFileName);

	void Trackmonocular(const cv::Mat& img, const double timeStamp);

public:
	Tracking* mp_tracker;
};
}
#endif
