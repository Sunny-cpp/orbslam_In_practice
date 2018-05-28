#include "system.h"
#include <iostream>

using namespace std;

namespace ORBSlam
{
	SlamSystem::SlamSystem(const std::string& configName, const std::string& bowFileName)
	{
		//step1:读入bowfile

		//step2:检查config文件是否存在
		cv::FileStorage config_file(configName, cv::FileStorage::READ);
		if (! config_file.isOpened())
		{
			cout << "not open config in the path: " << configName << endl;
			return;
		}

		mp_tracker = new Tracking(configName);
	}

	void SlamSystem::Trackmonocular(const cv::Mat& img, const double timeStamp)
	{
		//---step1:对tracker进行reset

		//---step2:进行跟踪
		mp_tracker->track_mono(img, timeStamp);
	}
}