#include "system.h"
#include <iostream>

using namespace std;

namespace ORBSlam
{
	SlamSystem::SlamSystem(const std::string& configName, const std::string& bowFileName)
	{
		//step1:����bowfile

		//step2:���config�ļ��Ƿ����
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
		//---step1:��tracker����reset

		//---step2:���и���
		mp_tracker->track_mono(img, timeStamp);
	}
}