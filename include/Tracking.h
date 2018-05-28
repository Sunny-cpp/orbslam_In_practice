#ifndef TRACKING_H
#define TRACKING_H

#include <string>

#include <opencv2/core/core.hpp>

#include <Eigen/Core>

#include "ORBextractor.h"
#include "Frame.h"
#include "Initializer.h"
#include "KeyFrame.h"
#include "Map.h"

namespace ORBSlam
{
class Tracking
{
public:
	enum TrackingSTATEType
	{
		STATE_NotReady = -1,
		STATE_NotImage = 0,
		STATE_NotInital = 1,
		STATE_TrackingOK = 2,
		STATE_Lost = 3,
	};

	Tracking(const std::string& configfileName);
	void track_mono(const cv::Mat& img_, const double timeStamp_);

	void Reset() {}

private:
	void track();

	//----初始化中的函数
	void MonocularInitialization();
	void CreateInitialMapMonocular();

	// 一些track过程中的函数
	bool TrackReferenceKeyFrame();
	bool TrackWithMotionModel();
	bool Relocalization() {}
	bool TrackLocalMap();


	// local map step中的几个函数
	void UpdateLocalMap();
	void UpdateLocalKeyFrames();
	void UpdateLocalPoints();
	void SearchLocalPoints();

	Eigen::Matrix3d mmcamK;
	Eigen::Matrix<double,1, 5> mvdisCoffes;  // 本代码畸变一律用五个参数

	bool mbRGB;
	double mdFps;

	//-----有关的特征提取器
	ORBextractor* mpIniOrbextractor;
	ORBextractor* mpOrbextractorleft;

	//-------跟踪时的状态标识
	TrackingSTATEType msCurstate, msProcessedstate;
	Frame mCurframe;

	//---上一帧有关的东西
	KeyFrame* mpLastKeyFrame;
	Frame mLastFrame;
	unsigned int mnLastKeyFrameId;
	unsigned int mnLastRelocFrameId;


	//---初始化时需要用到的
	Initializer* mpInitializer;
	std::vector<int> mvIniMatches;  // 用orbmatcher给出匹配结果
	Frame minitialRefFrame;
	std::vector<cv::Point3f> mvInitialPts3d;

	//---地图，全局唯一！用于组织管理keyframe 和 mappoints
	Map* mpUniqueMap;

	//  -----tracking中的速度模型
	Eigen::Matrix4d mVelocity; // 表示两帧之间Tfc之间的pose，f下c上，f为前帧，c为当前帧

	 //Local Map 有关的一些stuff
	 KeyFrame* mpReferenceKF; // 当前关键参考帧
	 std::vector<KeyFrame*> mvpLocalKeyFrames;
	 std::vector<MapPoint*> mvpLocalMapPoints;

	 //---------last frame, last keyframe, Relocalisation Info
	 KeyFrame* mpLastKeyFrame;
	 Frame mLastFrame;
	 unsigned int mnLastKeyFrameId;
	 unsigned int mnLastRelocFrameId;
};
}
#endif
