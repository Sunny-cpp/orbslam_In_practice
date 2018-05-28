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

	//----��ʼ���еĺ���
	void MonocularInitialization();
	void CreateInitialMapMonocular();

	// һЩtrack�����еĺ���
	bool TrackReferenceKeyFrame();
	bool TrackWithMotionModel();
	bool Relocalization() {}
	bool TrackLocalMap();


	// local map step�еļ�������
	void UpdateLocalMap();
	void UpdateLocalKeyFrames();
	void UpdateLocalPoints();
	void SearchLocalPoints();

	Eigen::Matrix3d mmcamK;
	Eigen::Matrix<double,1, 5> mvdisCoffes;  // ���������һ�����������

	bool mbRGB;
	double mdFps;

	//-----�йص�������ȡ��
	ORBextractor* mpIniOrbextractor;
	ORBextractor* mpOrbextractorleft;

	//-------����ʱ��״̬��ʶ
	TrackingSTATEType msCurstate, msProcessedstate;
	Frame mCurframe;

	//---��һ֡�йصĶ���
	KeyFrame* mpLastKeyFrame;
	Frame mLastFrame;
	unsigned int mnLastKeyFrameId;
	unsigned int mnLastRelocFrameId;


	//---��ʼ��ʱ��Ҫ�õ���
	Initializer* mpInitializer;
	std::vector<int> mvIniMatches;  // ��orbmatcher����ƥ����
	Frame minitialRefFrame;
	std::vector<cv::Point3f> mvInitialPts3d;

	//---��ͼ��ȫ��Ψһ��������֯����keyframe �� mappoints
	Map* mpUniqueMap;

	//  -----tracking�е��ٶ�ģ��
	Eigen::Matrix4d mVelocity; // ��ʾ��֮֡��Tfc֮���pose��f��c�ϣ�fΪǰ֡��cΪ��ǰ֡

	 //Local Map �йص�һЩstuff
	 KeyFrame* mpReferenceKF; // ��ǰ�ؼ��ο�֡
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
