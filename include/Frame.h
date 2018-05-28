#ifndef FRAME_H
#define FRAME_H

#include "ORBextractor.h"
#include "MapPoint.h"

#include <Eigen/Core>

namespace ORBSlam
{
#define FRAME_GRID_ROWS 48
#define	FRAME_GRID_COLS 64

	class MapPoint;

class Frame
{
public:
	// ��������У�time��image�� orbextractor��camk��discoff
	Frame(const double timeImg, const cv::Mat& img, ORBextractor* porbextractor,
		const Eigen::Matrix3d& camK, const Eigen::Matrix<double, 1, 5>& discoffes);

	Frame();

	Frame(const Frame& thr);
	Frame& operator=(const Frame& thr);

	bool isInFrustum(MapPoint* pMp, const double viewingCosLimit);

	std::vector<cv::KeyPoint> GetUnKeyPts() const {
		return mvUnKeypts;
	}

	std::vector<cv::KeyPoint>& GetUnKeyPts()
	{
		return mvUnKeypts;
	}

	cv::Mat GetDescriptors() const
	{
		return mcvDescriptors.clone();
	}

	Eigen::Matrix3d GetR() const { return mRwc;}

	Eigen::Matrix4d GetPose() const{ return mTwc;}

	Eigen::Vector3d GetT() const{ return mtwc;}

	void SetPose(const Eigen::Matrix4d& pose);

	 static void GetCameraPara(cv::Mat& camk);

	std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const;

	void ComputeBOW();

	static long long int mllnextid;
	long unsigned int mnId;

	//----����ȡ�õ�������������Ӧ��mappoint����ָ���������
    //��ӦmvpMappts��mvUnKeypts����ʾ���map����outlier;true��outlier
	// ---mvbOutlier�أ���true or false��������Ӧ��mappoint[i]!=NULL �²�����Ч�� ��Ȼ��ʼʱ����false��
	std::vector<MapPoint*> mvpMappts;
	std::vector<bool> mvbOutlier;   
	
	// ��ǰframe��ָ��Ĳο��ؼ�֡
	KeyFrame* mpReferenceKF;

private:
	//---������ȡ������ȡ�������������Ӧ��������
	ORBextractor* mpOrbextractor;
	std::vector<cv::KeyPoint> mvKeypts,mvUnKeypts;
	cv::Mat mcvDescriptors;

	static float miMinX, miMaxX, miMinY, miMaxY;

	static float mfGridElementWidthInv;
	static float mfGridElementHeightInv;
	// ÿ�����ӷ����������������ͼ��ֳɸ��ӣ���֤��ȡ��������ȽϾ���
	// FRAME_GRID_ROWS 48  FRAME_GRID_COLS 64
	std::vector<int> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];  //ÿ�������б��浱ǰ����ȡ���ĵ��id��

	//---�ڲ�����һ���þ�̬�����洢
	static bool mbInitalframe;
	static cv::Mat mcamk, mvdisCoffes;

	void ExtractorOrbFeatures(const cv::Mat& curImg);

	void UndistortedKeyPoints();
	void FindimageBound(const cv::Mat& initalImg);

	void AssignFeaturesToGrid();
	bool GetGridId(const double& x, const double&y, int& index_x, int& index_y);

	//----�������������ϵ��λ�˲���
	Eigen::Matrix3d mRwc;   // w=��������ϵ��c=cur camera����ϵ�� ����W��C��
	Eigen::Vector3d mtwc;
	Eigen::Matrix4d mTwc;   // Pose=[R,t]

};
}
#endif