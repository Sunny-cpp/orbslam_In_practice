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
	// 输入的量有：time，image， orbextractor，camk，discoff
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

	//----其提取得到的特征点所对应的mappoint，用指针进行连接
    //对应mvpMappts和mvUnKeypts，表示这个map点是outlier;true是outlier
	// ---mvbOutlier呢，其true or false在其所对应的mappoint[i]!=NULL 下才算有效； 当然开始时都是false的
	std::vector<MapPoint*> mvpMappts;
	std::vector<bool> mvbOutlier;   
	
	// 当前frame所指向的参考关键帧
	KeyFrame* mpReferenceKF;

private:
	//---特征提取器、提取到的特征及其对应的描述子
	ORBextractor* mpOrbextractor;
	std::vector<cv::KeyPoint> mvKeypts,mvUnKeypts;
	cv::Mat mcvDescriptors;

	static float miMinX, miMaxX, miMinY, miMaxY;

	static float mfGridElementWidthInv;
	static float mfGridElementHeightInv;
	// 每个格子分配的特征点数，将图像分成格子，保证提取的特征点比较均匀
	// FRAME_GRID_ROWS 48  FRAME_GRID_COLS 64
	std::vector<int> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];  //每个格子中保存当前点提取到的点的id号

	//---内参数，一律用静态变量存储
	static bool mbInitalframe;
	static cv::Mat mcamk, mvdisCoffes;

	void ExtractorOrbFeatures(const cv::Mat& curImg);

	void UndistortedKeyPoints();
	void FindimageBound(const cv::Mat& initalImg);

	void AssignFeaturesToGrid();
	bool GetGridId(const double& x, const double&y, int& index_x, int& index_y);

	//----相对于世界坐标系的位姿参数
	Eigen::Matrix3d mRwc;   // w=世界坐标系，c=cur camera坐标系， 其中W下C上
	Eigen::Vector3d mtwc;
	Eigen::Matrix4d mTwc;   // Pose=[R,t]

};
}
#endif