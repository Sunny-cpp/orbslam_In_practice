#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "Frame.h"
#include "MapPoint.h"
#include "Map.h"

namespace ORBSlam
{
	class Map;
	class Frame;

class KeyFrame
{
public:
	KeyFrame(const Frame& frame, Map* pmap);
	
	void ComputeBoW();  // 将keyframe中的mDescriptors 转化成BOW 特征

	void AddMapPoint(MapPoint* pMappts, int ptsId);

// 有关的pose参数，来自于普通frame；放在public好访问
	static long unsigned int mlNextid;
	long unsigned int mlnId;
	long unsigned int mnFrameId;
	
	int TrackedMapPoints(const int nObs) {}  // 看多少点被高质量地观测到，还未实现；2018-5-14

	bool IsBad() const
	{
		return mbBad;
	}

	Eigen::Matrix3d GetR() const
	{
		return mKfRwc;
	}

	void SetR(const Eigen::Matrix3d& rot)
	{
		mKfRwc = rot;
	}

	Eigen::Vector3d Gettrans() const
	{
		return mKftwc;
	}

	void Settrans(const Eigen::Vector3d& trans)
	{
		mKftwc = trans;
	}

	std::vector<MapPoint*> GetMapPoints() const
	{
		return mvpMappts;
	 }

	//首先计算此keyframe所有点的depth，然后取出中间值；用q取得控制取得的值
	double ComputeSceneMedianDepth(const int q);   

	// 跟普通的frame中一样，是将它其中的传过来的
	std::vector<cv::KeyPoint> mvUnKeypts;
	cv::Mat mcvDescriptors;

	// ---内参数
	static float fx, fy, cx, cy;

	//---Covisibility Graph相关操作
	void UpdateConnections();
	void AddConnection(KeyFrame *pKF, const int &weight);
	void UpdateBestCovisibles();

private:
	// w=世界坐标系，c=cur camera坐标系， 其中W下C上
	Eigen::Matrix3d mKfRwc;   
	Eigen::Vector3d mKftwc;

	std::vector<MapPoint*> mvpMappts;
	Map* mpmap;

	//----表示keyframe是否good，在后续步骤中进行实现
	bool mbBad;

	// Covisibility Graph
	std::map<KeyFrame*, int> mConnectedKeyFrameWeights;  //< 与该关键帧连接的, <关键帧,权重>
	std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames; //< 排序后的关键帧, 按权重从大到小
	std::vector<int> mvOrderedWeights;             //<  对应的排序权重(从大到小)
};

}
#endif
