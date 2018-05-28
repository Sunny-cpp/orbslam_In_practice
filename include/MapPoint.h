#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "KeyFrame.h"
#include "Map.h"

#include <Eigen/Core>

#include <map>

namespace ORBSlam
{
	class KeyFrame;
	class Map;

class MapPoint
{
public:
	MapPoint(const Eigen::Vector3d& pos, KeyFrame* pkeyframe, Map* pmap);

	void AddObservation(KeyFrame* pf, int id);  
	void ComputeDistinctiveDescriptors();   // mappoint可以被很多features观察到，获取最牛逼Descriptors的那个feature
	void UpdateNormalAndDepth();

	bool IsBad() const	{return mbBad;}

	Eigen::Vector3d Getpos() const
	{
		return mWorldPos;
	}

	Eigen::Vector3d GetNormalVector() const { return mNormalVector;	}

	void SetPos(const Eigen::Vector3d& pt)
	{
		mWorldPos = pt;
	}

	int Obsnums() const { return nObs; }
	
	std::map<KeyFrame*, int> GetObservations() const
	{
		return mObservations;
	}

	std::map<KeyFrame*, int>& GetObservations()
	{
		return mObservations;
	}

	bool IsInKeyframe(KeyFrame* pKF)
	{
		return mObservations.count(pKF);
	}

	 //---放在public区，好遍历
	static long unsigned int mnextId;
	long unsigned int mlnId;             // mapPoint id 标识号

    //---在track local map时，防止重复往local mappoint中添加点的标记
	long unsigned int mnTrackReferenceForFrame;
private:
	Eigen::Vector3d mWorldPos;  // 此mappoint在世界坐标系下的位置
	Eigen::Vector3d mNormalVector;          //Mean viewing direction

	//---与keyframe关的变量
	int nObs;
	std::map<KeyFrame*, int> mObservations;    // 这个mappoint 被哪个关键帧的哪个id的features观测到？

	//--参考帧和全局地图
	KeyFrame* mpRefKF;      // 这个mappoint由哪个keyframe来构造
	Map* mpMap;

	//-----is this mappoint bad?
	bool mbBad;
};

}
#endif
