#include "LocalMapping.h"

namespace ORBSlam
{
	LocalMapping::LocalMapping(Map* pMap):
		mpUniqueMap(pMap)
	{
		mpLNewKeyframes.clear();
	}

	void LocalMapping::Run()
	{
		if (CheckNewKeyFrames())
		{
			//---获取新的一帧，然后处理下Covisibility Graph，然后插入到map中
			ProcessNewKeyFrame();

			//---对进入的mappoint进行严格筛选

		}
	}

	void LocalMapping::ProcessNewKeyFrame()
	{
		//--取得先进入的帧
		mpcurKeyframe = mpLNewKeyframes.front();
		mpLNewKeyframes.pop_front();

		//---计算bow
		mpcurKeyframe->ComputeBoW();

		// 将此帧中跟踪到的mappoint进行keyframe上的绑定
		std::vector<MapPoint*> vcurPts=mpcurKeyframe->GetMapPoints();
		for (int i=0;i<vcurPts.size();i++)
		{
			MapPoint* pcurPt = vcurPts[i];
			if (pcurPt==NULL)
				continue;

			if (! pcurPt->IsBad())
			{
				if (! pcurPt->IsInKeyframe(mpcurKeyframe))
				{
					pcurPt->AddObservation(mpcurKeyframe, i);
					pcurPt->UpdateNormalAndDepth();
					pcurPt->ComputeDistinctiveDescriptors();
				}
			}
		}

		//---update Covisibility Graph 的连接关系
		mpcurKeyframe->UpdateConnections();
		mpUniqueMap->AddKeyFrame(mpcurKeyframe);
	}

	bool LocalMapping::CheckNewKeyFrames()
	{
		return !mpLNewKeyframes.empty();
	}

	void LocalMapping::InsertKeyFrame(KeyFrame* pKF)
	{
		mpLNewKeyframes.push_back(pKF);
	}
}