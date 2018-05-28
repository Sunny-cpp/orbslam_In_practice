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
			//---��ȡ�µ�һ֡��Ȼ������Covisibility Graph��Ȼ����뵽map��
			ProcessNewKeyFrame();

			//---�Խ����mappoint�����ϸ�ɸѡ

		}
	}

	void LocalMapping::ProcessNewKeyFrame()
	{
		//--ȡ���Ƚ����֡
		mpcurKeyframe = mpLNewKeyframes.front();
		mpLNewKeyframes.pop_front();

		//---����bow
		mpcurKeyframe->ComputeBoW();

		// ����֡�и��ٵ���mappoint����keyframe�ϵİ�
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

		//---update Covisibility Graph �����ӹ�ϵ
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