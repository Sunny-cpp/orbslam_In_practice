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
	void ComputeDistinctiveDescriptors();   // mappoint���Ա��ܶ�features�۲쵽����ȡ��ţ��Descriptors���Ǹ�feature
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

	 //---����public�����ñ���
	static long unsigned int mnextId;
	long unsigned int mlnId;             // mapPoint id ��ʶ��

    //---��track local mapʱ����ֹ�ظ���local mappoint����ӵ�ı��
	long unsigned int mnTrackReferenceForFrame;
private:
	Eigen::Vector3d mWorldPos;  // ��mappoint����������ϵ�µ�λ��
	Eigen::Vector3d mNormalVector;          //Mean viewing direction

	//---��keyframe�صı���
	int nObs;
	std::map<KeyFrame*, int> mObservations;    // ���mappoint ���ĸ��ؼ�֡���ĸ�id��features�۲⵽��

	//--�ο�֡��ȫ�ֵ�ͼ
	KeyFrame* mpRefKF;      // ���mappoint���ĸ�keyframe������
	Map* mpMap;

	//-----is this mappoint bad?
	bool mbBad;
};

}
#endif
