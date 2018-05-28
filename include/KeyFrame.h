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
	
	void ComputeBoW();  // ��keyframe�е�mDescriptors ת����BOW ����

	void AddMapPoint(MapPoint* pMappts, int ptsId);

// �йص�pose��������������ͨframe������public�÷���
	static long unsigned int mlNextid;
	long unsigned int mlnId;
	long unsigned int mnFrameId;
	
	int TrackedMapPoints(const int nObs) {}  // �����ٵ㱻�������ع۲⵽����δʵ�֣�2018-5-14

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

	//���ȼ����keyframe���е��depth��Ȼ��ȡ���м�ֵ����qȡ�ÿ���ȡ�õ�ֵ
	double ComputeSceneMedianDepth(const int q);   

	// ����ͨ��frame��һ�����ǽ������еĴ�������
	std::vector<cv::KeyPoint> mvUnKeypts;
	cv::Mat mcvDescriptors;

	// ---�ڲ���
	static float fx, fy, cx, cy;

	//---Covisibility Graph��ز���
	void UpdateConnections();
	void AddConnection(KeyFrame *pKF, const int &weight);
	void UpdateBestCovisibles();

private:
	// w=��������ϵ��c=cur camera����ϵ�� ����W��C��
	Eigen::Matrix3d mKfRwc;   
	Eigen::Vector3d mKftwc;

	std::vector<MapPoint*> mvpMappts;
	Map* mpmap;

	//----��ʾkeyframe�Ƿ�good���ں��������н���ʵ��
	bool mbBad;

	// Covisibility Graph
	std::map<KeyFrame*, int> mConnectedKeyFrameWeights;  //< ��ùؼ�֡���ӵ�, <�ؼ�֡,Ȩ��>
	std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames; //< �����Ĺؼ�֡, ��Ȩ�شӴ�С
	std::vector<int> mvOrderedWeights;             //<  ��Ӧ������Ȩ��(�Ӵ�С)
};

}
#endif
