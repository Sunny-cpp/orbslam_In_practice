#include "KeyFrame.h"

namespace ORBSlam
{
	long unsigned int KeyFrame::mlNextid = 0;
	float KeyFrame::fx;
	float KeyFrame::fy;
	float KeyFrame::cx;
	float KeyFrame::cy;

	KeyFrame::KeyFrame(const Frame& frame, Map* pmap):
		mpmap(pmap), mvUnKeypts(frame.GetUnKeyPts()), mvpMappts(frame.mvpMappts),
		mnFrameId(frame.mnId),mbBad(false),
		mKfRwc(frame.GetR()),mKftwc(frame.GetT())
	{
		mlnId = mlNextid++;
	}

	void KeyFrame::ComputeBoW() {}

	void KeyFrame::AddMapPoint(MapPoint* pMappts, int ptsId)
	{
		mvpMappts[ptsId] = pMappts;
	}

	double KeyFrame::ComputeSceneMedianDepth(const int q)
	{
		int N = mvpMappts.size();

		std::vector<double> vDepth;
		vDepth.reserve(N);

		// ȡ�ô�֡��Rwc��twc����Ҫ������std::mutex
		Eigen::Matrix3d Rwc = mKfRwc;
		Eigen::Vector3d twc = mKftwc;

		// ��λ����������ϵ��mvpMappts��ת����ǰ��camera����ϵ��
		for (int i=0;i<mvpMappts.size();i++)
		{
			if (mvpMappts[i])
			{
				Eigen::Vector3d curWorldPos=mvpMappts[i]->Getpos();
				Eigen::Vector3d Rwcrow3th = Rwc.row(2);

				double zC = Rwcrow3th.transpose()*curWorldPos + twc(2);
				vDepth.push_back(zC);
			}
		}

		std::sort(vDepth.begin(), vDepth.end());
		return vDepth[(vDepth.size() - 1) / q];
	}

	void KeyFrame::UpdateConnections()
	{
		//----step1: �ҳ����й���keyframe���乲��Ȩ��
		std::map<KeyFrame*, int> sKFcounts;
		for (int i=0;i<mvpMappts.size();i++)
		{
			MapPoint* pcurPt= mvpMappts[i];
			if (!pcurPt)
				continue;

			if (! pcurPt->IsBad())
			{
				std::map<KeyFrame*,int> obs= pcurPt->GetObservations();
				for (auto ite=obs.begin(); ite!=obs.end();ite++)
				{
					if (ite->first->mlnId== mlnId)
						continue;
					sKFcounts[ite->first]++;
				}
			}
		}
		if (sKFcounts.empty())
			return;

		//----step2: ��thֵ�ҳ����е�keyframe
		KeyFrame* pKFmost;
		int nmostNums = -1;
		
		int th = 15;   // Ȩ��Ϊ15���������趨��

		//--���ڴ洢����֡���ֱ�Ϊ<Ȩ�أ�keyframe>
		std::vector<std::pair<int, KeyFrame*> > KFsPairs;
		KFsPairs.reserve(sKFcounts.size() + 5);
		for (auto ite=sKFcounts.begin(); ite!=sKFcounts.end(); ite++)
		{
			if (ite->second>nmostNums)
			{
				nmostNums = ite->second;
				pKFmost = ite->first;
			}

			if (ite->second > th)
				KFsPairs.push_back(std::make_pair(ite->second, ite->first));
			ite->first->AddConnection(ite->first,ite->second);
		}

		if (KFsPairs.empty())
		{
			KFsPairs.push_back(std::make_pair(nmostNums, pKFmost));
			pKFmost->AddConnection(pKFmost, nmostNums);
		}

		//----step3: ����һ�¿�
		std::sort(KFsPairs.begin(), KFsPairs.end());
		mConnectedKeyFrameWeights = sKFcounts;
		mvpOrderedConnectedKeyFrames.clear();
		mvOrderedWeights.clear();
		for(int i=0;i<KFsPairs.size();i++)
		{
			mvpOrderedConnectedKeyFrames.push_back(KFsPairs[i].second);
			mvOrderedWeights.push_back(KFsPairs[i].first);
		}
	}

	// ----���һ��keyframe���������Ӵ�֡Ȩ��
	void KeyFrame::AddConnection(KeyFrame* pKF, const int& weight)
	{
		if (!mConnectedKeyFrameWeights.count(pKF) || mConnectedKeyFrameWeights[pKF]!=weight)
		{
			mConnectedKeyFrameWeights[pKF] = weight;
		}
		else
			return;

		UpdateBestCovisibles();
	}

	//----����Ȩ�������򣬸ı�mvOrderedWeights��mvpOrderedConnectedKeyFrames
	void KeyFrame::UpdateBestCovisibles()
	{
		std::vector<std::pair<int, KeyFrame*> > vKFpairs;
		for (auto ite=mConnectedKeyFrameWeights.begin();ite!= mConnectedKeyFrameWeights.end();ite++)
		{
			vKFpairs.push_back(std::make_pair(ite->second, ite->first));
		}

		std::sort(vKFpairs.begin(), vKFpairs.end());
		mvpOrderedConnectedKeyFrames.clear();
		mvOrderedWeights.clear();
		for (int i=0;i<vKFpairs.size();i++)
		{
			mvpOrderedConnectedKeyFrames.push_back(vKFpairs[i].second);
			mvOrderedWeights.push_back(vKFpairs[i].first);
		}
	}
}