#include "Tracking.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

namespace ORBSlam
{
	Tracking::Tracking(const std::string& configfileName):msCurstate(STATE_NotImage), 
		msProcessedstate(STATE_NotImage), mpInitializer(static_cast<Initializer*>(NULL))
	{
		cv::FileStorage figFile(configfileName, cv::FileStorage::READ);
		if (!figFile.isOpened())
		{
			std::cout<< "not open config at the path: " << configfileName <<std::endl;
			return;
		}
		//---����һЩ�йصĲ���
		double fx = figFile["Camera.fx"];
		double fy = figFile["Camera.fy"];
		double cx = figFile["Camera.cx"];
		double cy = figFile["Camera.cy"];
		mmcamK << fx, 0.0, cx,
			0.0, fy, cy,
			0.0, 0.0, 1.0;
		mvdisCoffes << figFile["Camera.k1"], figFile["Camera.k2"], figFile["Camera.p1"], figFile["Camera.p2"],
			figFile["Camera.k3"];
		
		mdFps = figFile["Camera.fps"];
		if (mdFps <= 0.0)
			mdFps = 10.0;

		int isRBG = figFile["Camera.RGB"];
		if (isRBG > 0)
			mbRGB = true;
		else
			mbRGB = false;
		
		int nFeatures = figFile["ORBextractor.nFeatures"];
		double scaleFactor=figFile["ORBextractor.scaleFactor"];
		int nLevels=figFile["ORBextractor.nLevels"];		
		int iniThFAST=figFile["ORBextractor.iniThFAST"];
		int minThFAST=figFile["ORBextractor.minThFAST"];

		//---��������extractor�ĳ�ʼ��
		mpOrbextractorleft = new ORBextractor(nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST);
		mpIniOrbextractor = new ORBextractor(2*nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST);

		//---��ͼ��
		mpUniqueMap = new Map();
	}

	void Tracking::track_mono(const cv::Mat& img_, const double timeStamp_)
	{
		cv::Mat img_gray=img_;
		if (img_gray.channels()==3)
		{
			if (mbRGB)
				cv::cvtColor(img_gray, img_gray, CV_RGB2GRAY);
			else
				cv::cvtColor(img_gray, img_gray, CV_BGR2GRAY);
		}
		else if(img_gray.channels()==4)
		{
			if (mbRGB)
				cv::cvtColor(img_gray, img_gray, CV_RGBA2GRAY);
			else
				cv::cvtColor(img_gray, img_gray, CV_BGRA2GRAY);
		}

		//---��ȡ����
		if (msCurstate== STATE_NotImage ||msCurstate== STATE_NotInital)
		{
			mCurframe = Frame(timeStamp_,img_gray,mpIniOrbextractor,mmcamK,mvdisCoffes);
		}
		else
		{
			mCurframe = Frame(timeStamp_, img_gray, mpOrbextractorleft, mmcamK, mvdisCoffes);
		}
		track();
	}

	void Tracking::track()
	{
		if (msCurstate == STATE_NotImage)
			msCurstate = STATE_NotInital;
		if (msCurstate==STATE_NotInital)
		{
			MonocularInitialization();

			if (msCurstate != STATE_TrackingOK)
				return;
		}
		else  //track: ok or lost
		{
			bool bOK = false;

		   //----���ַ���������֮֡��pose
		  if (msCurstate==STATE_TrackingOK)
		  {
			if (mVelocity.isZero())
			{
				bOK=TrackReferenceKeyFrame();
			}
			else
			{
				bOK = TrackWithMotionModel();
				if (!bOK)
					TrackReferenceKeyFrame();
			}
		  }
		  else
		  {
				bOK = Relocalization();
		  }

		  // ---��֮֡���pose�������ˣ�����local map�н���point tracking
		  if(bOK)
			  bOK = TrackLocalMap();

		  if (bOK)
			  msCurstate = STATE_TrackingOK;
		  else
			  msCurstate = STATE_Lost;

		  // ---do decision for insert keyframe to localmapping thread
		  if (bOK)
		  {
			  // --- 1. update mVelocityģ��
			  if (! mLastFrame.GetPose().isZero())
			  {
				  mVelocity = mCurframe.GetPose()*mLastFrame.GetPose().inverse();
			  }
			  else
			  {
				  mVelocity.setZero();
			  }

			  // ----�ж��Ƿ���һ��keyframe��Ȼ�����
		  }

		  if (msCurstate== STATE_Lost)
		  {
			  // Reset
		  }

		  if (!mCurframe.mpReferenceKF)
			  mCurframe, mpReferenceKF = mpReferenceKF;

		  mLastFrame = Frame(mCurframe);
		}

		//---��¼�µ�ǰ֡��λ����Ϣ
	}

	void Tracking::MonocularInitialization()
	{
		if (! mpInitializer)
		{
			if (mCurframe.GetUnKeyPts().size()>100)
			{
				mpInitializer = new Initializer(mCurframe, 1.0, 200);
				mvIniMatches.resize(mCurframe.GetUnKeyPts().size());
				std::fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

				minitialRefFrame =mCurframe;
				mLastFrame = Frame(mCurframe);
				return;
			}
		}

		if (mCurframe.GetUnKeyPts().size()<=100)
		{
			delete mpInitializer;
			mpInitializer = static_cast<Initializer*>(NULL);
			mvIniMatches.clear();
			return;
		}

		ORBmatcher matcher(0.9, true);
		std::vector<cv::Point2f> vbPrevMatched;
		vbPrevMatched.resize(minitialRefFrame.GetUnKeyPts().size());
		for (int i=0;i<minitialRefFrame.GetUnKeyPts().size();i++)
		{
			vbPrevMatched[i] = minitialRefFrame.GetUnKeyPts()[i].pt;
		}

		int matcherCounts = matcher.SearchForInitialization(minitialRefFrame, mCurframe, vbPrevMatched,mvIniMatches,100);
		if (matcherCounts<=100)
		{
			delete mpInitializer;
			mpInitializer = static_cast<Initializer*>(NULL);
			mvIniMatches.clear();
			return;
		}

		Eigen::Matrix3d Rot;
		Eigen::Vector3d Trans;
		std::vector<cv::Point3f> vP3d;
		std::vector<bool> vbTriangulated;
		if (mpInitializer->Initialize(mCurframe, mvIniMatches,Rot,Trans, mvInitialPts3d,vbTriangulated))
		{
			for (int i=0;i<mvIniMatches.size();i++)
			{
				if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
					mvIniMatches[i] = -1;
			}

			// ---�������õ���Rot �� Trans����R12��T12��1��2��
			Eigen::Matrix4d pose,curFpose;
			pose.setIdentity();
			minitialRefFrame.SetPose(pose);
			curFpose << Rot, Trans, Eigen::Vector3d::Zero().transpose(), 1;
			mCurframe.SetPose(curFpose);

			// �õ��˳�ʼ��֡�󣬾Ϳ�����initial point and pose��
			CreateInitialMapMonocular();
			return;
		}
		delete mpInitializer;
		mpInitializer = static_cast<Initializer*>(NULL);
		mvIniMatches.clear();
	}

	void Tracking::CreateInitialMapMonocular()
	{
		KeyFrame* pkeyfRef = new KeyFrame(minitialRefFrame, mpUniqueMap);
		KeyFrame* pkeyfinital = new KeyFrame(mCurframe, mpUniqueMap);
		pkeyfRef->ComputeBoW();
		pkeyfinital->ComputeBoW();

		mpUniqueMap->AddKeyFrame(pkeyfRef);
		mpUniqueMap->AddKeyFrame(pkeyfinital);

		for (int i=0;i<mvIniMatches.size();i++)
		{
			if(mvIniMatches[i]<0)
				continue;

			Eigen::Vector3d posAlias(mvInitialPts3d[i].x, mvInitialPts3d[i].y, mvInitialPts3d[i].z);
			MapPoint* worldpos = new MapPoint(posAlias,pkeyfinital,mpUniqueMap);

			pkeyfRef->AddMapPoint(worldpos,i);
			pkeyfinital->AddMapPoint(worldpos, mvIniMatches[i]);

			worldpos->AddObservation(pkeyfRef, i);
			worldpos->AddObservation(pkeyfinital, mvIniMatches[i]);

			mpUniqueMap->AddMapPoint(worldpos);
		}
		
		//ȫ���Ż�
		Optimizer::GlobalBundleAdjustemnt(mpUniqueMap, 20);
		
		double invDepth = 1.0/pkeyfRef->ComputeSceneMedianDepth(2);
		if (invDepth<0 || pkeyfinital->TrackedMapPoints(1)<100)
		{
			std::cout << "not initialization�� Reset��\n";
			Reset();
			return;
		}

		//�߶ȹ�һ��,trans!
		Eigen::Vector3d kfTrans=pkeyfinital->Gettrans();
		kfTrans *= invDepth;
		pkeyfinital->Settrans(kfTrans);

		//----point ��һ��
		std::vector<MapPoint*>& vpAllMapPoints = pkeyfRef->GetMapPoints();
		for (int i=0;i<vpAllMapPoints.size();i++)
		{
			if (vpAllMapPoints[i])
			{
				Eigen::Vector3d pos = vpAllMapPoints[i]->Getpos();
				vpAllMapPoints[i]->SetPos(pos*invDepth);
			}
		}

		// ����һЩ��ʼ
		mpReferenceKF = pkeyfinital;
		mCurframe.mpReferenceKF = pkeyfinital;
		msCurstate = STATE_TrackingOK;
	}

	//  ����10�����ok��   --------good,ok(2018-5-16)
	bool Tracking::TrackReferenceKeyFrame()
	{
		//---step1:����֡ת��BOW����
		mCurframe.ComputeBOW();

		//---step2: ��BOW����ƥ��
		ORBmatcher matcher(0.7, true);
		std::vector<MapPoint*> vpMapPointMatches;

		// ����2��ͨ���������BoW�ӿ쵱ǰ֡��ο�֮֡���������ƥ��
		// �������ƥ���ϵ��MapPoints����ά��
		int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurframe, vpMapPointMatches);
		if (nmatches < 15)
			return false;

		mCurframe.mvpMappts = vpMapPointMatches;
		Eigen::Matrix4d curfRT;
		curfRT.setIdentity();
		curfRT.block<3, 3>(0, 0) = mpReferenceKF->GetR();
		curfRT.block<3, 1>(0, 3) = mpReferenceKF->Gettrans();
		mCurframe.SetPose(curfRT);

		Optimizer::PoseOptimization(&mCurframe);

		//---����ɨβ���裬�Ǻǣ�
		int ngoods = 0;
		for (int i=0;i<mCurframe.mvpMappts.size();i++)
		{
			if (mCurframe.mvpMappts[i])
				if (mCurframe.mvbOutlier[i])
				{
					mCurframe.mvpMappts[i]= static_cast<MapPoint*>(NULL);
					mCurframe.mvbOutlier[i] = false;
				}
				else if (mCurframe.mvpMappts[i]->Obsnums()>0)
				{
					ngoods++;
				}
		}

		return ngoods >= 10;
	}

	bool Tracking::TrackWithMotionModel()
	{
		//---step1: ����projection��match
		ORBmatcher matcher(0.9, true);

		Eigen::Matrix4d Twc;
		Twc = mVelocity*mLastFrame.GetPose();
		mCurframe.SetPose(Twc);

		float th = 7.f;
		std::vector<MapPoint*>& vmappts=mCurframe.mvpMappts;
		std::fill(vmappts.begin(), vmappts.end(), static_cast<MapPoint*>(NULL));

		// step2������ƥ��
		int nmatchers=matcher.SearchByProjection(mCurframe,mLastFrame,th,true);
		if (nmatchers<20 )
		{
			std::fill(vmappts.begin(), vmappts.end(), static_cast<MapPoint*>(NULL));
			nmatchers = matcher.SearchByProjection(mCurframe, mLastFrame, th*2, true);
		}

		if (nmatchers < 20)
			return false;
		
		// step3: pose only �Ż�
		Optimizer::PoseOptimization(&mCurframe);

		// step4�� ɨβ����
		int nmatcherCounts = 0;
		for (int i=0; i<vmappts.size();i++)
			if (vmappts[i])
			{
				if (mCurframe.mvbOutlier[i])
				{
					mCurframe.mvbOutlier[i] = false;
					vmappts[i] = NULL;
				}
				else if (vmappts[i]->Obsnums() > 0)
					nmatcherCounts++;
			}
		
		return nmatcherCounts >= 10;
	}

	bool Tracking::TrackLocalMap()
	{
		// step1: local map�ĸ��£�
		UpdateLocalMap();

		// step2: local map�������. vary important function
		SearchLocalPoints();
	
		Optimizer::PoseOptimization(&mCurframe);

		// ɨβ����

		return true;
	}

	void Tracking::SearchLocalPoints()
	{

	}

	void Tracking::UpdateLocalMap()
	{
		UpdateLocalKeyFrames();
		UpdateLocalPoints();
	}

	void Tracking::UpdateLocalKeyFrames()
	{
		// step1:�ҳ�k1
		std::map<KeyFrame*, int> keyframeCounts;   // k1�еĹؼ�֡�����ƣ��۲쵽�ĵ������
		std::vector<MapPoint*>& curfMappts = mCurframe.mvpMappts;
		for (int i=0;i<curfMappts.size();i++)
		{
			if (! curfMappts[i]->IsBad())
			{
				std::map<KeyFrame*, int>& KFandfeatureId = curfMappts[i]->GetObservations();
				for (auto ite = KFandfeatureId.begin(); ite != KFandfeatureId.end(); ite++)
					keyframeCounts[ite->first]++;
			}
			else
			{
				curfMappts[i] = NULL;
			}
		}

		if (keyframeCounts.empty())
			return;

		// --��k1���ҳ���ţ���Ǹ��������Ӷ���ߵ���һ֡
		int maxCounts = -1;
		KeyFrame* pKFMost = NULL;

		mvpLocalKeyFrames.clear();
		mvpLocalKeyFrames.reserve(3 * keyframeCounts.size());
		for (auto ite= keyframeCounts.begin(); ite!=keyframeCounts.end();ite++)
		{
			if (ite->first->IsBad())
				continue;

			if (ite->second>maxCounts)
			{
				maxCounts = ite->second;
				pKFMost = ite->first;
			}
			mvpLocalKeyFrames.push_back(ite->first);
		}

		// ----step2: �ҳ�k2; 

	}

	void Tracking::UpdateLocalPoints()
	{
		mvpLocalMapPoints.clear();

		for (int i=0;i<mvpLocalKeyFrames.size();i++)
		{
			KeyFrame* pKFi = mvpLocalKeyFrames[i];
			if (pKFi->IsBad())
				continue;

			std::vector<MapPoint*> vpMPs=pKFi->GetMapPoints();

			for (int j=0;j<vpMPs.size();j++)
			{
				if (vpMPs[j]==NULL)
					continue;

				if (vpMPs[i]->mnTrackReferenceForFrame==mCurframe.mnId)
					continue;
				if (! vpMPs[j]->IsBad())
				{
					mvpLocalMapPoints.push_back(vpMPs[j]);
					vpMPs[j]->mnTrackReferenceForFrame = mCurframe.mnId;
				}
			}
		}
	}
}