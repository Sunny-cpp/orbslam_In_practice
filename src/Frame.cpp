#include "Frame.h"

namespace ORBSlam
{
	long long int Frame::mllnextid=0;
    bool Frame::mbInitalframe=true;
	cv::Mat Frame::mcamk, Frame::mvdisCoffes;
	float Frame::miMinX, Frame::miMaxX, Frame::miMinY, Frame::miMaxY;

	Frame::Frame():mnId(-1), mpOrbextractor(static_cast<ORBextractor*>(NULL))
	{}

	Frame::Frame(const Frame& thr): mnId(thr.mnId), mpOrbextractor(thr.mpOrbextractor),
		mvKeypts(thr.mvKeypts), mvUnKeypts(thr.mvUnKeypts), mcvDescriptors(thr.mcvDescriptors.clone())		
	{
		for (int i = 0; i < FRAME_GRID_COLS; i++)
			for (int j = 0; j < FRAME_GRID_ROWS; j++)
				mGrid[i][j] = thr.mGrid[i][j];

		int N = thr.mvUnKeypts.size();
		mvpMappts.resize(N, static_cast<MapPoint*>(NULL));
	}

	Frame& Frame::operator=(const Frame& thr)
	{
		mnId = thr.mnId;
		mpOrbextractor = thr.mpOrbextractor;
		mvKeypts = thr.mvKeypts;
		mvUnKeypts = thr.mvUnKeypts;
		mcvDescriptors = thr.mcvDescriptors.clone();

		for (int i = 0; i < FRAME_GRID_COLS; i++)
			for (int j = 0; j < FRAME_GRID_ROWS; j++)
				mGrid[i][j] = thr.mGrid[i][j];

		mvpMappts.resize(thr.mvUnKeypts.size(), static_cast<MapPoint*>(NULL));
		return *this;
	}

	Frame::Frame(const double timeImg, const cv::Mat& img, ORBextractor* porbextractor,
		const Eigen::Matrix3d& camK, const Eigen::Matrix<double, 1, 5>& discoffes):mpOrbextractor(porbextractor)
	{
		if (mbInitalframe)
		{
			//---传递内参
			cv::Mat k(camK.rows(), camK.cols(), CV_32FC1);
			for (int i=0;i<k.rows;i++)
				for(int j=0;j<k.cols;j++)
					k.at<float>(i, j) = camK(i, j);
			mcamk = k.clone();
				
			cv::Mat vecdis(discoffes.cols(), 1, CV_32FC1);
			for (int i = 0; i <vecdis.rows; i++)
				vecdis.at<float>(i) = discoffes(i);
			mvdisCoffes = vecdis.clone();

			//----分配grid参数
			FindimageBound(img);
			mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (miMaxX - miMinX);
			mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (miMaxY - miMinY);

			mbInitalframe = false;
		}
		ExtractorOrbFeatures(img);  // step1：提取

		// step2:去畸变
		UndistortedKeyPoints();
		// step3:分配格子
		AssignFeaturesToGrid();

		//---给其他相关的变量赋值
		mvpMappts.resize(mvUnKeypts.size(), static_cast<MapPoint*>(NULL));
	}

	void Frame::ExtractorOrbFeatures(const cv::Mat& curImg)
	{
		(*mpOrbextractor)(curImg, cv::Mat(), mvKeypts, mcvDescriptors);
	}

	void Frame::UndistortedKeyPoints()
	{
		if (mvdisCoffes.at<float>(0)==0.0)
		{
			mvUnKeypts = mvKeypts;
			return;
		}
		int N = mvKeypts.size();
		if (N <= 0)
			return;
		cv::Mat keyPts(N, 2, CV_32FC1);
		for (int i=0;i<N;i++)
		{
			keyPts.at<float>(i, 0) = mvKeypts[i].pt.x;
			keyPts.at<float>(i, 1) = mvKeypts[i].pt.y;
		}
		keyPts=keyPts.reshape(2);
		cv::Mat UnkeyPts;
		cv::undistortPoints(keyPts, UnkeyPts, mcamk, mvdisCoffes, cv::Mat(), mcamk);
		UnkeyPts = UnkeyPts.reshape(1);

		mvUnKeypts.resize(N);
		for (int i=0;i<N;i++)
		{
			cv::KeyPoint kpt = mvKeypts[i];
			kpt.pt.x = UnkeyPts.at<float>(i, 0);
			kpt.pt.y = UnkeyPts.at<float>(i, 0);
			mvUnKeypts[i] = kpt;
		}
	}

	void Frame::FindimageBound(const cv::Mat& initalImg)
	{
		if (mvdisCoffes.at<float>(0)==0.0)
		{
			miMinX = 0;
			miMaxX = initalImg.cols;
			miMinY = 0;
			miMaxY = initalImg.rows;
		}
		else
		{
			cv::Mat cornerPts(4, 2, CV_32FC1);
			cornerPts.at<float>(0, 0) = 0.0;         //左上
			cornerPts.at<float>(0, 1) = 0.0;
			cornerPts.at<float>(1, 0) = initalImg.cols; //右上
			cornerPts.at<float>(1, 1) = 0.0;
			cornerPts.at<float>(2, 0) = 0.0;         //左下
			cornerPts.at<float>(2, 1) = initalImg.rows;
			cornerPts.at<float>(3, 0) = initalImg.cols; //右下
			cornerPts.at<float>(3, 0) = initalImg.cols; //右下
			cornerPts.at<float>(3, 1) = initalImg.rows;
			cornerPts = cornerPts.reshape(2);

			cv::Mat UndisCornerPts;
			cv::undistortPoints(cornerPts, UndisCornerPts, mcamk, mvdisCoffes, cv::Mat(), mcamk);
			UndisCornerPts = UndisCornerPts.reshape(1);
			miMinX = std::min(UndisCornerPts.at<float>(0, 0), UndisCornerPts.at<float>(2, 0));
			miMaxX = std::min(UndisCornerPts.at<float>(1, 0), UndisCornerPts.at<float>(3, 0));
			miMinY= std::min(UndisCornerPts.at<float>(0, 1), UndisCornerPts.at<float>(1, 1));
			miMaxY = std::min(UndisCornerPts.at<float>(2, 1), UndisCornerPts.at<float>(3, 1));
		}
	}

	void Frame::AssignFeaturesToGrid()
	{
		int N = mvUnKeypts.size();
		for (int i=0;i<FRAME_GRID_COLS;i++)
			for (int j=0;j<FRAME_GRID_ROWS;j++)
			{
				mGrid[i][j].reserve(N / 2);
			}
		for (int i=0;i<N;i++)
		{
			const cv::KeyPoint kpts = mvUnKeypts[i];
			int x_id, y_id;
			if (GetGridId(kpts.pt.x, kpts.pt.y, x_id, y_id))
				mGrid[x_id][y_id].push_back(i);
		}
	}

	bool Frame::GetGridId(const double& x, const double&y, int& index_x, int& index_y)
	{
		index_x = std::round((x - miMinX)*mfGridElementWidthInv);
		index_y = std::round((y - miMaxY)*mfGridElementHeightInv);
		if (index_x < 0 || index_x >= FRAME_GRID_COLS || index_y < 0 || index_y >= FRAME_GRID_ROWS)
			return false;
		return true;
	}

	bool Frame::isInFrustum(MapPoint* pMp, const double viewingCosLimit)
	{
		Eigen::Vector3d posW= pMp->Getpos();
		
		// 1.depth
		Eigen::Vector3d posC = mRwc*posW + mtwc;
		if (posC(2) < 0)
			return false;

		// 2. project
		float fx = mcamk.at<float>(0, 0);
		float fy = mcamk.at<float>(1, 1);
		float cx = mcamk.at<float>(0, 2);
		float cy = mcamk.at<float>(1, 2);

		float ui = cx + fx*posC(0) / posC(2);
		float vi = cy + fy*posC(1) / posC(2);

		if (ui < miMinX || ui >= miMaxY || vi < miMinY || vi >= miMaxY)
			return false;

		//--- 3. distance
		Eigen::Vector3d C0 = -mRwc*mtwc;  // 本相机的光心位置
		Eigen::Vector3d v = posC - C0;
		double maxDist, minDist;  // 这两个值应该从pMp中来，但是我没有写
		if (v.norm() < minDist || v.norm() > maxDist)
			return false;

		// ------ 4. angle between v and n
		Eigen::Vector3d n = pMp->GetNormalVector();
		double cosValue = v.dot(n) / (v.norm()*n.norm());
		if (cosValue < viewingCosLimit)
			return false;

		return true;
	}

	void Frame::GetCameraPara(cv::Mat& camk)
	{
		camk = mcamk.clone();
	}

	void Frame::SetPose(const Eigen::Matrix4d& pose)
	{
		mTwc = pose;
		mRwc = mTwc.block<3, 3>(0, 0);
		mtwc = mTwc.block<3, 1>(0, 3);
	}

	std::vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
	{
		std::vector<size_t> vIndices;
		int N = mvUnKeypts.size();
		vIndices.reserve(N);

		const int nMinCellX = std::max(0, (int)floor((x - miMinX - r)*mfGridElementWidthInv));
		if (nMinCellX >= FRAME_GRID_COLS)
			return vIndices;

		const int nMaxCellX = std::min((int)FRAME_GRID_COLS - 1, (int)ceil((x - miMinX + r)*mfGridElementWidthInv));
		if (nMaxCellX < 0)
			return vIndices;

		const int nMinCellY = std::max(0, (int)floor((y - miMinY - r)*mfGridElementHeightInv));
		if (nMinCellY >= FRAME_GRID_ROWS)
			return vIndices;

		const int nMaxCellY = std::min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - miMinY + r)*mfGridElementHeightInv));
		if (nMaxCellY < 0)
			return vIndices;

		const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

		for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
		{
			for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
			{
				const std::vector<int> vCell = mGrid[ix][iy];
				if (vCell.empty())
					continue;

				for (size_t j = 0, jend = vCell.size(); j < jend; j++)
				{
					const cv::KeyPoint &kpUn = mvUnKeypts[vCell[j]];
					if (bCheckLevels)
					{
						if (kpUn.octave < minLevel)
							continue;
						if (maxLevel >= 0)
							if (kpUn.octave > maxLevel)
								continue;
					}
					const float distx = kpUn.pt.x - x;
					const float disty = kpUn.pt.y - y;
					if (fabs(distx) < r && fabs(disty) < r)
						vIndices.push_back(vCell[j]);
				}
			}
		}

		return vIndices;
	}

	// ---还未实现
	void Frame::ComputeBOW()
	{

	}
}