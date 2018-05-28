#include "Initializer.h"
#include "Converter.h"

#include <math.h>
#include <thread>
#include <Thirdparty/DBoW2/DUtils/Random.h>

#include <Eigen/SVD>

using std::min;

namespace ORBSlam
{
	Initializer::Initializer(const Frame& ReferenceFrame, float sigma, int iterations):
		mfsigma(sigma), mfsigma2(sigma*sigma), miterations(iterations),
		mvKeypts1(ReferenceFrame.GetUnKeyPts())
	{
		ReferenceFrame.GetCameraPara(mcamK);
	}

	bool Initializer::Initialize(const Frame& curframe, const std::vector<int>& vIniMatches,
		Eigen::Matrix3d& R12, Eigen::Vector3d& T12, std::vector<cv::Point3f>& vp3d,
		std::vector<bool>& vbTriangulated)
	{
		mvKeypts2 = curframe.GetUnKeyPts();
		if (mvKeypts1.size() != vIniMatches.size())
			return false;

		//---step1: 做ransac set
		mvbMatched1.resize(mvKeypts1.size(),false);
		for (int i=0;i<vIniMatches.size();i++)
		{
			if (vIniMatches[i]>=0)
			{
				mvbMatched1[i] = true;
				mvMatchers12.push_back(std::make_pair(i, vIniMatches[i]));
			}
		}
		int nMatchCounts = mvMatchers12.size();
		std::vector<int> allMatchindice(nMatchCounts,0),curIndice;
		for (int i = 0; i < allMatchindice.size(); i++)
			allMatchindice[i] = i;

		DUtils::Random::SeedRandOnce(0);
		ransacSets.resize(miterations);
		for (int i=0;i<ransacSets.size();i++)
		{
			ransacSets[i].resize(8, 0);
			curIndice = allMatchindice;
			for (int j=0;j<8;j++)
			{
				int randId=DUtils::Random::RandomInt(0, curIndice.size() - 1);
				ransacSets[i][j] = curIndice[randId];

				curIndice[randId] = curIndice.back();
				curIndice.pop_back();
			}
		}

		//----step2: 分别求解H and F matrix
		std::vector<bool> vbMatchedInliersH, vbMatchedInliersF;
		float scoreH, scoreF;
		Eigen::Matrix3d finalF, finalH;
		std::thread threadH(&Initializer::FindHomography, this, std::ref(vbMatchedInliersH), std::ref(scoreH), std::ref(finalH));
		std::thread threadF(&Initializer::FindFundamental, this, std::ref(vbMatchedInliersF), std::ref(scoreF), std::ref(finalF));
		threadF.join();
		threadH.join();

		double RH = scoreH / (scoreH + scoreF);
		if (RH>0.45)
		{
			return DecomposeHmatrix(vbMatchedInliersH, finalH, R12, T12, vbTriangulated, vp3d, 1.0, 50);
		}
		else
		{
			return DecomposeFmatrix(vbMatchedInliersF, finalF, R12, T12, vbTriangulated, vp3d, 1.0, 50);
		}

		return false;
	}

	void Initializer::FindFundamental(std::vector<bool>& vbMatchesInliers, float& score, Eigen::Matrix3d& F21)
	{
		std::vector<cv::Point2f> vnorPts1, vnorPts2;
		Eigen::Matrix3d T1, T2;
		Normalize(mvKeypts1, vnorPts1, T1);
		Normalize(mvKeypts2, vnorPts2, T2);

		score = -1.0f;
		vbMatchesInliers.clear();

		std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> >v8Pts1(8), v8Pts2(8);

		for (int i=0;i<ransacSets.size();i++)
		{
			for (int j=0;j<ransacSets[i].size();j++)
			{
				int match12Index = ransacSets[i][j];
				v8Pts1[j] << vnorPts1[mvMatchers12[match12Index].first].x,
					vnorPts1[mvMatchers12[match12Index].first].y;
				v8Pts2[j] << vnorPts2[mvMatchers12[match12Index].second].x,
					vnorPts2[mvMatchers12[match12Index].second].y;
			}

			Eigen::Matrix3d F_cur = EigenPointsComputeFmatrix(v8Pts1, v8Pts2);
			F_cur = T2.transpose()*F_cur*T1;

			std::vector<bool>  vcurInliers;
			float curScore = CheckFundamental(F_cur, vcurInliers, mfsigma);
			if (curScore>score)
			{
				score = curScore;
				vbMatchesInliers = vcurInliers;
				F21 = F_cur;
			}
		}
	}

	float Initializer::CheckFundamental(const Eigen::Matrix3d& F, std::vector<bool>& vbMatchesInliers, const float sigma)
	{
		vbMatchesInliers.clear();
		int N = mvMatchers12.size();
		if (N <= 7)
			return -1.0;

		vbMatchesInliers.resize(N, false);
		float finalScore = 0.f;

		float f11 = F(0, 0);
		float f12 = F(0, 1);
		float f13 = F(0, 2);
		float f21 = F(1, 0);
		float f22 = F(1, 1);
		float f23 = F(1, 2);
		float f31 = F(2, 0);
		float f32 = F(2, 1);
		float f33 = F(2, 2);

		double invSigma = 1.0 / (sigma*sigma);
		const float th = 3.841;
		const float thScore = 5.991;

		bool isIniPts=true;

		for (int i=0;i<N;i++)
		{
			float u1 = mvKeypts1[mvMatchers12[i].first].pt.x;
			float v1 = mvKeypts1[mvMatchers12[i].first].pt.y;

			float u2 = mvKeypts2[mvMatchers12[i].second].pt.x;
			float v2 = mvKeypts2[mvMatchers12[i].second].pt.y;

			float l2a = f11*u1 + f12*v1 + f13;
			float l2b = f21*u1 + f22*v1 + f23;
			float l2c = f31*u1 + f32*v1 + f33;

			double geomedist1 = u2*l2a + v2*l2b + l2c;
			geomedist1 *= geomedist1;
			geomedist1 /= (l2a*l2a + l2b*l2b);

			double  chiSquare1 = geomedist1*invSigma;
			if (chiSquare1>th)
			{
				isIniPts = false;
			}
			else
			{
				finalScore += thScore - chiSquare1;
			}

			//----第二张图像上的投影
			float l1a = u2*f11 + v2*f21 + f31;
			float l1b = u2*f12 + v2*f22 + f32;
			float l1c = u2*f13 + v2*f23 + f33;
			double geomedist2 = l1a*u1 + l1b*v1 + l1c;
			geomedist2 *= geomedist2;
			geomedist2 /= (l1a*l1a + l1b*l1b);
			double  chiSquare2 = geomedist2*invSigma;
			if (chiSquare2>th)
			{
				isIniPts = false;
			}
			else
			{
				finalScore += thScore - chiSquare2;
			}

			if (isIniPts)
				vbMatchesInliers[i] = true;
			else
				vbMatchesInliers[i] = false;
		}

		return finalScore;
	}

	Eigen::Matrix3d Initializer::EigenPointsComputeFmatrix(const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> >& Pts1,
		const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> >& Pts2)
	{
		const int N = Pts1.size();
		Eigen::Matrix3d F;
		F.setZero();
		if (Pts2.size()!=Pts1.size() || N<=7)
		{
			return F;
		}

		Eigen::MatrixXd A(N, 9);
		for (int i=0;i<N;i++)
		{
			double u = Pts1[i](0);
			double v = Pts1[i](1);
			double u_ba = Pts1[i](0);
			double v_ba = Pts2[i](1);
			A.row(i) << u*u_ba, u*v_ba, u, v*u_ba, v*v_ba,
				v, u_ba, v_ba, 1;
		}
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,Eigen::ComputeFullV);
		Eigen::VectorXd f=svd.matrixV().col(8);

		F << f(0), f(1), (2),
			f(3), f(4), f(5),
			f(6), f(7), f(8);
		Eigen::JacobiSVD<Eigen::Matrix3d> fsvd(F, Eigen::ComputeFullV| Eigen::ComputeFullU);
		Eigen::Matrix3d fU=fsvd.matrixU();
		Eigen::Matrix3d fV = fsvd.matrixV();
		Eigen::Matrix3d fDiag;
		fDiag.setZero();
		fDiag(0, 0) = fsvd.singularValues()(0);
		fDiag(1, 1) = fsvd.singularValues()(1);

		return fU*fDiag*fV;
	}

	void Initializer::Normalize(const std::vector<cv::KeyPoint>& vKeys, std::vector<cv::Point2f>& vNormalizedPoints, Eigen::Matrix3d& T)
	{
		if (vKeys.size() <= 0)
			return;

		int N = vKeys.size();

		float meanX(0.f), meanY(0.f);
		for (int i=0;i<N;i++)
		{
			meanX += vKeys[i].pt.x;
			meanY += vKeys[i].pt.y;
		}

		meanX /= N;
		meanY /= N;

		float devX(0.f), devY(0.f);
		vNormalizedPoints.resize(N);
		for (int i=0;i<N;i++)
		{
			vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
			vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;
			
			devX += std::abs(vNormalizedPoints[i].x);
			devY += std::abs(vNormalizedPoints[i].y);
		}

		devX /= N;
		devY /= N;

		float sX = 1.0 / devX;
		float sY = 1.0 / devY;

		for (int i=0;i<N;i++)
		{
			vNormalizedPoints[i].x = vNormalizedPoints[i].x* sX;
			vNormalizedPoints[i].y = vNormalizedPoints[i].y*sY;
		}
		T << sX, 0.0, -meanX*sX,
			0.0, sY, -meanY*sY,
			0, 0, 1.0;
	}

	void Initializer::FindHomography(std::vector<bool>& vbMatchesInliers, float& score, Eigen::Matrix3d& H21)
	{
		std::vector<cv::Point2f> vnorPts1, vnorPts2;
		Eigen::Matrix3d T1, T2;
		Normalize(mvKeypts1, vnorPts1, T1);
		Normalize(mvKeypts2, vnorPts2, T2);

		std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > v8Pts1(8), v8Pts2(8);
		score = -1.0f;
		vbMatchesInliers.clear();

		for (int i = 0; i < ransacSets.size(); i++)
		{
			for (int j = 0; j < ransacSets[i].size(); j++)
			{
				int match12Index = ransacSets[i][j];
				v8Pts1[j] << vnorPts1[mvMatchers12[match12Index].first].x,
					vnorPts1[mvMatchers12[match12Index].first].y;
				v8Pts2[j] << vnorPts2[mvMatchers12[match12Index].second].x,
					vnorPts2[mvMatchers12[match12Index].second].y;
			}

			Eigen::Matrix3d H_cur = ComputeHomography(v8Pts1, v8Pts2);
			Eigen::Matrix3d H_true = T2.inverse()*H_cur*T1;

			std::vector<bool>  vcurInliers;
			float curScore = CheckHomography(H_true,vcurInliers,mfsigma);
			if (curScore>score)
			{
				score = curScore;
				vbMatchesInliers = vcurInliers;
				H21 = H_true;
			}
		}
	}

	Eigen::Matrix3d Initializer::ComputeHomography(const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> >& v8Pts1,
		const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> >& v8Pts2)
	{
		Eigen::Matrix3d H;
		H.setZero();
		if (v8Pts1.size()!=v8Pts2.size())
			return H;
		
		const int N = v8Pts1.size();
		if (N <= 3)
			return H;

		Eigen::MatrixXd A(2 * N, 9);
		for (int i=0;i<N;i++)
		{
			double x1 = v8Pts1[i](0);
			double y1 = v8Pts1[i](1);
			double x2 = v8Pts2[i](0);
			double y2 = v8Pts2[i](1);

			A.row(2 * i) << 0, 0, 0,
				-x1, -y1, -1.0,
				y2*x1, y2*y1, y2;
			A.row(2 * i + 1) << x1, y1, 1.0,
				0, 0, 0,
				-x2*x1, -x2*y1, -x2;
		}

		Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
		Eigen::VectorXd h = svd.matrixV().col(8);
		H << h(0), h(1), h(2), h(3),
			h(4), h(5), h(6), h(7), h(8);
		return H;
	}

	float Initializer::CheckHomography(const Eigen::Matrix3d& H, std::vector<bool>& vbMatchesInliers, float sigma)
	{
		Eigen::Matrix3d H_inv = H.inverse();
		vbMatchesInliers.clear();

		int N=mvMatchers12.size();
		if (N<=7)
		{
			return -1.0f;
		}

		const double th = 5.991;
		double invSigma = 1.0 / (sigma*sigma);
		vbMatchesInliers.resize(N, false);
		float finalScore = 0.0f;

		bool bInital = true;
		for (int i=0;i<N;i++)
		{
			float u1 = mvKeypts1[i].pt.x;
			float v1 = mvKeypts1[i].pt.y;

			float u2 = mvKeypts2[i].pt.x;
			float v2 = mvKeypts2[i].pt.y;

			Eigen::Vector3d x1, x2;
			x1 << u1, v1, 1.0;
			x2 << u2, v2, 1.0;

			Eigen::Vector3d x2_est = H*x1;
			x2_est /= x2_est(2);
			Eigen::Vector3d residual2 = x2 - x2_est;
			double dist2 = residual2.norm()*invSigma;
			if (dist2 > th)
				bInital = false;
			else
				finalScore += th - dist2;

			Eigen::Vector3d x1_est = H_inv*x2;
			x1_est /= x1_est(2);
			Eigen::Vector3d residual1 = x1 - x1_est;
			double dist1 = residual1.norm()*invSigma;
			if (dist1 > th)
				bInital = false;
			else
				finalScore += th - dist1;

			if (bInital)
				vbMatchesInliers[i] = true;
		}

		return finalScore;
	}

	bool Initializer::DecomposeFmatrix(const std::vector<bool>& vbCheckedInliers, const Eigen::Matrix3d& F,
		Eigen::Matrix3d& Rot, Eigen::Vector3d& Trans, std::vector<bool>& vbTriangulated,std::vector<cv::Point3f>& vP3d,
		const float minParallaxTh, const int minTriangulatedCounts)
	{
		Eigen::Matrix3d K= MatcamKtoEigen(mcamK);
		Eigen::Matrix3d E = K.transpose()*F*K;

		Eigen::Matrix3d R1, R2;
		Eigen::Vector3d t1, t2;
		DecomposeEtoRT(E, R1, R2, t1);
		t2 = -t1;

		std::vector<cv::Point3f> vP3d1, vP3d2, vP3d3, vP3d4;
		float thCheckRT = 4.0*mfsigma2;
		std::vector<bool> vbTriangulated1, vbTriangulated2, vbTriangulated3, vbTriangulated4;
		float parallax1, parallax2, parallax3, parallax4;

		int nGood1 = CheckRT(R1, t1, mvKeypts1, mvKeypts2, mvMatchers12, vbCheckedInliers, mcamK, vP3d1,thCheckRT, vbTriangulated1, parallax1);
		int nGood2 = CheckRT(R1, t2, mvKeypts1, mvKeypts2, mvMatchers12, vbCheckedInliers, mcamK, vP3d2, thCheckRT, vbTriangulated2, parallax2);
		int nGood3 = CheckRT(R2, t1, mvKeypts1, mvKeypts2, mvMatchers12, vbCheckedInliers, mcamK, vP3d3, thCheckRT, vbTriangulated3, parallax3);
		int nGood4 = CheckRT(R2, t2, mvKeypts1, mvKeypts2, mvMatchers12, vbCheckedInliers, mcamK, vP3d4, thCheckRT, vbTriangulated4, parallax4);

		int NInliers = 0;
		for (int i = 0; i < vbCheckedInliers.size(); i++)
			if (vbCheckedInliers[i])
				NInliers++;

		int nminGood = std::max(static_cast<int>(0.9*NInliers), minTriangulatedCounts);

		int ngoodRTTimes = 0;
		int maxGoods = std::max(nGood1 > nGood2 ? nGood1 : nGood2, nGood3 > nGood4 ? nGood3 : nGood4);

		if (maxGoods > 0.7*nGood1)
			ngoodRTTimes++;
		if (maxGoods > 0.7*nGood2)
			ngoodRTTimes++;
		if (maxGoods > 0.7*nGood3)
			ngoodRTTimes++;
		if (maxGoods > 0.7*nGood4)
			ngoodRTTimes++;

		if (ngoodRTTimes>1 || maxGoods<nminGood)
		{
			return false;
		}

		if (maxGoods==nGood1)
		{
			if (parallax1>minParallaxTh)
			{
				Rot = R1;
				Trans = t1;
				vbTriangulated = vbTriangulated1;
				vP3d = vP3d1;
				return true;
			}
		}
		else if (maxGoods == nGood2)
		{
			if (parallax2 > minParallaxTh)
			{
				Rot = R1;
				Trans = t2;
				vbTriangulated = vbTriangulated2;
				vP3d = vP3d2;
				return true;
			}
		}
		else if (maxGoods == nGood3)
		{
			if (parallax3 > minParallaxTh)
			{
				Rot = R2;
				Trans = t1;
				vbTriangulated = vbTriangulated3;
				vP3d = vP3d3;
				return true;
			}
		}
		else if (maxGoods == nGood4)
		{
			if (parallax4 > minParallaxTh)
			{
				Rot = R2;
				Trans = t2;
				vbTriangulated = vbTriangulated4;
				vP3d = vP3d4;
				return true;
			}
		}
		return false;
	}

	void Initializer::DecomposeEtoRT(const Eigen::Matrix3d& E, Eigen::Matrix3d& R1,
		Eigen::Matrix3d& R2, Eigen::Vector3d& T)
	{
		Eigen::Matrix3d W, WT;
		W << 0, -1, 0,
			1, 0, 0,
			0, 0, 1;
		WT = W.transpose();

		Eigen::JacobiSVD<Eigen::Matrix3d> svdE(E,Eigen::ComputeFullU||Eigen::ComputeFullV);
		Eigen::Matrix3d U = svdE.matrixU();
		Eigen::Matrix3d VT = svdE.matrixV().transpose();
		R1 = U*W*VT;
		if (R1.determinant() < 0)
			R1 = -R1;

		R2 = U*W.transpose()*VT;
		if (R2.determinant() < 0)
			R2 = -R2;

		T = U.col(2);
		T /= T.norm();
	}

	int Initializer::CheckRT(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
		const std::vector<cv::KeyPoint>& vKeys1, const std::vector<cv::KeyPoint>& vKeys2,
		const std::vector<std::pair<int, int> >& vMatches12,
		const std::vector<bool>& vbModelInliers,
		const cv::Mat& K,
		std::vector<cv::Point3f>& vP3D, const float th2, std::vector<bool>& vbGood, float& goodparallax)
	{
		int nGood = 0;  //统计有多少个好的可以被Triangulate的点
		std::vector<float> vCosParallax(0);

		float fx = K.at<float>(0, 0);
		float fy = K.at<float>(1, 1);
		float cx = K.at<float>(0, 2);
		float cy = K.at<float>(1, 2);

		int N = vKeys1.size();
		vbGood.resize(N, false);  // 可以被三角化的点标记为true，它以vkeyPts1作为遍历基准
		vP3D.clear();
		vP3D.resize(N);     // 跟vbGood一样

		Eigen::Matrix4d camP1, camP2;
		Eigen::Matrix3d camK = MatcamKtoEigen(K);
		camP1.setZero();
		camP2.setZero();
		camP1.block<3, 3>(0, 0) = camK*Eigen::Matrix3d::Identity();

		camP2.block<3, 3>(0, 0) = camK*R;
		camP2.block<3, 1>(0, 3) = camK*t;

		// 两个相机中心点，用于计算视差
		Eigen::Vector3d CO1, CO2;
		CO1.setZero();
		CO2 = -R.transpose()*t;

		for (int i=0;i<vMatches12.size();i++)
		{
			if (!vbModelInliers[i])
				continue;

			Eigen::Vector2d pt1(mvKeypts1[vMatches12[i].first].pt.x, mvKeypts1[vMatches12[i].first].pt.y);
			Eigen::Vector2d pt2(mvKeypts2[vMatches12[i].second].pt.x, mvKeypts2[vMatches12[i].second].pt.y);
			
			Eigen::Vector3d p3dX;
			Triangulate(pt1, pt2, camP1, camP2, p3dX);

			//对获取的p3dX进行筛选，从几个方面：数值有效，视差，depth，重投影误差大小
			// 1>数值有效性
			if(!std::isfinite(p3dX(0)) || !std::isfinite(p3dX(1)) || !std::isfinite(p3dX(2)))
				continue;

			// 2> 视差
			Eigen::Vector3d n1 = p3dX - CO1,n2=p3dX-CO2;
			double cosParallax = n1.dot(n2) / (n1.norm()*n2.norm());

			if (p3dX(2)<0 && cosParallax<0.9998)
				continue;

			Eigen::Vector3d p3dX2 = R*p3dX + t;
			if (p3dX2(2) < 0 && cosParallax < 0.9998)
				continue;

			// 3> 重投影误差
			float invZ1 = 1.0 / p3dX(2);
			Eigen::Vector2d estpt1(fx*p3dX(0)*invZ1 + cx, fy*p3dX(1)*invZ1 + cy);
			float squareError1 = (estpt1 - pt1).norm();
			if (squareError1>th2)
				continue;

			float invZ2 = 1.0 / p3dX2(2);
			Eigen::Vector2d estpt2(fx*p3dX2(0)*invZ2 + cx, fy*p3dX2(1)*invZ2 + cy);
			float squareError2 = (estpt2 - pt2).norm();
			if (squareError2 > th2)
				continue;

			if (cosParallax<0.9998)
			{
				nGood++;
				vbGood[vMatches12[i].first] = true;
				vP3D[vMatches12[i].first] = cv::Point3f(p3dX(0), p3dX(1), p3dX(2));
				vCosParallax.push_back(cosParallax);
			}
		}

		if (nGood>0)
		{
			std::sort(vCosParallax.begin(), vCosParallax.end());
			int indx = 50 < (vCosParallax.size() - 1) ? 50 : (vCosParallax.size() - 1);
			goodparallax = vCosParallax[indx];
		}
		else
		{
			goodparallax = 0.f;
		}
		return nGood;
	}

	void Initializer::Triangulate(const Eigen::Vector2d& kp1, const Eigen::Vector2d& kp2,
		const Eigen::Matrix4d& P1, const Eigen::Matrix4d& P2, Eigen::Vector3d& p3d)
	{
		Eigen::Matrix4d A;
		double u1 = kp1(0);
		double v1 = kp1(1);
		double u2 = kp2(0);
		double v2 = kp2(1);
		
		A.row(0) = u1*P1.row(2) - P1.row(0);
		A.row(1) = v1*P1.row(2) - P1.row(1);
		A.row(2) = u2*P2.row(2) - P2.row(0);
		A.row(3) = v2*P2.row(2) - P2.row(1);

		Eigen::JacobiSVD<Eigen::Matrix4d> SvdA(A, Eigen::ComputeFullV);
		Eigen::Vector4d hp3d = SvdA.matrixV().col(3);
		hp3d /= hp3d(3);
		p3d << hp3d(0), hp3d(1), hp3d(2);
	}

	bool Initializer::DecomposeHmatrix(const std::vector<bool>& vbCheckedInliers, const Eigen::Matrix3d& H,
		Eigen::Matrix3d& Rot, Eigen::Vector3d& Trans, std::vector<bool>& vbTriangulated, std::vector<cv::Point3f>& vP3d,
		const float minParallaxTh, const int minTriangulatedCounts)
	{

		Eigen::Matrix3d camK = MatcamKtoEigen(mcamK);
		Eigen::Matrix3d H_nor=camK.inverse()*H*camK;

		std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> > vRhypos(8);
		std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vThypos(8), vnhypos(8);

		Eigen::JacobiSVD<Eigen::Matrix3d> svdH(H_nor, Eigen::ComputeFullU || Eigen::ComputeFullV);
		Eigen::Matrix3d HU = svdH.matrixU();
		Eigen::Matrix3d HV = svdH.matrixV();
		float s = HU.determinant()*HV.determinant();

		float d1, d2, d3;
		d1 = svdH.singularValues()(0);
		d2 = svdH.singularValues()(1);
		d3 = svdH.singularValues()(2);
		
		//---如果d1≈d2, 或者d2≈d3,那么就直接放弃得了；本程序就只计算d1≠d2≠d3
		if (d1 / d2 <= 1.001 || d2 / d3 < 1.0001)
			return false;

		// 计算第一组的四个值
		float x1 = std::sqrt((d1*d1 - d2*d2) / (d1*d1 - d3*d3));
		float x3 = std::sqrt((d2*d2 - d3*d3) / (d1*d1 - d3*d3));
		float aux_x1[4] = { x1,x1 ,-x1,-x1};
		float aux_x3[4] = { x3,-x3,x3,-x3 };

		float costheta = (d1*d3 + d2*d2) / (d1*d2 + d3*d2);
		float sintheta = std::sqrt((d1*d1 - d2*d2)*(d2*d2 - d3*d3)) / (d1*d2 + d3*d2);
		float aux_sintheta[4] = { sintheta ,-sintheta ,-sintheta ,sintheta };

		//----做R'
		for (int i=0;i<4;i++)
		{
			Eigen::Matrix3d Rp;
			Rp.setIdentity();
			Rp(0, 0) = Rp(2, 2) = costheta;
			Rp(0, 2) = -1.0*aux_sintheta[i];
			Rp(2, 0) = aux_sintheta[i];

			Eigen::Matrix3d R = s*HU*Rp*HV.transpose();
			vRhypos[i] = R;

			Eigen::Vector3d tp = (d1 - d3)*Eigen::Vector3d(aux_x1[i], 0.0, -aux_x3[i]);
			Eigen::Vector3d t = HU*tp;
			vThypos[i] = t;
		}

		//----第二组四个值的计算
		costheta = (d1*d3 - d2*d2) / (d1*d2 - d3*d2);
		sintheta = std::sqrt((d1*d1 - d2*d2) / (d2*d2 - d3*d3)) / (d1*d2 - d3*d2);
		float new_aux_sintheta[4] = { sintheta ,-sintheta ,-sintheta ,sintheta };

		for (int i=0;i<4;i++)
		{
			Eigen::Matrix3d Rp;
			Rp.setZero();
			Rp(0, 0) = costheta;
			Rp(1, 1) = -1.0;
			Rp(2, 2) = -costheta;			
			Rp(2,0)=Rp(0, 2) = new_aux_sintheta[i];

			Eigen::Matrix3d R = s*HU*Rp*HV.transpose();
			vRhypos[i+4] = R;

			Eigen::Vector3d tp = (d1 + d3)*Eigen::Vector3d(aux_x1[i], 0.0, aux_x3[i]);
			Eigen::Vector3d t = HU*tp;
			vThypos[i+4] = t;
		}

		int bestbGoods = 0, secendnGoods = 0;

		int idbest = -1;
		std::vector<cv::Point3f> vbestp3d;
		std::vector<bool> vbestTriangulate;
		float bestParallax;

		for (int i=0;i<vRhypos.size();i++)
		{
			std::vector<cv::Point3f> vCurp3d;
			std::vector<bool> vbCurTriangulate;
			float curParallax;
			int nGoods=CheckRT(vRhypos[i],vThypos[i],
				mvKeypts1, mvKeypts2,
				mvMatchers12,
				vbCheckedInliers,
				mcamK,
				vCurp3d, 4.0*mfsigma2, vbCurTriangulate, curParallax);

			if (nGoods>bestbGoods)
			{
				vbestp3d = vCurp3d;
				vbestTriangulate = vbCurTriangulate;
				bestParallax = curParallax;
				idbest = i;

				secendnGoods = bestbGoods;
				bestbGoods = nGoods;
			}
			else if(nGoods>secendnGoods)
			{
				secendnGoods = nGoods;
			}
		}

		//------选择一个好的H的RT值，有n多同时需要满足的条件
		int N = 0;
		for (int i = 0; i < vbCheckedInliers.size(); i++)
			if (vbCheckedInliers[i])
				N++;
			
		if (secendnGoods<0.75*bestbGoods && bestParallax>minParallaxTh && bestbGoods>minTriangulatedCounts &&
			bestbGoods>0.85*N)
		{
			Rot = vRhypos[idbest];
			Trans = vThypos[idbest];
			vbTriangulated = vbestTriangulate;
			vP3d = vbestp3d;

			return true;
		}
		return false;
	}
}