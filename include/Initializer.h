#ifndef INITIALIZER_H
#define INITIALIZER_H

#include "Frame.h"

namespace ORBSlam
{
class Initializer
{
public:
	Initializer(const Frame& ReferenceFrame, float sigma = 1.0, int iterations = 200);

	bool Initialize(const Frame& curframe, const std::vector<int>& vIniMatches,
		Eigen::Matrix3d& R12, Eigen::Vector3d& T12, std::vector<cv::Point3f>& vp3d,
		std::vector<bool>& vbTriangulated);

private:
	float mfsigma, mfsigma2;

	//---这些变量用于做ransac
	int miterations;
	std::vector<std::vector<int> > ransacSets; // 外层：迭代的次数；内层：八点数集合
	std::vector<bool> mvbMatched1;
	std::vector<std::pair<int, int> > mvMatchers12;  // point1和point2的对应id

	std::vector<cv::KeyPoint> mvKeypts1, mvKeypts2;

	cv::Mat mcamK;

	// 计算F matrix的函数
	void FindFundamental(std::vector<bool>& vbMatchesInliers, float& score, Eigen::Matrix3d& F21);

	void Normalize(const std::vector<cv::KeyPoint>& vKeys, std::vector<cv::Point2f>& vNormalizedPoints, Eigen::Matrix3d& T);
	Eigen::Matrix3d EigenPointsComputeFmatrix(const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> >& Pts1,
		const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> >& Pts2);

	float CheckFundamental(const Eigen::Matrix3d& F, std::vector<bool>& vbMatchesInliers, float sigma);

	// 计算H matrix的函数
	void FindHomography(std::vector<bool>& vbMatchesInliers, float& score, Eigen::Matrix3d& H21);
	Eigen::Matrix3d ComputeHomography(const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> >& v8Pts1, 
		const std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> >& v8Pts2);
	float CheckHomography(const Eigen::Matrix3d& H, std::vector<bool>& vbMatchesInliers, float sigma);

	//从F or H matrix中分解出RT
	bool DecomposeFmatrix(const std::vector<bool>& vbCheckedInliers, const Eigen::Matrix3d& F,
		Eigen::Matrix3d& Rot, Eigen::Vector3d& Trans, std::vector<bool>& vbTriangulated, std::vector<cv::Point3f>& vP3d,
		const float minParallaxTh, const int minTriangulatedCounts);
	bool DecomposeHmatrix(const std::vector<bool>& vbCheckedInliers, const Eigen::Matrix3d& H,
		Eigen::Matrix3d& Rot, Eigen::Vector3d& Trans, std::vector<bool>& vbTriangulated, std::vector<cv::Point3f>& vP3d,
		const float minParallaxTh, const int minTriangulatedCounts);

	void DecomposeEtoRT(const Eigen::Matrix3d& E, Eigen::Matrix3d& R1,
		Eigen::Matrix3d& R2, Eigen::Vector3d& T);

	int CheckRT(const Eigen::Matrix3d& R, const Eigen::Vector3d& t, 
		const std::vector<cv::KeyPoint>& vKeys1, const std::vector<cv::KeyPoint>& vKeys2,
		const std::vector<std::pair<int, int> >& vMatches12,
		const std::vector<bool>& vbModelInliers,
		const cv::Mat& K, 
		std::vector<cv::Point3f>& vP3D, const float th2, std::vector<bool>& vbGood, float& goodparallax);

//	Triangulate(kp1, kp2, P1, P2, p3dC1);
	void Triangulate(const Eigen::Vector2d& kp1, const Eigen::Vector2d& kp2,
		const Eigen::Matrix4d& P1, const Eigen::Matrix4d& P2, Eigen::Vector3d& p3d);
};
}
#endif
