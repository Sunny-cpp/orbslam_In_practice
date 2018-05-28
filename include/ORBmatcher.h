#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include "Frame.h"

namespace ORBSlam
{
class ORBmatcher
{
public:
	ORBmatcher(float nnratio = 0.6, bool checkOri = true):mfNNratio(nnratio),
		mbCheckOrientation(checkOri)
	{
	}

	int SearchForInitialization(Frame& F1, Frame& F2, 
		std::vector<cv::Point2f>& vbPrevMatched, std::vector<int> &vnMatches12, int windowSize);

	int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
	void ComputeThreeMaxima(std::vector<int>* histo, const int L, int& ind1, int& ind2, int& ind3);

	int SearchByBoW(KeyFrame* pKF1, Frame F2, std::vector<MapPoint*> &vpMatches12) {} 

	int SearchByProjection(Frame& CurrentFrame, const Frame& LastFrame, const float th, const bool bMono) {}

private:
	static const int HISTO_LENGTH;
	static const int TH_LOW;

	float mfNNratio;
	bool mbCheckOrientation;
};
}
#endif
