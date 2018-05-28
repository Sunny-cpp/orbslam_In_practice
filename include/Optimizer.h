#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "KeyFrame.h"

namespace ORBSlam
{
class Optimizer
{
public:
	void static BundleAdjustment(const std::vector<KeyFrame*>& vpKF, const std::vector<MapPoint*>& vpMP,
		int nIterations = 5, bool* pbStopFlag = NULL, const unsigned long nLoopKF = 0,
		const bool bRobust = true);

	// --所谓全局，就是优化all the points and pose
	void static GlobalBundleAdjustemnt(Map* pMap, int nIterations = 5, bool *pbStopFlag = NULL,
		const unsigned long nLoopKF = 0, const bool bRobust = true);

	int static PoseOptimization(Frame* pFrame);
};

}
#endif
