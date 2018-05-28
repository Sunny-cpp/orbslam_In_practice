#include "MapPoint.h"

namespace ORBSlam
{
	long unsigned int MapPoint::mnextId = 0;

	MapPoint::MapPoint(const Eigen::Vector3d& pos, KeyFrame* pkeyframe, Map* pmap):
		mWorldPos(pos), mpMap(pmap), mpRefKF(pkeyframe), nObs(0),
		mbBad(false)
	{
		mWorldPos = pos;
		mNormalVector.setZero();

		mlnId = mnextId++;
	}

	void MapPoint::AddObservation(KeyFrame* pf, int id)
	{
		if (mObservations.count(pf))
			return;

		mObservations[pf] = id;     
		nObs++;
	}

}