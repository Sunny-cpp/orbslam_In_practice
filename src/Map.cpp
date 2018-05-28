#include "Map.h"

namespace ORBSlam
{
	Map::Map():mnMaxKFid(0)
	{}

	void Map::AddKeyFrame(KeyFrame* pkeyf)
	{
		mspKeyframes.insert(pkeyf);
		if (pkeyf->mlnId>mnMaxKFid)
		{
			mnMaxKFid = pkeyf->mlnId;
		}
	}

	void Map::AddMapPoint(MapPoint* pPts)
	{
		mspMapPts.insert(pPts);
	}

	std::vector<KeyFrame*> Map::GetKeyFrames() const
	{
		std::vector<KeyFrame*> out(mspKeyframes.begin(), mspKeyframes.end());
		return out;
	}

	std::vector<MapPoint*> Map::GetMapPoints() const
	{
		std::vector<MapPoint*> out(mspMapPts.begin(), mspMapPts.end());
		return out;
	}
}