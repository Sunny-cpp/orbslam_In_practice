#ifndef MAP_H
#define MAP_H

#include "KeyFrame.h"
#include "MapPoint.h"

#include <set>

namespace ORBSlam
{
	class KeyFrame;
	class MapPoint;

class Map
{
public:
	Map();

	void AddKeyFrame(KeyFrame* pkeyf);
	void AddMapPoint(MapPoint* pPts);

	// get keyframe and mappoint
	std::vector<KeyFrame*> GetKeyFrames() const; 
	std::vector<MapPoint*> GetMapPoints() const;

private:
	long unsigned int mnMaxKFid;    //标识当前所有keyframe中最大的frameid号码
	std::set<KeyFrame*> mspKeyframes;
	std::set<MapPoint*> mspMapPts;

	std::vector<MapPoint*> mvReferenceMapPoints;
};

}
#endif
