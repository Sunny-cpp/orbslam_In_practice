#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include <list>

#include "KeyFrame.h"
#include "Map.h"

namespace ORBSlam
{
class LocalMapping
{
public:
	LocalMapping(Map* pMap);

	// LocalMapping的一系列相关操作
	void Run();
	void ProcessNewKeyFrame();

protected:
	bool CheckNewKeyFrames();  // 从头出帧
	void InsertKeyFrame(KeyFrame* pKF);  // 从尾插帧


private:
	std::list<KeyFrame*> mpLNewKeyframes;
	KeyFrame* mpcurKeyframe;
	Map* mpUniqueMap;
};
}
#endif
