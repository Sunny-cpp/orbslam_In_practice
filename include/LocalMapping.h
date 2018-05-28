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

	// LocalMapping��һϵ����ز���
	void Run();
	void ProcessNewKeyFrame();

protected:
	bool CheckNewKeyFrames();  // ��ͷ��֡
	void InsertKeyFrame(KeyFrame* pKF);  // ��β��֡


private:
	std::list<KeyFrame*> mpLNewKeyframes;
	KeyFrame* mpcurKeyframe;
	Map* mpUniqueMap;
};
}
#endif
