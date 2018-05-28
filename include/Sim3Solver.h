#ifndef SIM3_SOLVER_H
#define SIM3_SOLVER_H

#include <Eigen/Core>
#include <vector>

class Sim3Solver
{
public:
	Sim3Solver();

private:
	void ComputeSim3(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >& vPl,
		const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >& vPr);

};
#endif
