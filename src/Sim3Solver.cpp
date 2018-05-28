#include "Sim3Solver.h"

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

  //------普遍意义来说，根据三个点来计算RTs
void Sim3Solver::ComputeSim3(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >& vPl,
	const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >& vPr)
{
	if (vPl.size() != vPr.size())
		return;

	//---step1:求重心
	Eigen::Vector3d p_Centroid_l, p_Centroid_r;
	p_Centroid_l.setZero();
	p_Centroid_r.setZero();

	int N = vPl.size();
	for (int i=0;i<N;i++)
	{
		p_Centroid_l += vPl[i];
		p_Centroid_r += vPr[i];
	}
	p_Centroid_l /= N;
	p_Centroid_r /= N;

	//----step2: 把点移到重心坐标系下
	std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > p0l(N), p0r(N);
	for (int i=0;i<N;i++)
	{
		p0l[i] = vPl[i] - p_Centroid_l;
		p0r[i] = vPr[i] - p_Centroid_r;
	}

	Eigen::Matrix3d M;
	M.setZero();
	for (int i=0;i<N;i++)
	{
		M += p0l[i] * p0r[i].transpose();
	}

	double* pM = M.data();
	Eigen::Matrix4d Nmat;
	Nmat << pM[0] + pM[4] + pM[8], pM[5] - pM[7], pM[6] - pM[2], pM[1] - pM[3],
		pM[5] - pM[7], pM[0] - pM[4] - pM[8], pM[1] + pM[3], pM[6] + pM[2],
		pM[6] - pM[2], pM[1] + pM[3], -pM[0] + pM[4] - pM[8], pM[5] + pM[7],
		pM[1] - pM[3], pM[6] + pM[2], pM[5] + pM[7], -pM[0] - pM[4] + pM[8];

	//------step3:求取R
	Eigen::EigenSolver<Eigen::Matrix4d> eigN(Nmat);
	Eigen::Vector4d q4 = eigN.eigenvectors().col(0);
	Eigen::Quaterniond quatR(q4);
	Eigen::Matrix3d Rot = quatR.toRotationMatrix();

	//--------step4: 求取s
	double scale(1.0), snom(0.0), sden(0.0);
	Eigen::Vector3d p0_newl;
	for (int i=0;i<N;i++)
	{
		p0_newl = Rot*p0l[i];
		snom += p0r[i].transpose()*p0_newl;
		sden += p0_newl.transpose()*p0_newl;
	}

	scale = snom / sden;

	//----step5: 求取t
	Eigen::Vector3d tran = p_Centroid_r - scale*Rot*p_Centroid_l;
}
