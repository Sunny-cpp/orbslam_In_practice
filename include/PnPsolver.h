#ifndef PNPSOLVER_H
#define PNPSOLVER_H

#include <Eigen/Core>
#include <vector>

namespace ORBSlam
{
class PnPsolver
{
public:
	PnPsolver();

	double compute_pose(Eigen::Matrix3d& rot, Eigen::Vector3d& trans);

private:
	//-----epnp的相关求解函数
	void choose_control_points();
 	void compute_barycentric_coordinates();
	void fill_M(Eigen::MatrixXd& Mmatrix);
 	void compute_L_6x10(const Eigen::Matrix<double,12,4>& v_1234,Eigen::Matrix<double,6,10>& L_6x10);
	void compute_rho(Eigen::Matrix<double,6,1>& rho);
	
	void find_betas_approx_1(const Eigen::Matrix<double, 6, 10>& L6X10,
		const Eigen::Matrix<double, 6, 1>& rho6, double* betas);
	void find_betas_approx_2(const Eigen::Matrix<double, 6, 10>& L6X10,
		const Eigen::Matrix<double, 6, 1>& rho6, double* betas);
	void find_betas_approx_3(const Eigen::Matrix<double, 6, 10>& L6X10,
		const Eigen::Matrix<double, 6, 1>& rho6, double* betas);

	void gauss_newton(const Eigen::Matrix<double, 6, 10>& L6X10,
		const Eigen::Matrix<double, 6, 1>& rho6, double* betas);  //betas开始给定初值，函数结束后给出优化值
	void compute_A_and_b_gauss_newton(const Eigen::Matrix<double, 6, 10>& L6X10,
		const Eigen::Matrix<double, 6, 1>& rho6, const Eigen::Vector4d& beat,
		Eigen::Matrix<double, 6, 4>& A, Eigen::Matrix<double, 6, 1>& b);

	//----用u and beat计算RT的值，返回的是重投影误差的大小
	double compute_R_and_t(const Eigen::Matrix<double, 12, 4>& V_1234, const double* betas,
		Eigen::Matrix3d& R, Eigen::Vector3d& t);

	void compute_ccs(const Eigen::Matrix<double, 12, 4>& V_1234, const double* betas);  // 计算control point在camera system下的坐标
	void compute_pcs();
	void estimate_R_and_t(Eigen::Matrix3d& R, Eigen::Vector3d& t);   // 用naive ICP得到进行对齐RT
	double reprojection_error(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
	// epnp相关的变量
	std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > Cws, Ccs;  // 两组control point 
	std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > Pws,Pcs;  //world坐标系中的3d point
	std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > pus;  //图像点，2d n个
	std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > alpha_ij;
	int number_of_correspondences;  // 有多少组的对应点，算是Pws的size吧

	//------相机参数相关
	double fx, fy, cx, cy;
};
}
#endif
