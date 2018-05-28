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
	//-----epnp�������⺯��
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
		const Eigen::Matrix<double, 6, 1>& rho6, double* betas);  //betas��ʼ������ֵ����������������Ż�ֵ
	void compute_A_and_b_gauss_newton(const Eigen::Matrix<double, 6, 10>& L6X10,
		const Eigen::Matrix<double, 6, 1>& rho6, const Eigen::Vector4d& beat,
		Eigen::Matrix<double, 6, 4>& A, Eigen::Matrix<double, 6, 1>& b);

	//----��u and beat����RT��ֵ�����ص�����ͶӰ���Ĵ�С
	double compute_R_and_t(const Eigen::Matrix<double, 12, 4>& V_1234, const double* betas,
		Eigen::Matrix3d& R, Eigen::Vector3d& t);

	void compute_ccs(const Eigen::Matrix<double, 12, 4>& V_1234, const double* betas);  // ����control point��camera system�µ�����
	void compute_pcs();
	void estimate_R_and_t(Eigen::Matrix3d& R, Eigen::Vector3d& t);   // ��naive ICP�õ����ж���RT
	double reprojection_error(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
	// epnp��صı���
	std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > Cws, Ccs;  // ����control point 
	std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > Pws,Pcs;  //world����ϵ�е�3d point
	std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > pus;  //ͼ��㣬2d n��
	std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > alpha_ij;
	int number_of_correspondences;  // �ж�����Ķ�Ӧ�㣬����Pws��size��

	//------����������
	double fx, fy, cx, cy;
};
}
#endif
