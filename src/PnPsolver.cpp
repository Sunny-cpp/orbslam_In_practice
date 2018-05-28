#include "PnPsolver.h"

#include <Eigen/Svd>
#include <Eigen/QR>

#include <iostream>
#include <math.h>

namespace ORBSlam
{
	PnPsolver::PnPsolver()
	{
	}

	double PnPsolver::compute_pose(Eigen::Matrix3d& rot, Eigen::Vector3d& trans)
	{
		//---step1: 选择四个world control point
		choose_control_points();

		//----step2：计算系数alphas
		compute_barycentric_coordinates();

		//----step3：获取M矩阵
		Eigen::MatrixXd M;
		M.resize(2 * number_of_correspondences, 12);
		fill_M(M);

		Eigen::Matrix<double, 12, 12> MTM, V;
		MTM = M.transpose()*M;
		Eigen::JacobiSVD<Eigen::Matrix<double, 12, 12> > svdMTM(MTM, Eigen::ComputeFullV);
		V = svdMTM.matrixV();

		//----step4: 写出L6X10矩阵 和rho6X1矩阵
		Eigen::Matrix<double, 12, 4> V_1234 = V.block<12, 4>(0, 8);
		Eigen::Matrix<double, 6, 10> L_6_10;
		Eigen::Matrix<double, 6, 1> rho6;
		compute_L_6x10(V_1234,L_6_10);
		compute_rho(rho6);

		//----step5: 计算出几个beta值
		std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> > vR;
		std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > vt;
		vR.resize(3);
		vt.resize(3);   // 只求解出三组即可

		double Beats[3][4];
		find_betas_approx_1(L_6_10, rho6, Beats[0]);
		gauss_newton(L_6_10,rho6,Beats[0]);
		double repro1 = compute_R_and_t(V_1234, Beats[0], vR[0], vt[0]);

		find_betas_approx_2(L_6_10, rho6, Beats[1]);
		gauss_newton(L_6_10, rho6, Beats[1]);
		double repro2 = compute_R_and_t(V_1234, Beats[1], vR[1], vt[1]);

		find_betas_approx_3(L_6_10, rho6, Beats[2]);
		gauss_newton(L_6_10, rho6, Beats[2]);
		double repro3 = compute_R_and_t(V_1234, Beats[2], vR[2], vt[2]);

		int chooseN = 1;
		if (repro2<repro1)
		{
			chooseN = repro3 < repro2 ? 3 : 2;
		}
		
		double repros[3] = { repro1, repro2, repro3 };
		return repros[chooseN - 1];
	}

	double PnPsolver::compute_R_and_t(const Eigen::Matrix<double, 12, 4>& V_1234, const double* betas,
		Eigen::Matrix3d& R, Eigen::Vector3d& t)
	{
		//计算control point on camera下的值
		 compute_ccs(V_1234, betas);

		 //计算3d point在camera system下的坐标值
		 compute_pcs();
		 estimate_R_and_t(R, t);
		 return reprojection_error(R, t);
	}

	double PnPsolver::reprojection_error(const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
	{
		double sum = 0.0;
		for (int i=0;i<number_of_correspondences;i++)
		{
			Eigen::Vector3d pc_est = R*Pws[i] + t;
			double u_est = cx + fx*pc_est(0) / pc_est(2);
			double v_est = cy + fy*pc_est(1) / pc_est(2);
			Eigen::Vector2d pu_est(u_est, v_est);
			sum += (pus[i] - pu_est).norm();
		}
		return sum / number_of_correspondences;
	}

	void PnPsolver::compute_ccs(const Eigen::Matrix<double, 12, 4>& V_1234, const double* betas)
	{
		Ccs.resize(4);
		Eigen::Matrix<double, 12, 1> Vsum;
		Vsum.setZero();
		for (int i=0;i<4;i++)
		{
			Vsum += V_1234.col(3 - i)*betas[i];
		}
		for (int i=0;i<4;i++)
		{
			Ccs.push_back(Vsum.segment<3>(i * 3));
		}
	}

	void PnPsolver::estimate_R_and_t(Eigen::Matrix3d& R, Eigen::Vector3d& t)
	{
		Eigen::Vector3d Pc_mean, Pw_mean;
		Pc_mean.setZero();
		Pw_mean.setZero();

		for (int i=0;i<number_of_correspondences;i++)
		{
			Pc_mean += Pcs[i];
			Pw_mean += Pws[i];
		}
		Pc_mean /= number_of_correspondences;
		Pw_mean /= number_of_correspondences;

		Eigen::Matrix3d H;
		H.setZero();
		for (int i=0;i<number_of_correspondences;i++)
		{
			H += (Pcs[i] - Pc_mean)*(Pws[i] - Pw_mean).transpose();
		}

		Eigen::JacobiSVD<Eigen::Matrix3d> svdH(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
		R = svdH.matrixV()*svdH.matrixU().transpose();
		t = Pc_mean - R*Pw_mean;
	}

	void PnPsolver::compute_pcs()
	{
		Pcs.clear();
		Pcs.resize(number_of_correspondences);
		for (int i=0;i<number_of_correspondences;i++)
		{
			Pcs[i].setZero();
			for (int j = 0; j < 4; j++)
				Pcs[i] += Ccs[j] * alpha_ij[i](j);
		}
	}

	void PnPsolver::compute_rho(Eigen::Matrix<double, 6, 1>& rho)
	{
		rho(0) = (Cws[0] - Cws[1]).norm();
		rho(1) = (Cws[0] - Cws[2]).norm();
		rho(2) = (Cws[0] - Cws[3]).norm();
		rho(3) = (Cws[1] - Cws[2]).norm();
		rho(4) = (Cws[1] - Cws[3]).norm();
		rho(5) = (Cws[2] - Cws[3]).norm();
	}

	void PnPsolver::fill_M(Eigen::MatrixXd& Mmatrix)
	{
		if (Mmatrix.cols() != 12 || Mmatrix.rows() != 2 * number_of_correspondences)
			Mmatrix.resize(2 * number_of_correspondences, 12);

		for (int i=0;i<number_of_correspondences;i++)
			for (int j=0;j<4;j++)
			{
				Mmatrix(i * 2,3 * j) = fx*alpha_ij[i][j];
				Mmatrix(i * 2, 3 * j + 1) = 0.0;
				Mmatrix(i * 2, 3 * j + 2) = (cx - pus[i](0))*alpha_ij[i][j];

				Mmatrix(i * 2+1, 3 * j) = 0.0;
				Mmatrix(i * 2+1, 3 * j + 1) = fy*alpha_ij[i][j];
				Mmatrix(i * 2+1, 3 * j + 2) = (cy - pus[i](1))*alpha_ij[i][j];
			}
	}

	void PnPsolver::choose_control_points()
	{
		Eigen::Vector3d centroidPt;
		centroidPt.setZero();
		for (int i=0;i<number_of_correspondences;i++)
		{
			centroidPt += Pws[i];
		}
		centroidPt /= number_of_correspondences;

		Eigen::MatrixXd PwN3;
		PwN3.resize(number_of_correspondences, 3);
		for (int i = 0; i < number_of_correspondences; i++)
			PwN3.row(i) = Pws[i] - centroidPt;

		Eigen::Matrix3d PWTPW = PwN3.transpose()*PwN3;
		Eigen::JacobiSVD<Eigen::Matrix3d> svdPtP(PWTPW,Eigen::ComputeFullU||Eigen::ComputeFullV);
		Eigen::Matrix3d U = svdPtP.matrixU();

		//----第一个是重心，其他三个依次是PCA主分量
		Cws.clear();
		Cws.push_back(centroidPt);
		for (int i=0;i<3;i++)
		{
			Eigen::Vector3d tempcw = svdPtP.singularValues()(i)*U.col(i) + centroidPt;
			Cws.push_back(tempcw);
		}
	}

	void PnPsolver::compute_barycentric_coordinates()
	{
		alpha_ij.resize(number_of_correspondences);
		
		//----step1: 求CC的逆矩阵
		Eigen::Matrix3d CC,CC_inv;
		CC << Cws[1] - Cws[0], Cws[2] - Cws[0], Cws[3] - Cws[0];
		CC_inv = CC.inverse();

		//----step2: 计算每组alpha
		for (int i=0;i<number_of_correspondences;i++)
		{
			Eigen::Vector3d normalPws = Pws[i] - Cws[0];
			Eigen::Vector3d alpha_ij234 = CC_inv*normalPws;
			double alpha_ij0 = 1.0 - alpha_ij234(0) - alpha_ij234(1) - alpha_ij234(2);
			alpha_ij[i] << alpha_ij0, alpha_ij234;
		}
	}

	void PnPsolver::compute_L_6x10(const Eigen::Matrix<double, 12, 4>& v_1234, Eigen::Matrix<double, 6, 10>& L_6x10)
	{
		Eigen::Matrix<Eigen::Vector3d, 4, 6> dvkij;   // k=1~4, 表示vk-vk的差值， j表示1~6，表示横着的<i,j>组合
		for (int k = 0; k < 4; k++)
		{
			Eigen::VectorXd vk = v_1234.col(3 - k);
			int a = 0, b = 1;
			for (int j = 0; j < 6; j++)
			{
				dvkij(k, j) = vk.segment<3>(a * 3) - vk.segment<3>(b * 3);

				b++;
				if (b>3)
				{
					a++;
					b = a + 1;
				}
			}
		}

		for (int i=0;i<6;i++)
		{
			L_6x10.row(i) << dvkij(0, i).dot(dvkij(0, i)),
				dvkij(0, i).dot(dvkij(1, i)),
				dvkij(1, i).dot(dvkij(1, i)),
				dvkij(0, i).dot(dvkij(2, i)),
				dvkij(2, i).dot(dvkij(1, i)),
				dvkij(2, i).dot(dvkij(2, i)),
				dvkij(0, i).dot(dvkij(3, i)),
				dvkij(1, i).dot(dvkij(3, i)),
				dvkij(2, i).dot(dvkij(3, i)),
				dvkij(3, i).dot(dvkij(3, i));
		}
	}

	void PnPsolver::compute_A_and_b_gauss_newton(const Eigen::Matrix<double, 6, 10>& L6X10,
		const Eigen::Matrix<double, 6, 1>& rho6, const Eigen::Vector4d& beat,
		Eigen::Matrix<double, 6, 4>& A, Eigen::Matrix<double, 6, 1>& b)
	{
		A << 2.0*L6X10.col(0)*beat(0) + L6X10.col(1)*beat(1) + L6X10.col(3)*beat(2) + L6X10.col(6)*beat(3),
			L6X10.col(1)*beat(0) + 2.0*L6X10.col(1)*beat(1) + L6X10.col(4)*beat(2) + L6X10.col(7)*beat(3),
			L6X10.col(3)*beat(0) + L6X10.col(4)*beat(1) + 2.0*L6X10.col(5)*beat(2) + L6X10.col(8)*beat(3),
			L6X10.col(6)*beat(0) + L6X10.col(7)*beat(1) + L6X10.col(8)*beat(2) + 2.0*L6X10.col(9)*beat(3);

		Eigen::Matrix<double, 10, 1> pij;
		const double* pb = beat.data();
		pij << pb[0] * pb[0],
			pb[0] * pb[1], pb[1] * pb[1],
			pb[0] * pb[2], pb[1] * pb[2],
			pb[2] * pb[2], pb[0] * pb[3],
			pb[1] * pb[3], pb[2] * pb[3],
			pb[3] * pb[3];
		b = rho6 - L6X10*pij;
	}

	void PnPsolver::gauss_newton(const Eigen::Matrix<double, 6, 10>& L6X10,
		const Eigen::Matrix<double, 6, 1>& rho6, double* betas)
	{
		Eigen::Matrix<double, 6, 4> A;
		Eigen::Matrix<double, 6, 1> b;
		Eigen::Vector4d X;

		const int numItercounts = 5;

		Eigen::Vector4d beatsAlias(betas);
		for (int i=0;i<numItercounts;i++)
		{
			compute_A_and_b_gauss_newton(L6X10,rho6,beatsAlias,A,b);
			
			X = A.colPivHouseholderQr().solve(b);
			beatsAlias += X;
		}

		for (int i = 0; i < 4; i++)
			betas[i] = beatsAlias(i);
	}

	//----选取其中4个，[b11,b12,b13,b14]
	void PnPsolver::find_betas_approx_1(const Eigen::Matrix<double, 6, 10>& L6X10,
		const Eigen::Matrix<double, 6, 1>& rho6, double* betas)
	{
		Eigen::Matrix<double, 6, 4> L6X4;
		Eigen::Vector4d beat;
		L6X4 << L6X10.col(0), L6X10.col(1), L6X10.col(3), L6X10.col(6);
		beat=L6X4.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(rho6);

		// 计算得到beat的值
		if (beat(0) < 0)
		{
			double beat1 = std::sqrt(-beat(0));
			betas[0] = beat1;
			betas[1] = -beat(1) / beat1;
			betas[2] = -beat(2) / beat1;
			betas[3] = -beat(3) / beat1;
		}
		else
		{
			double beat1 = std::sqrt(beat(0));
			betas[0] = beat1;
			betas[1] = beat(1) / beat1;
			betas[2] = beat(2) / beat1;
			betas[3] = beat(3) / beat1;
		}
	}

	// ----[B11 B12 B22]
	void PnPsolver::find_betas_approx_2(const Eigen::Matrix<double, 6, 10>& L6X10,
		const Eigen::Matrix<double, 6, 1>& rho6, double* betas)
	{
		Eigen::Matrix<double, 6, 3> L6X3;
		Eigen::Vector3d beat;
		L6X3 << L6X10.col(0), L6X10.col(1), L6X10.col(3);
		beat=L6X3.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(rho6);

		if (beat(0)<0.0)
		{
			betas[0] = std::sqrt(-beat(0));
			betas[1] = beat(2) < 0 ? std::sqrt(-beat(1)) : 0.0;
		}
		else
		{
			betas[0] = std::sqrt(beat(0));
			betas[1] = beat(2) > 0 ? std::sqrt(beat(1)) : 0.0 ;
		}
		if (beat(1) < 0)
			betas[0] *= -1.0;

		betas[2] = betas[3] = 0.0;
	}

	//----[B11 B12 B22 B13 B23
	void PnPsolver::find_betas_approx_3(const Eigen::Matrix<double, 6, 10>& L6X10,
		const Eigen::Matrix<double, 6, 1>& rho6, double* betas)
	{
		Eigen::Matrix<double, 6, 5> L6x5;
		Eigen::Matrix<double, 5, 1> b5;
		L6x5 << L6X10.col(0), L6X10.col(1), L6X10.col(2), L6X10.col(3), L6X10.col(4);
		b5=L6x5.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(rho6);

		if (b5(0) < 0.0)
		{
			betas[0] = std::sqrt(-b5(0));
			betas[1] = b5(2) < 0 ? std::sqrt(-b5(1)) : 0.0;
		}
		else
		{
			betas[0] = std::sqrt(b5(0));
			betas[1] = b5(2) > 0 ? std::sqrt(b5(1)) : 0.0;
		}
		if (b5(1) < 0)
			betas[0] *= -1.0;

		betas[2] = b5(3) / betas[0];
		betas[3] = 0.0;
	}
}