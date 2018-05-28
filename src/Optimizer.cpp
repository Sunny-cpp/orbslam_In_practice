#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/types/types_sba.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"

namespace ORBSlam
{
	void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool *pbStopFlag,
		const unsigned long nLoopKF, const bool bRobust)
	{
		std::vector<KeyFrame*> vpKeyfs=pMap->GetKeyFrames();
		std::vector<MapPoint*> vpMappts = pMap->GetMapPoints();

		BundleAdjustment(vpKeyfs, vpMappts, nIterations, pbStopFlag, nLoopKF, bRobust);
	}

	void Optimizer::BundleAdjustment(const std::vector<KeyFrame*>& vpKF, const std::vector<MapPoint*>& vpMP,
		int nIterations , bool* pbStopFlag, const unsigned long,
		const bool bRobust)
	{
		std::vector<bool> vbNotIncludedMP(vpMP.size(), false);
		long unsigned int maxKFid;

		// ----step1: 做一个 SparseOptimizer
		g2o::SparseOptimizer optimizer;

		g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>* plinearSolver =
			new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
		g2o::BlockSolver_6_3* pblockSover = new g2o::BlockSolver_6_3(plinearSolver);
		g2o::OptimizationAlgorithmLevenberg* palgo = new g2o::OptimizationAlgorithmLevenberg(pblockSover);
		optimizer.setAlgorithm(palgo);

		if (pbStopFlag)
			optimizer.setForceStopFlag(pbStopFlag);

		//-------step2: 添加pose vertex
		for (int i = 0; i < vpKF.size(); i++)
		{
			if (vpKF[i]->IsBad())
				continue;

			g2o::SE3Quat pose(vpKF[i]->GetR(), vpKF[i]->Gettrans());
			g2o::VertexSE3Expmap* pVse3 = new g2o::VertexSE3Expmap();
			pVse3->setEstimate(pose);
			pVse3->setId(vpKF[i]->mlnId);
			if (vpKF[i]->mlnId == 0)
				pVse3->setFixed(true);

			optimizer.addVertex(pVse3);

			if (vpKF[i]->mlnId > maxKFid)
				maxKFid = vpKF[i]->mlnId;
		}

		const float thHuber2D = sqrt(5.99);

		  // 添加point vertex and edge
		for (int i=0;i<vpMP.size();i++)
		{
			MapPoint* pCurMp = vpMP[i];
			if (pCurMp->IsBad())
				continue;

			long unsigned int id = pCurMp->mlnId + maxKFid + 1;
			g2o::VertexSBAPointXYZ* pVpts=new g2o::VertexSBAPointXYZ();
			pVpts->setEstimate(pCurMp->Getpos());
			pVpts->setId(id);
			pVpts->setMarginalized(true);
			optimizer.addVertex(pVpts);

			int nedge = 0;
			//---添加edge
			std::map<KeyFrame*, int>& Observations= pCurMp->GetObservations();
			for (auto ite=Observations.begin();ite!=Observations.end();ite++)
			{
				KeyFrame* pKeyf = ite->first;
				if (pKeyf->IsBad() || pKeyf->mlnId>maxKFid)
					continue;

				nedge++;

				Eigen::Vector2d obs;
				obs << pKeyf->mvUnKeypts[ite->second].pt.x, pKeyf->mvUnKeypts[ite->second].pt.y;

				g2o::EdgeSE3ProjectXYZ* pcurEdge=new g2o::EdgeSE3ProjectXYZ();
				pcurEdge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));  //point
				pcurEdge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKeyf->mlnId)));  //pose
				pcurEdge->setMeasurement(obs);

				float invSigma2;   // 这个值后续再设定
				pcurEdge->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

				// set robust
				if (bRobust)
				{
					g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
					pcurEdge->setRobustKernel(rk);
					rk->setDelta(thHuber2D);
				}

				pcurEdge->fx = pKeyf->fx;
				pcurEdge->fy = pKeyf->fy;
				pcurEdge->cx = pKeyf->cx;
				pcurEdge->cy = pKeyf->cy;

				optimizer.addEdge(pcurEdge);
			}

			if (nedge == 0)
			{
				optimizer.removeVertex(pVpts);
				vbNotIncludedMP[i] = true;
			}
			else
				vbNotIncludedMP[i] = false;
		}

		optimizer.initializeOptimization();
		optimizer.optimize(nIterations);

		//给出优化的结果, <还没有做，跟local mapping 有关 2018-5-14>

	}

	int Optimizer::PoseOptimization(Frame* pFrame)
	{
		int ninitialCorres = 0;
		cv::Mat camk;
		pFrame->GetCameraPara(camk);
		float fx = camk.at<float>(0, 0);
		float fy = camk.at<float>(1, 1);
		float cx = camk.at<float>(0, 1);
		float cy = camk.at<float>(0, 2);


		//---step1:做优化器
		g2o::SparseOptimizer optimizer;
		g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>* plinearSolver =
			new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

		g2o::BlockSolver_6_3* pblockSolver = new g2o::BlockSolver_6_3(plinearSolver);
		g2o::OptimizationAlgorithmLevenberg* palgoLM = new g2o::OptimizationAlgorithmLevenberg(pblockSolver);
		optimizer.setAlgorithm(palgoLM);
		
		//---step2: 做pose顶点
		g2o::VertexSE3Expmap* pVSE3=new g2o::VertexSE3Expmap();
		pVSE3->setId(0);
		pVSE3->setFixed(false);
		g2o::SE3Quat q(pFrame->GetR(),pFrame->GetT());
		pVSE3->setEstimate(q);
		optimizer.addVertex(pVSE3);

		std::vector<MapPoint*>& vpmappts = pFrame->mvpMappts;
		const std::vector<cv::KeyPoint>& vkeyPts = pFrame->GetUnKeyPts();  // 原位直接取到
		int N = vpmappts.size();

		std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdges;
		std::vector<size_t> vnIndexEdge;
		vpEdges.reserve(N+20);
		vnIndexEdge.reserve(N+20);

		// ----step3: 做边
		float delta = 5.99;

		for (int i=0;i<N;i++)
		{
			if (vpmappts[i])
			{
				ninitialCorres++;

				pFrame->mvbOutlier[i] = false;

				Eigen::Vector2d obs;
				obs << vkeyPts[i].pt.x, vkeyPts[i].pt.y;

				g2o::EdgeSE3ProjectXYZOnlyPose* pe = new g2o::EdgeSE3ProjectXYZOnlyPose();
				pe->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
				pe->setMeasurement(obs);
				float invSigma;  // 这个值还没有设定
				pe->setInformation(Eigen::Matrix2d::Identity()*invSigma);

				//---set
				g2o::RobustKernelHuber* pKernelHuber = new g2o::RobustKernelHuber();
				pe->setRobustKernel(pKernelHuber);
				pKernelHuber->setDelta(delta);

				pe->fx = fx;
				pe->fy = fy;
				pe->cy = cy;
				pe->cx = cx;

				pe->Xw = vpmappts[i]->Getpos();
				optimizer.addEdge(pe);

				vnIndexEdge.push_back(i);
				vpEdges.push_back(pe);
			}
		}

		if (ninitialCorres < 3)
			return 0;

		// -------开始优化
		int nBads;
		const double chi2Table[4] = { 5.991,5.991,5.991,5.991 };
		// ---进行四次优化
		for (int it=0;it<4;it++)
		{
			optimizer.initializeOptimization(0);  // level 0层优化
			optimizer.optimize(10);

			nBads = 0;
			for (int i=0; i<vpEdges.size();i++)
			{
				g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdges[i];
				int ptIndex = vnIndexEdge[i];

				if (pFrame->mvbOutlier[ptIndex])
				{
					e->computeError();
				}

				double chi2 = e->chi2();
				if (chi2>chi2Table[it])
				{
					e->setLevel(1);
					pFrame->mvbOutlier[ptIndex] = true;
					nBads++;
				}
				else
				{
					e->setLevel(0);
					pFrame->mvbOutlier[ptIndex] = false;
				}

				if (it == 2)
					e->setRobustKernel(0);
			}
			if (optimizer.edges().size()<10)
				break;
		}

		g2o::VertexSE3Expmap* pVnew=dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
		g2o::SE3Quat SE3quat_New = pVnew->estimate();
		Eigen::Quaterniond q_New = SE3quat_New.rotation();
		q_New.normalize();

		Eigen::Matrix3d R_new(q_New.toRotationMatrix());
		Eigen::Vector3d t_new(SE3quat_New.translation());
		Eigen::Matrix4d Pose_new;
		Pose_new << R_new, t_new, Eigen::Vector3d::Zero().transpose(), 1;
		pFrame->SetPose(Pose_new);

		return ninitialCorres-nBads;
	}
}