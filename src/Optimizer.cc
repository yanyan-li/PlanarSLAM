#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include "Converter.h"

#include <mutex>

#include "g2oAddition/Plane3D.h"
#include "g2oAddition/EdgePlane.h"
#include "g2oAddition/EdgePlanePoint.h"
#include "g2oAddition/VertexPlane.h"
#include "g2oAddition/EdgeVerticalPlane.h"
#include "g2oAddition/EdgeParallelPlane.h"

#include <ctime>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;
using namespace g2o;

namespace Planar_SLAM {


    void Optimizer::GlobalBundleAdjustemnt(Map *pMap, int nIterations, bool *pbStopFlag,
                                           const unsigned long nLoopKF, const bool bRobust) {
        vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
        vector<MapPoint *> vpMP = pMap->GetAllMapPoints();
        vector<MapLine *> vpML = pMap->GetAllMapLines();
        vector<MapPlane *> vpMPL = pMap->GetAllMapPlanes();
        BundleAdjustment(vpKFs, vpMP, vpML, vpMPL, nIterations, pbStopFlag, nLoopKF, bRobust);
    }

    void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                     const vector<MapLine *> &vpML, const vector<MapPlane *> &vpMPL,
                                     int nIterations, bool *pbStopFlag,
                                     const unsigned long nLoopKF, const bool bRobust) {
        vector<bool> vbNotIncludedMP, vbNotIncludedML, vbNotIncludedMPL;
        vbNotIncludedMP.resize(vpMP.size());
        vbNotIncludedML.resize(vpML.size());
        vbNotIncludedMPL.resize(vpMPL.size());

        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        if (pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        long unsigned int maxKFid = 0;

        // Set KeyFrame vertices
        for (size_t i = 0; i < vpKFs.size(); i++) {
            KeyFrame *pKF = vpKFs[i];
            if (pKF->isBad())
                continue;
            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
            vSE3->setId(pKF->mnId);
            vSE3->setFixed(pKF->mnId == 0);
            optimizer.addVertex(vSE3);
            if (pKF->mnId > maxKFid)
                maxKFid = pKF->mnId;
        }

        const float thHuber2D = sqrt(5.99);
        const float thHuber3D = sqrt(7.815);

        vector<g2o::EdgeSE3ProjectXYZ *> vpEdgesMono;
        vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;

        long unsigned int maxMapPointId = maxKFid;

        int totalEdges = 0;
        // Set MapPoint vertices
        for (size_t i = 0; i < vpMP.size(); i++) {
            MapPoint *pMP = vpMP[i];

            if (pMP->isBad())
                continue;

            g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            const int id = pMP->mnId + maxKFid + 1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            if (id > maxMapPointId) {
                maxMapPointId = id;
            }

            const map<KeyFrame *, size_t> observations = pMP->GetObservations();

            int nEdges = 0;
            //SET EDGES
            for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++) {

                KeyFrame *pKF = mit->first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                nEdges++;

                const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

                if (pKF->mvuRight[mit->second] < 0) {
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    if (bRobust) {
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber2D);
                    }

                    e->fx = pKF->fx;
                    e->fy = pKF->fy;
                    e->cx = pKF->cx;
                    e->cy = pKF->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                } else {
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_ur = pKF->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    if (bRobust) {
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber3D);
                    }

                    e->fx = pKF->fx;
                    e->fy = pKF->fy;
                    e->cx = pKF->cx;
                    e->cy = pKF->cy;
                    e->bf = pKF->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                }
            }

            totalEdges += nEdges;

            if (nEdges == 0) {
                optimizer.removeVertex(vPoint);
                vbNotIncludedMP[i] = true;
            } else {
                vbNotIncludedMP[i] = false;
            }
        }

        cout << "GBA: Total point edges: " << totalEdges << endl;
        cout << "GBA: Max Point id: " << maxMapPointId << endl;

        int maxMapLineId = maxMapPointId;

        for (size_t i = 0; i < vpML.size(); i++) {
            MapLine *pML = vpML[i];

            if (pML->isBad())
                continue;

            g2o::VertexSBAPointXYZ *vStartPoint = new g2o::VertexSBAPointXYZ();
            vStartPoint->setEstimate(pML->GetWorldPos().head(3));
            const int id1 = (2 * pML->mnId) + 1 + maxMapPointId;
            vStartPoint->setId(id1);
            vStartPoint->setMarginalized(true);
            optimizer.addVertex(vStartPoint);

            g2o::VertexSBAPointXYZ *vEndPoint = new g2o::VertexSBAPointXYZ();
            vEndPoint->setEstimate(pML->GetWorldPos().tail(3));
            const int id2 = (2 * (pML->mnId + 1)) + maxMapPointId;
            vEndPoint->setId(id2);
            vEndPoint->setMarginalized(true);
            optimizer.addVertex(vEndPoint);

            if (id2 > maxMapLineId) {
                maxMapLineId = id2;
            }

            cout << "GBA: Line id1: " << id1 << ", id2: " << id2 << ", Max: " << maxMapLineId << endl;

            const map<KeyFrame *, size_t> observations = pML->GetObservations();

            int nEdges = 0;

            for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++) {
                KeyFrame *pKF = mit->first;

                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                nEdges++;

                Eigen::Vector3d lineObs = pKF->mvKeyLineFunctions[mit->second];

                EdgeLineProjectXYZ *es = new EdgeLineProjectXYZ();
                es->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
                es->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                es->setMeasurement(lineObs);
                es->setInformation(Eigen::Matrix3d::Identity());

                if (bRobust) {
                    g2o::RobustKernelHuber *rks = new g2o::RobustKernelHuber;
                    es->setRobustKernel(rks);
                    rks->setDelta(thHuber3D);
                }

                es->fx = pKF->fx;
                es->fy = pKF->fy;
                es->cx = pKF->cx;
                es->cy = pKF->cy;

                optimizer.addEdge(es);

                EdgeLineProjectXYZ *ee = new EdgeLineProjectXYZ();
                ee->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
                ee->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                ee->setMeasurement(lineObs);
                ee->setInformation(Eigen::Matrix3d::Identity());

                if (bRobust) {
                    g2o::RobustKernelHuber *rke = new g2o::RobustKernelHuber;
                    ee->setRobustKernel(rke);
                    rke->setDelta(thHuber3D);
                }

                ee->fx = pKF->fx;
                ee->fy = pKF->fy;
                ee->cx = pKF->cx;
                ee->cy = pKF->cy;

                optimizer.addEdge(ee);
            }

            if (nEdges == 0) {
                optimizer.removeVertex(vStartPoint);
                optimizer.removeVertex(vEndPoint);
                vbNotIncludedML[i] = true;
            } else {
                vbNotIncludedML[i] = false;
            }
        }

        double angleInfo = Config::Get<double>("Plane.AngleInfo");
        angleInfo = 3282.8 / (angleInfo * angleInfo);
        double disInfo = Config::Get<double>("Plane.DistanceInfo");
        disInfo = disInfo * disInfo;
        double parInfo = Config::Get<double>("Plane.ParallelInfo");
        parInfo = 3282.8 / (parInfo * parInfo);
        double verInfo = Config::Get<double>("Plane.VerticalInfo");
        verInfo = 3282.8 / (verInfo * verInfo);
        double planeChi = Config::Get<double>("Plane.Chi");
        const float deltaPlane = sqrt(planeChi);

        double VPplaneChi = Config::Get<double>("Plane.VPChi");
        const float VPdeltaPlane = sqrt(VPplaneChi);

        // Set MapPlane vertices
        for (size_t i = 0; i < vpMPL.size(); i++) {
            MapPlane *pMP = vpMPL[i];
            if (pMP->isBad())
                continue;

            g2o::VertexPlane *vPlane = new g2o::VertexPlane();
            vPlane->setEstimate(Converter::toPlane3D(pMP->GetWorldPos()));
            const int id = pMP->mnId + maxMapLineId + 1;
            vPlane->setId(id);
            vPlane->setMarginalized(true);
            optimizer.addVertex(vPlane);

            cout << "GBA: Plane id: " << id << endl;

            int nEdges = 0;

            const map<KeyFrame *, size_t> observations = pMP->GetObservations();
            for (const auto &observation : observations) {

                KeyFrame *pKF = observation.first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                nEdges++;

                g2o::EdgePlane *e = new g2o::EdgePlane();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setMeasurement(Converter::toPlane3D(pKF->mvPlaneCoefficients[observation.second]));
                //TODO
                Eigen::Matrix3d Info;
                Info << angleInfo, 0, 0,
                        0, angleInfo, 0,
                        0, 0, disInfo;
                e->setInformation(Info);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaPlane);

                e->planePoints = pMP->mvPlanePoints;

                optimizer.addEdge(e);
            }

            const map<KeyFrame *, size_t> verObservations = pMP->GetVerObservations();
            for (const auto &observation : verObservations) {

                KeyFrame *pKF = observation.first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                nEdges++;

                g2o::EdgeVerticalPlane *e = new g2o::EdgeVerticalPlane();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setMeasurement(Converter::toPlane3D(pKF->mvPlaneCoefficients[observation.second]));
                //TODO
                Eigen::Matrix2d Info;
                Info << angleInfo, 0,
                        0, angleInfo;
                e->setInformation(Info);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(VPdeltaPlane);

                optimizer.addEdge(e);
            }

            const map<KeyFrame *, size_t> parObservations = pMP->GetParObservations();
            for (const auto &observation : parObservations) {

                KeyFrame *pKF = observation.first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                nEdges++;

                g2o::EdgeParallelPlane *e = new g2o::EdgeParallelPlane();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setMeasurement(Converter::toPlane3D(pKF->mvPlaneCoefficients[observation.second]));
                //TODO
                Eigen::Matrix2d Info;
                Info << angleInfo, 0,
                        0, angleInfo;
                e->setInformation(Info);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(VPdeltaPlane);

                optimizer.addEdge(e);
            }

            if (nEdges == 0) {
                optimizer.removeVertex(vPlane);
                vbNotIncludedMPL[i] = true;
            } else {
                vbNotIncludedMPL[i] = false;
            }
        }

        // Optimize!
        optimizer.initializeOptimization();
        optimizer.optimize(nIterations);

        int bad = 0;
        int PNMono = 0;
        double PEMono = 0;
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
            g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];


            const float chi2 = e->chi2();
            //cout<<"optimize chi2"<<chi2<<endl;
            PNMono++;
            PEMono += chi2;

            if (chi2 > thHuber2D * thHuber2D) {
                bad++;
                cout << " GBA: Bad point: " << chi2 << endl;
            }
        }

        if (PNMono == 0)
            cout << "GBA: No mono points " << " ";
        else
            cout << "GBA: Mono points: " << PEMono / PNMono << " ";

        int PNStereo = 0;
        double PEStereo = 0;
        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
            g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];

            const float chi2 = e->chi2();
            //cout<<"optimize chi2"<<chi2<<endl;
            PNStereo++;
            PEStereo += chi2;

            if (chi2 > thHuber3D * thHuber3D) {
                bad++;
                cout << "GBA: Bad stereo point: " << chi2 << endl;
            }
        }
        if (PNStereo == 0)
            cout << "GBA: No stereo points " << " ";
        else
            cout << "GBA: Stereo points: " << PEStereo / PNStereo << endl;

        cout << "GBA: Total bad point edges: " << bad << endl;

        // Recover optimized data

        //Keyframes
        for (size_t i = 0; i < vpKFs.size(); i++) {
            KeyFrame *pKF = vpKFs[i];
            if (pKF->isBad())
                continue;
            g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            if (nLoopKF == 0) {
                pKF->SetPose(Converter::toCvMat(SE3quat));
            } else {
                pKF->mTcwGBA.create(4, 4, CV_32F);
                Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
                pKF->mnBAGlobalForKF = nLoopKF;
            }
        }

        //Points
        for (size_t i = 0; i < vpMP.size(); i++) {
            if (vbNotIncludedMP[i])
                continue;

            MapPoint *pMP = vpMP[i];

            if (pMP->isBad())
                continue;

            g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));

            double dis = cv::norm(Converter::toCvMat(vPoint->estimate()) - pMP->GetWorldPos());
            if (dis > 0.5) {
                std::cout << "Point id: " << pMP->mnId << ", bad: " << pMP->isBad()  << ", pose - before: " << pMP->GetWorldPos().t()
                          << ", after: " << Converter::toCvMat(vPoint->estimate()).t() << std::endl;
            }

            if (nLoopKF == 0) {
                pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
                pMP->UpdateNormalAndDepth();
            } else {
                pMP->mPosGBA.create(3, 1, CV_32F);
                Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
                pMP->mnBAGlobalForKF = nLoopKF;
            }
        }

        //Lines
        for (size_t i = 0; i < vpML.size(); i++) {

            if (vbNotIncludedML[i])
                continue;

            MapLine *pML = vpML[i];

            if (pML->isBad())
                continue;

            g2o::VertexSBAPointXYZ *vStartPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(
                    (2 * pML->mnId) + 1 + maxMapPointId));
            g2o::VertexSBAPointXYZ *vEndPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(
                    2 * (pML->mnId + 1) + maxMapPointId));

            if (nLoopKF == 0) {
                Vector6d linePos;
                linePos << vStartPoint->estimate(), vEndPoint->estimate();
                pML->SetWorldPos(linePos);
                pML->UpdateAverageDir();
            } else {
                pML->mPosGBA.create(6, 1, CV_32F);
                Converter::toCvMat(vStartPoint->estimate()).copyTo(pML->mPosGBA.rowRange(0, 3));
                Converter::toCvMat(vEndPoint->estimate()).copyTo(pML->mPosGBA.rowRange(3, 6));
                pML->mnBAGlobalForKF = nLoopKF;
            }
        }

        //Planes
        for (size_t i = 0; i < vpMPL.size(); i++) {
            if (vbNotIncludedMPL[i])
                continue;

            MapPlane *pMP = vpMPL[i];

            if (pMP->isBad())
                continue;

            g2o::VertexPlane *vPlane = static_cast<g2o::VertexPlane *>(optimizer.vertex(
                    pMP->mnId + maxMapLineId + 1));

            if (nLoopKF == 0) {
                pMP->SetWorldPos(Converter::toCvMat(vPlane->estimate()));
                pMP->UpdateCoefficientsAndPoints();
            } else {
                pMP->mPosGBA.create(4, 1, CV_32F);
                Converter::toCvMat(vPlane->estimate()).copyTo(pMP->mPosGBA);
                pMP->mnBAGlobalForKF = nLoopKF;
            }
        }
    }

    int Optimizer::PoseOptimization(Frame *pFrame) {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        int nInitialCorrespondences = 0;

        // Set Frame vertex
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // Set MapPoint vertices
        const int N = pFrame->N;

        vector<g2o::EdgeSE3ProjectXYZOnlyPose *> vpEdgesMono;
        vector<size_t> vnIndexEdgeMono;
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);

        vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
        vector<size_t> vnIndexEdgeStereo;
        vpEdgesStereo.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float deltaMono = sqrt(5.991);
        const float deltaStereo = sqrt(7.815);

        vector<double> vMonoPointInfo(N, 1);
        vector<double> vSteroPointInfo(N, 1);

        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);


            for (int i = 0; i < N; i++) {
                MapPoint *pMP = pFrame->mvpMapPoints[i];
                if (pMP) {
                    // Monocular observation
                    if (pFrame->mvuRight[i] < 0) {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        Eigen::Matrix<double, 2, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        obs << kpUn.pt.x, kpUn.pt.y;

                        g2o::EdgeSE3ProjectXYZOnlyPose *e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaMono);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        cv::Mat Xw = pMP->GetWorldPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    } else  // Stereo observation
                    {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        //SET EDGE
                        Eigen::Matrix<double, 3, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        const float &kp_ur = pFrame->mvuRight[i];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                        e->setInformation(Info);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaStereo);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        e->bf = pFrame->mbf;
                        cv::Mat Xw = pMP->GetWorldPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesStereo.push_back(e);
                        vnIndexEdgeStereo.push_back(i);
                    }
                }

            }
        }

        const int NL = pFrame->NL;

        vector<EdgeLineProjectXYZOnlyPose *> vpEdgesLineSp;
        vector<size_t> vnIndexLineEdgeSp;
        vpEdgesLineSp.reserve(NL);
        vnIndexLineEdgeSp.reserve(NL);

        vector<EdgeLineProjectXYZOnlyPose *> vpEdgesLineEp;
        vector<size_t> vnIndexLineEdgeEp;
        vpEdgesLineEp.reserve(NL);
        vnIndexLineEdgeEp.reserve(NL);

        vector<double> vMonoStartPointInfo(NL, 1);
        vector<double> vMonoEndPointInfo(NL, 1);
        vector<double> vSteroStartPointInfo(NL, 1);
        vector<double> vSteroEndPointInfo(NL, 1);

        // Set MapLine vertices
        {
            unique_lock<mutex> lock(MapLine::mGlobalMutex);

            for (int i = 0; i < NL; i++) {
                MapLine *pML = pFrame->mvpMapLines[i];
                if (pML) {
                    nInitialCorrespondences++;
                    pFrame->mvbLineOutlier[i] = false;

                    Eigen::Vector3d line_obs;
                    line_obs = pFrame->mvKeyLineFunctions[i];

                    EdgeLineProjectXYZOnlyPose *els = new EdgeLineProjectXYZOnlyPose();

                    els->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    els->setMeasurement(line_obs);
                    els->setInformation(Eigen::Matrix3d::Identity());

                    g2o::RobustKernelHuber *rk_line_s = new g2o::RobustKernelHuber;
                    els->setRobustKernel(rk_line_s);
                    rk_line_s->setDelta(deltaStereo);

                    els->fx = pFrame->fx;
                    els->fy = pFrame->fy;
                    els->cx = pFrame->cx;
                    els->cy = pFrame->cy;

                    els->Xw = pML->mWorldPos.head(3);
                    optimizer.addEdge(els);

                    vpEdgesLineSp.push_back(els);
                    vnIndexLineEdgeSp.push_back(i);

                    EdgeLineProjectXYZOnlyPose *ele = new EdgeLineProjectXYZOnlyPose();

                    ele->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    ele->setMeasurement(line_obs);
                    ele->setInformation(Eigen::Matrix3d::Identity());

                    g2o::RobustKernelHuber *rk_line_e = new g2o::RobustKernelHuber;
                    ele->setRobustKernel(rk_line_e);
                    rk_line_e->setDelta(deltaStereo);

                    ele->fx = pFrame->fx;
                    ele->fy = pFrame->fy;
                    ele->cx = pFrame->cx;
                    ele->cy = pFrame->cy;

                    ele->Xw = pML->mWorldPos.tail(3);

                    optimizer.addEdge(ele);

                    vpEdgesLineEp.push_back(ele);
                    vnIndexLineEdgeEp.push_back(i);
                }
            }
        }

        //Set Plane vertices
        const int M = pFrame->mnPlaneNum;

        vector<g2o::EdgePlaneOnlyPose *> vpEdgesPlane;
        vector<size_t> vnIndexEdgePlane;
        vpEdgesPlane.reserve(M);
        vnIndexEdgePlane.reserve(M);

        vector<vector<g2o::EdgePlanePoint *>> vpEdgesPlanePoint;
        vector<vector<size_t>> vnIndexEdgePlanePoint;
        vpEdgesPlanePoint = vector<vector<g2o::EdgePlanePoint *>>(M);
        vnIndexEdgePlanePoint = vector<vector<size_t>>(M);

        vector<g2o::EdgeParallelPlaneOnlyPose *> vpEdgesParPlane;
        vector<size_t> vnIndexEdgeParPlane;
        vpEdgesParPlane.reserve(M);
        vnIndexEdgeParPlane.reserve(M);

        vector<g2o::EdgeVerticalPlaneOnlyPose *> vpEdgesVerPlane;
        vector<size_t> vnIndexEdgeVerPlane;
        vpEdgesVerPlane.reserve(M);
        vnIndexEdgeVerPlane.reserve(M);

        double angleInfo = Config::Get<double>("Plane.AngleInfo");
        angleInfo = 3282.8 / (angleInfo * angleInfo);
        double disInfo = Config::Get<double>("Plane.DistanceInfo");
        disInfo = disInfo * disInfo;
        double parInfo = Config::Get<double>("Plane.ParallelInfo");
        parInfo = 3282.8 / (parInfo * parInfo);
        double verInfo = Config::Get<double>("Plane.VerticalInfo");
        verInfo = 3282.8 / (verInfo * verInfo);
        double planeChi = Config::Get<double>("Plane.Chi");
        const float deltaPlane = sqrt(planeChi);

        double VPplaneChi = Config::Get<double>("Plane.VPChi");
        const float VPdeltaPlane = sqrt(VPplaneChi);

        {
            unique_lock<mutex> lock(MapPlane::mGlobalMutex);
            int PNum = 0;
            double PEror = 0, PMax = 0;
            for (int i = 0; i < M; ++i) {
                MapPlane *pMP = pFrame->mvpMapPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbPlaneOutlier[i] = false;

                    g2o::EdgePlaneOnlyPose *e = new g2o::EdgePlaneOnlyPose();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    //TODO
                    Eigen::Matrix3d Info;
                    Info << angleInfo, 0, 0,
                            0, angleInfo, 0,
                            0, 0, disInfo;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    //TODO
                    rk->setDelta(deltaPlane);

                    e->Xw = Converter::toPlane3D(pMP->GetWorldPos());
                    e->planePoints = pMP->mvPlanePoints;

                    optimizer.addEdge(e);

                    vpEdgesPlane.push_back(e);
                    vnIndexEdgePlane.push_back(i);

//                    int nPointMatches = pFrame->mvPlanePointMatches[i].size();
//
//                    vector<g2o::EdgePlanePoint*> edgesPlanePoint;
//                    vector<size_t> indexEdgePlanePoint;
//                    for (int j = 0; j < nPointMatches; j++) {
//                        MapPoint *mapPoint = pFrame->mvPlanePointMatches[i][j];
//                        if (mapPoint) {
//                            g2o::EdgePlanePoint *edge = new g2o::EdgePlanePoint();
//                            edge->setVertex(0, optimizer.vertex(0));
//                            edge->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
//                            edge->setInformation(Eigen::Matrix3d::Identity() * 1);
//
//                            cv::Mat Pw = mapPoint->GetWorldPos();
//                            edge->Xw[0] = Pw.at<float>(0);
//                            edge->Xw[1] = Pw.at<float>(1);
//                            edge->Xw[2] = Pw.at<float>(2);
//
//                            g2o::RobustKernelHuber *rkEdge = new g2o::RobustKernelHuber;
//                            edge->setRobustKernel(rkEdge);
//                            rkEdge->setDelta(deltaMono);
//
//                            optimizer.addEdge(edge);
//
//                            edgesPlanePoint.push_back(edge);
//                            indexEdgePlanePoint.push_back(j);
//                        }
//                    }
//
//                    int pointEdges = edgesPlanePoint.size();
//                    int nLineMatches = pFrame->mvPlaneLineMatches[i].size();
//
//                    for (int j = 0, index = pointEdges; j < nLineMatches; j++) {
//                        MapLine *mapLine = pFrame->mvPlaneLineMatches[i][j];
//                        if (mapLine) {
//                            g2o::EdgePlanePoint *edgeStart = new g2o::EdgePlanePoint();
//                            edgeStart->setVertex(0, optimizer.vertex(0));
//                            edgeStart->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
//                            edgeStart->setInformation(Eigen::Matrix3d::Identity() * 1);
//
//                            Vector3d startPoint = mapLine->mWorldPos.head(3);
//                            edgeStart->Xw[0] = startPoint(0);
//                            edgeStart->Xw[1] = startPoint(1);
//                            edgeStart->Xw[2] = startPoint(2);
//
//                            g2o::RobustKernelHuber *rkEdgeStart = new g2o::RobustKernelHuber;
//                            edgeStart->setRobustKernel(rkEdgeStart);
//                            rkEdgeStart->setDelta(deltaMono);
//
//                            optimizer.addEdge(edgeStart);
//
//                            edgesPlanePoint.push_back(edgeStart);
//                            indexEdgePlanePoint.push_back(index++);
//
//                            g2o::EdgePlanePoint *edgeEnd = new g2o::EdgePlanePoint();
//                            edgeEnd->setVertex(0, optimizer.vertex(0));
//                            edgeEnd->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
//                            edgeEnd->setInformation(Eigen::Matrix3d::Identity() * 1);
//
//                            Vector3d endPoint = mapLine->mWorldPos.tail(3);
//                            edgeEnd->Xw[0] = endPoint(0);
//                            edgeEnd->Xw[1] = endPoint(1);
//                            edgeEnd->Xw[2] = endPoint(2);
//
//                            g2o::RobustKernelHuber *rkEdgeEnd = new g2o::RobustKernelHuber;
//                            edgeEnd->setRobustKernel(rkEdgeEnd);
//                            rkEdgeEnd->setDelta(deltaMono);
//
//                            optimizer.addEdge(edgeEnd);
//
//                            edgesPlanePoint.push_back(edgeEnd);
//                            indexEdgePlanePoint.push_back(index++);
//                        }
//                    }
//
//                    vpEdgesPlanePoint[i] = edgesPlanePoint;
//                    vnIndexEdgePlanePoint[i] = indexEdgePlanePoint;


                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            //cout << " Plane: " << PEror / PNum << " ";

            PNum = 0;
            PEror = 0;
            PMax = 0;
            for (int i = 0; i < M; ++i) {
                // add parallel planes!
                MapPlane *pMP = pFrame->mvpParallelPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbParPlaneOutlier[i] = false;

                    g2o::EdgeParallelPlaneOnlyPose *e = new g2o::EdgeParallelPlaneOnlyPose();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    Eigen::Matrix2d Info;
                    Info << parInfo, 0,
                            0, parInfo;

                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(VPdeltaPlane);

                    e->Xw = Converter::toPlane3D(pMP->GetWorldPos());

                    optimizer.addEdge(e);

                    vpEdgesParPlane.push_back(e);
                    vnIndexEdgeParPlane.push_back(i);

                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            //cout << " Par Plane: " << PEror / PNum << " ";

            PNum = 0;
            PEror = 0;
            PMax = 0;

            for (int i = 0; i < M; ++i) {
                // add vertical planes!
                MapPlane *pMP = pFrame->mvpVerticalPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbVerPlaneOutlier[i] = false;

                    g2o::EdgeVerticalPlaneOnlyPose *e = new g2o::EdgeVerticalPlaneOnlyPose();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    Eigen::Matrix2d Info;
                    Info << verInfo, 0,
                            0, verInfo;

                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(VPdeltaPlane);

                    e->Xw = Converter::toPlane3D(pMP->GetWorldPos());

                    optimizer.addEdge(e);

                    vpEdgesVerPlane.push_back(e);
                    vnIndexEdgeVerPlane.push_back(i);

                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            //cout << " Ver Plane: " << PEror / PNum << endl;
        }

        if (nInitialCorrespondences < 3)
            return 0;

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
        const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;

        for (size_t it = 0; it < 4; it++) {

            vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad = 0;

            int PNMono = 0;
            double PEMono = 0, PMaxMono = 0;
            for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
                g2o::EdgeSE3ProjectXYZOnlyPose *e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                if (pFrame->mvbOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                //cout<<"optimize chi2"<<chi2<<endl;
                PNMono++;
                PEMono += chi2;
                PMaxMono = PMaxMono > chi2 ? PMaxMono : chi2;

                if (chi2 > chi2Mono[it]) {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    pFrame->mvbOutlier[idx] = false;
                    vMonoPointInfo[i] = 1.0 / sqrt(chi2);
                    e->setLevel(0);
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

//            if (PNMono == 0)
//                cout << "No mono points " << " ";
//            else
//                cout << " Mono points: " << PEMono / PNMono << " ";

            int PNStereo = 0;
            double PEStereo = 0, PMaxStereo = 0;
            for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
                g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if (pFrame->mvbOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                //cout<<"optimize chi2"<<chi2<<endl;
                PNStereo++;
                PEStereo += chi2;
                PMaxStereo = PMaxStereo > chi2 ? PMaxStereo : chi2;

                if (chi2 > chi2Stereo[it]) {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbOutlier[idx] = false;
                    vSteroPointInfo[i] = 1.0 / sqrt(chi2);
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }
//            if (PNStereo == 0)
//                cout << "No stereo points " << " ";
//            else
//                cout << " Stereo points: " << PEStereo / PNStereo << endl;

            int PNLine = 0;
            double PELine = 0, PMaxLine = 0;
            for (size_t i = 0, iend = vpEdgesLineSp.size(); i < iend; i++) {
                EdgeLineProjectXYZOnlyPose *e1 = vpEdgesLineSp[i];  //线段起始点
                EdgeLineProjectXYZOnlyPose *e2 = vpEdgesLineEp[i];  //线段终止点

                const size_t idx = vnIndexLineEdgeSp[i];    //线段起始点和终止点的误差边的index一样

                if (pFrame->mvbLineOutlier[idx]) {
                    e1->computeError();
                    e2->computeError();
                }
                e1->computeError();
                e2->computeError();

                const float chi2_s = e1->chiline();//e1->chi2();
                const float chi2_e = e2->chiline();//e2->chi2();
//                cout<<"Optimization: chi2_s "<<chi2_s<<", chi2_e "<<chi2_e<<endl;

                PNLine++;
                PELine += chi2_s + chi2_e;
                PMaxLine = PMaxLine > chi2_s + chi2_e ? PMaxLine : chi2_s + chi2_e;


                if (chi2_s > 2 * chi2Mono[it] || chi2_e > 2 * chi2Mono[it]) {
                    pFrame->mvbLineOutlier[idx] = true;
                    e1->setLevel(1);
                    e2->setLevel(1);
                    nBad++;
                } else {
                    pFrame->mvbLineOutlier[idx] = false;
                    e1->setLevel(0);
                    e2->setLevel(0);
                    vSteroEndPointInfo[i] = 1.0 / sqrt(chi2_e);
                    vSteroStartPointInfo[i] = 1.0 / sqrt(chi2_s);
                }

                if (it == 2) {
                    e1->setRobustKernel(0);
                    e2->setRobustKernel(0);
                }
            }

//            if (PNLine == 0)
//                cout << "No lines " << " ";
//            else
//                cout << " Lines: " << PELine / PNLine << endl;

            int PN = 0;
            double PE = 0, PMax = 0;

            for (size_t i = 0, iend = vpEdgesPlane.size(); i < iend; i++) {
                g2o::EdgePlaneOnlyPose *e = vpEdgesPlane[i];

                const size_t idx = vnIndexEdgePlane[i];

                if (pFrame->mvbPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > planeChi) {
                    pFrame->mvbPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
//                    cout << "bad: " << chi2 << ", id: " << idx << "  Pc : " << pFrame->ComputePlaneWorldCoeff(idx).t()
//                         << "  Pw :" << pFrame->mvpMapPlanes[idx]->GetWorldPos().t() << endl;
                } else {
                    e->setLevel(0);
                    pFrame->mvbPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);

//                if (vpEdgesPlanePoint[i].size() > 0) {
//                    int PPN = 0;
//                    double PPE = 0, PPMax = 0;
//                    for (size_t j = 0, jend = vpEdgesPlanePoint[i].size(); j < jend; j++) {
//                        g2o::EdgePlanePoint *edge = vpEdgesPlanePoint[i][j];
//
//                        const size_t index = vnIndexEdgePlanePoint[i][j];
//
//                        const float chi2 = edge->chi2();
////                    cout<<"optimize chi2"<<chi2<<endl;
//                        PPN++;
//                        PPE += chi2;
//                        PPMax = PPMax > chi2 ? PPMax : chi2;
//
//                        if (chi2 > chi2Mono[it]) {
//                            edge->setLevel(1);
//                            nBad++;
//                        } else {
//                            edge->setLevel(0);
//                        }
//
//                        if (it == 2)
//                            edge->setRobustKernel(0);
//                    }
//
//                    if (PPN == 0)
//                        cout << "planetest No plane point matches " << " ";
//                    else
//                        cout << "planetest  Plane point matches: " << PPE / PPN << " "; //<< " Max: " << PMax << endl;
//                }
            }
//            if (PN == 0)
//                cout << "No plane " << " ";
//            else
//                cout << " Plane: " << PE / PN << " "; //<< " Max: " << PMax << endl;

            PN = 0;
            PE = 0;
            PMax = 0;
            for (size_t i = 0, iend = vpEdgesParPlane.size(); i < iend; i++) {
                g2o::EdgeParallelPlaneOnlyPose *e = vpEdgesParPlane[i];

                const size_t idx = vnIndexEdgeParPlane[i];

                if (pFrame->mvbParPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > VPplaneChi) {
                    pFrame->mvbParPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
//                    cout << "bad Par: " << chi2 << ", id: " << idx << "  Pc : "
//                         << pFrame->ComputePlaneWorldCoeff(idx).t() << "  Pw :"
//                         << pFrame->mvpParallelPlanes[idx]->GetWorldPos().t() << endl;
                } else {
                    e->setLevel(0);
                    pFrame->mvbParPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }
//            if (PN == 0)
//                cout << "No par plane " << " ";
//            else
//                cout << "par Plane: " << PE / PN << " "; //<< " Max: " << PMax << endl;

            PN = 0;
            PE = 0;
            PMax = 0;

            for (size_t i = 0, iend = vpEdgesVerPlane.size(); i < iend; i++) {
                g2o::EdgeVerticalPlaneOnlyPose *e = vpEdgesVerPlane[i];

                const size_t idx = vnIndexEdgeVerPlane[i];

                if (pFrame->mvbVerPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > VPplaneChi) {
                    pFrame->mvbVerPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
//                    cout << "bad Ver: " << chi2 << ", id: " << idx << "  Pc : "
//                         << pFrame->ComputePlaneWorldCoeff(idx).t() << "  Pw :"
//                         << pFrame->mvpVerticalPlanes[idx]->GetWorldPos().t() << endl;
                } else {
                    e->setLevel(0);
                    pFrame->mvbVerPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }
//            if (PN == 0)
//                cout << "No Ver plane " << endl;
//            else
//                cout << "Ver Plane: " << PE / PN << endl; //<< " Max: " << PMax << endl;

            if (optimizer.edges().size() < 10)
                break;
        }

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        pFrame->SetPose(Converter::toCvMat(SE3quat_recov));

        return nInitialCorrespondences - nBad;
    }


    int Optimizer::PoseOptimizationPointsOnly(Frame *pFrame) {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        int nInitialCorrespondences = 0;

        // Set Frame vertex
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // Set MapPoint vertices
        const int N = pFrame->N;

        vector<g2o::EdgeSE3ProjectXYZOnlyPose *> vpEdgesMono;
        vector<size_t> vnIndexEdgeMono;
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);

        vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
        vector<size_t> vnIndexEdgeStereo;
        vpEdgesStereo.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float deltaMono = sqrt(5.991);
        const float deltaStereo = sqrt(7.815);


        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);

            for (int i = 0; i < N; i++) {
                MapPoint *pMP = pFrame->mvpMapPoints[i];
                if (pMP) {
                    // Monocular observation
                    if (pFrame->mvuRight[i] < 0) {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        Eigen::Matrix<double, 2, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        obs << kpUn.pt.x, kpUn.pt.y;

                        g2o::EdgeSE3ProjectXYZOnlyPose *e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaMono);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        cv::Mat Xw = pMP->GetWorldPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    } else  // Stereo observation
                    {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        //SET EDGE
                        Eigen::Matrix<double, 3, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        const float &kp_ur = pFrame->mvuRight[i];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                        e->setInformation(Info);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaStereo);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        e->bf = pFrame->mbf;
                        cv::Mat Xw = pMP->GetWorldPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesStereo.push_back(e);
                        vnIndexEdgeStereo.push_back(i);
                    }
                }

            }
        }

        //Set Plane vertices
        const int M = pFrame->mnPlaneNum;
        vector<g2o::EdgePlaneOnlyPose *> vpEdgesPlane;
        vector<size_t> vnIndexEdgePlane;
        vpEdgesPlane.reserve(M);
        vnIndexEdgePlane.reserve(M);

        vector<vector<g2o::EdgePlanePoint *>> vpEdgesPlanePoint;
        vector<vector<size_t>> vnIndexEdgePlanePoint;
        vpEdgesPlanePoint = vector<vector<g2o::EdgePlanePoint *>>(M);
        vnIndexEdgePlanePoint = vector<vector<size_t>>(M);

        vector<g2o::EdgeParallelPlaneOnlyPose *> vpEdgesParPlane;
        vector<size_t> vnIndexEdgeParPlane;
        vpEdgesParPlane.reserve(M);
        vnIndexEdgeParPlane.reserve(M);

        vector<g2o::EdgeVerticalPlaneOnlyPose *> vpEdgesVerPlane;
        vector<size_t> vnIndexEdgeVerPlane;
        vpEdgesVerPlane.reserve(M);
        vnIndexEdgeVerPlane.reserve(M);

        double angleInfo = Config::Get<double>("Plane.AngleInfo");
        angleInfo = 3282.8 / (angleInfo * angleInfo);
        double disInfo = Config::Get<double>("Plane.DistanceInfo");
        disInfo = disInfo * disInfo;
        double parInfo = Config::Get<double>("Plane.ParallelInfo");
        parInfo = 3282.8 / (parInfo * parInfo);
        double verInfo = Config::Get<double>("Plane.VerticalInfo");
        verInfo = 3282.8 / (verInfo * verInfo);
        double planeChi = Config::Get<double>("Plane.Chi");
        const float deltaPlane = sqrt(planeChi);

        double VPplaneChi = Config::Get<double>("Plane.VPChi");
        const float VPdeltaPlane = sqrt(VPplaneChi);

        {
            unique_lock<mutex> lock(MapPlane::mGlobalMutex);
            int PNum = 0;
            double PEror = 0, PMax = 0;
            unsigned long maxPlaneid = 0;
            for (int i = 0; i < M; ++i) {
                MapPlane *pMP = pFrame->mvpMapPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbPlaneOutlier[i] = false;

                    g2o::EdgePlaneOnlyPose *e = new g2o::EdgePlaneOnlyPose();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    //TODO
                    Eigen::Matrix3d Info;
                    Info << angleInfo, 0, 0,
                            0, angleInfo, 0,
                            0, 0, disInfo;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    //TODO
                    rk->setDelta(deltaPlane);

                    e->Xw = Converter::toPlane3D(pMP->GetWorldPos());

                    optimizer.addEdge(e);

                    vpEdgesPlane.push_back(e);
                    vnIndexEdgePlane.push_back(i);

//                    int nPointMatches = pFrame->mvPlanePointMatches[i].size();
//
//                    vector<g2o::EdgePlanePoint*> edgesPlanePoint;
//                    vector<size_t> indexEdgePlanePoint;
//                    for (int j = 0; j < nPointMatches; j++) {
//                        MapPoint *mapPoint = pFrame->mvPlanePointMatches[i][j];
//                        if (mapPoint) {
//                            g2o::EdgePlanePoint *edge = new g2o::EdgePlanePoint();
//                            edge->setVertex(0, optimizer.vertex(0));
//                            edge->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
//                            edge->setInformation(Eigen::Matrix3d::Identity() * 1);
//
//                            cv::Mat Pw = mapPoint->GetWorldPos();
//                            edge->Xw[0] = Pw.at<float>(0);
//                            edge->Xw[1] = Pw.at<float>(1);
//                            edge->Xw[2] = Pw.at<float>(2);
//
//                            g2o::RobustKernelHuber *rkEdge = new g2o::RobustKernelHuber;
//                            edge->setRobustKernel(rkEdge);
//                            rkEdge->setDelta(deltaMono);
//
//                            optimizer.addEdge(edge);
//
//                            edgesPlanePoint.push_back(edge);
//                            indexEdgePlanePoint.push_back(j);
//                        }
//                    }
//
//                    int pointEdges = edgesPlanePoint.size();
//                    int nLineMatches = pFrame->mvPlaneLineMatches[i].size();
//
//                    for (int j = 0, index = pointEdges; j < nLineMatches; j++) {
//                        MapLine *mapLine = pFrame->mvPlaneLineMatches[i][j];
//                        if (mapLine) {
//                            g2o::EdgePlanePoint *edgeStart = new g2o::EdgePlanePoint();
//                            edgeStart->setVertex(0, optimizer.vertex(0));
//                            edgeStart->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
//                            edgeStart->setInformation(Eigen::Matrix3d::Identity() * 1);
//
//                            Vector3d startPoint = mapLine->mWorldPos.head(3);
//                            edgeStart->Xw[0] = startPoint(0);
//                            edgeStart->Xw[1] = startPoint(1);
//                            edgeStart->Xw[2] = startPoint(2);
//
//                            g2o::RobustKernelHuber *rkEdgeStart = new g2o::RobustKernelHuber;
//                            edgeStart->setRobustKernel(rkEdgeStart);
//                            rkEdgeStart->setDelta(deltaMono);
//
//                            optimizer.addEdge(edgeStart);
//
//                            edgesPlanePoint.push_back(edgeStart);
//                            indexEdgePlanePoint.push_back(index++);
//
//                            g2o::EdgePlanePoint *edgeEnd = new g2o::EdgePlanePoint();
//                            edgeEnd->setVertex(0, optimizer.vertex(0));
//                            edgeEnd->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
//                            edgeEnd->setInformation(Eigen::Matrix3d::Identity() * 1);
//
//                            Vector3d endPoint = mapLine->mWorldPos.tail(3);
//                            edgeEnd->Xw[0] = endPoint(0);
//                            edgeEnd->Xw[1] = endPoint(1);
//                            edgeEnd->Xw[2] = endPoint(2);
//
//                            g2o::RobustKernelHuber *rkEdgeEnd = new g2o::RobustKernelHuber;
//                            edgeEnd->setRobustKernel(rkEdgeEnd);
//                            rkEdgeEnd->setDelta(deltaMono);
//
//                            optimizer.addEdge(edgeEnd);
//
//                            edgesPlanePoint.push_back(edgeEnd);
//                            indexEdgePlanePoint.push_back(index++);
//                        }
//                    }
//
//                    vpEdgesPlanePoint[i] = edgesPlanePoint;
//                    vnIndexEdgePlanePoint[i] = indexEdgePlanePoint;


                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            //cout << " Plane: " << PEror / PNum << " ";

            PNum = 0;
            PEror = 0;
            PMax = 0;
            for (int i = 0; i < M; ++i) {
                // add parallel planes!
                MapPlane *pMP = pFrame->mvpParallelPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbParPlaneOutlier[i] = false;

                    g2o::EdgeParallelPlaneOnlyPose *e = new g2o::EdgeParallelPlaneOnlyPose();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    Eigen::Matrix2d Info;
                    Info << parInfo, 0,
                            0, parInfo;

                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(VPdeltaPlane);

                    e->Xw = Converter::toPlane3D(pMP->GetWorldPos());

                    optimizer.addEdge(e);

                    vpEdgesParPlane.push_back(e);
                    vnIndexEdgeParPlane.push_back(i);

                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            //cout << " Par Plane: " << PEror / PNum << " ";

            PNum = 0;
            PEror = 0;
            PMax = 0;

            for (int i = 0; i < M; ++i) {
                // add vertical planes!
                MapPlane *pMP = pFrame->mvpVerticalPlanes[i];
                if (pMP) {
                    nInitialCorrespondences++;
                    pFrame->mvbVerPlaneOutlier[i] = false;

                    g2o::EdgeVerticalPlaneOnlyPose *e = new g2o::EdgeVerticalPlaneOnlyPose();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    Eigen::Matrix2d Info;
                    Info << verInfo, 0,
                            0, verInfo;

                    e->setInformation(Info);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(VPdeltaPlane);

                    e->Xw = Converter::toPlane3D(pMP->GetWorldPos());

                    optimizer.addEdge(e);

                    vpEdgesVerPlane.push_back(e);
                    vnIndexEdgeVerPlane.push_back(i);

                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
                }
            }
            //cout << " Ver Plane: " << PEror / PNum << endl;
        }

        if (nInitialCorrespondences < 3)
            return 0;

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
        const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;
        for (size_t it = 0; it < 4; it++) {

            vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad = 0;
            for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
                g2o::EdgeSE3ProjectXYZOnlyPose *e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                if (pFrame->mvbOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                //cout<<"optimize point chi2, "<<chi2<<endl;
                if (chi2 > chi2Mono[it]) {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    pFrame->mvbOutlier[idx] = false;
                    e->setLevel(0);
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            //cout << "Opti:vpEdgesMono:" << vpEdgesMono.size() << "," << vpEdgesStereo.size() << endl;
            for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
                g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if (pFrame->mvbOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if (chi2 > chi2Stereo[it]) {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            int PN = 0;
            double PE = 0, PMax = 0;

            for (size_t i = 0, iend = vpEdgesPlane.size(); i < iend; i++) {
                g2o::EdgePlaneOnlyPose *e = vpEdgesPlane[i];

                const size_t idx = vnIndexEdgePlane[i];

                if (pFrame->mvbPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > planeChi) {
                    pFrame->mvbPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
//                    cout << "bad: " << chi2 << ", id: " << idx << "  Pc : " << pFrame->ComputePlaneWorldCoeff(idx).t()
//                         << "  Pw :" << pFrame->mvpMapPlanes[idx]->GetWorldPos().t() << endl;
                } else {
                    e->setLevel(0);
                    pFrame->mvbPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);

//                if (vpEdgesPlanePoint[i].size() > 0) {
//                    int PPN = 0;
//                    double PPE = 0, PPMax = 0;
//                    for (size_t j = 0, jend = vpEdgesPlanePoint[i].size(); j < jend; j++) {
//                        g2o::EdgePlanePoint *edge = vpEdgesPlanePoint[i][j];
//
//                        const size_t index = vnIndexEdgePlanePoint[i][j];
//
//                        const float chi2 = edge->chi2();
////                    cout<<"optimize chi2"<<chi2<<endl;
//                        PPN++;
//                        PPE += chi2;
//                        PPMax = PPMax > chi2 ? PPMax : chi2;
//
//                        if (chi2 > chi2Mono[it]) {
//                            edge->setLevel(1);
//                            nBad++;
//                        } else {
//                            edge->setLevel(0);
//                        }
//
//                        if (it == 2)
//                            edge->setRobustKernel(0);
//                    }
//
//                    if (PPN == 0)
//                        cout << "planetest No plane point matches " << " ";
//                    else
//                        cout << "planetest  Plane point matches: " << PPE / PPN << " "; //<< " Max: " << PMax << endl;
//                }
            }
//            if (PN == 0)
//                cout << "No plane " << " ";
//            else
//                cout << " Plane: " << PE / PN << " "; //<< " Max: " << PMax << endl;

            PN = 0;
            PE = 0;
            PMax = 0;
            for (size_t i = 0, iend = vpEdgesParPlane.size(); i < iend; i++) {
                g2o::EdgeParallelPlaneOnlyPose *e = vpEdgesParPlane[i];

                const size_t idx = vnIndexEdgeParPlane[i];

                if (pFrame->mvbParPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > VPplaneChi) {
                    pFrame->mvbParPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
//                    cout << "bad Par: " << chi2 << ", id: " << idx << "  Pc : "
//                         << pFrame->ComputePlaneWorldCoeff(idx).t() << "  Pw :"
//                         << pFrame->mvpParallelPlanes[idx]->GetWorldPos().t() << endl;
                } else {
                    e->setLevel(0);
                    pFrame->mvbParPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }
//            if (PN == 0)
//                cout << "No par plane " << " ";
//            else
//                cout << "par Plane: " << PE / PN << " "; //<< " Max: " << PMax << endl;

            PN = 0;
            PE = 0;
            PMax = 0;

            for (size_t i = 0, iend = vpEdgesVerPlane.size(); i < iend; i++) {
                g2o::EdgeVerticalPlaneOnlyPose *e = vpEdgesVerPlane[i];

                const size_t idx = vnIndexEdgeVerPlane[i];

                if (pFrame->mvbVerPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > VPplaneChi) {
                    pFrame->mvbVerPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
//                    cout << "bad Ver: " << chi2 << ", id: " << idx << "  Pc : "
//                         << pFrame->ComputePlaneWorldCoeff(idx).t() << "  Pw :"
//                         << pFrame->mvpVerticalPlanes[idx]->GetWorldPos().t() << endl;
                } else {
                    e->setLevel(0);
                    pFrame->mvbVerPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }
//            if (PN == 0)
//                cout << "No Ver plane " << endl;
//            else
//                cout << "Ver Plane: " << PE / PN << endl; //<< " Max: " << PMax << endl;

            if (optimizer.edges().size() < 10)
                break;
        }

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        pFrame->SetPose(Converter::toCvMat(SE3quat_recov));

        return nInitialCorrespondences - nBad;
    }

    void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map *pMap) {
        // Local KeyFrames: First Breath Search from Current Keyframe
        list<KeyFrame*> lLocalKeyFrames;

        lLocalKeyFrames.push_back(pKF);
        pKF->mnBALocalForKF = pKF->mnId;

        const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
        for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
        {
            KeyFrame* pKFi = vNeighKFs[i];
            pKFi->mnBALocalForKF = pKF->mnId;
            if(!pKFi->isBad())
                lLocalKeyFrames.push_back(pKFi);
        }

        // Local MapPoints seen in Local KeyFrames
        list<MapPoint*> lLocalMapPoints;
        for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
            for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
            {
                MapPoint* pMP = *vit;
                if(pMP)
                    if(!pMP->isBad())
                        if(pMP->mnBALocalForKF!=pKF->mnId)
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF=pKF->mnId;
                        }
            }
        }

        // Local MapLines seen in Local KeyFrames
        list<MapLine *> lLocalMapLines;
        for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
            vector<MapLine *> vpMLs = (*lit)->GetMapLineMatches();
            for (vector<MapLine *>::iterator vit = vpMLs.begin(), vend = vpMLs.end(); vit != vend; vit++) {

                MapLine *pML = *vit;
                if (pML) {
                    if (!pML->isBad()) {
                        if (pML->mnBALocalForKF != pKF->mnId) {
                            lLocalMapLines.push_back(pML);
                            pML->mnBALocalForKF = pKF->mnId;
                        }
                    }
                }
            }
        }

        // Local MapPlanes seen in Local KeyFrames
        list<MapPlane *> lLocalMapPlanes;
        for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++) {
            vector<MapPlane *> vpMPs = (*lit)->GetMapPlaneMatches();
            for (vector<MapPlane *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++) {

                MapPlane *pMP = *vit;
                if (pMP) {
                    if (!pMP->isBad()) {
                        if (pMP->mnBALocalForKF != pKF->mnId) {
                            lLocalMapPlanes.push_back(pMP);
                            pMP->mnBALocalForKF = pKF->mnId;
                        }
                    }
                }
            }
        }

        // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
        list<KeyFrame*> lFixedCameras;
        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
            for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
                {
                    pKFi->mnBAFixedForKF=pKF->mnId;
                    if(!pKFi->isBad())
                        lFixedCameras.push_back(pKFi);
                }
            }
        }
        for(list<MapLine*>::iterator lit=lLocalMapLines.begin(), lend=lLocalMapLines.end(); lit!=lend; lit++)
        {
            map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
            for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
                {
                    pKFi->mnBAFixedForKF=pKF->mnId;
                    if(!pKFi->isBad())
                        lFixedCameras.push_back(pKFi);
                }
            }
        }
        for(list<MapPlane*>::iterator lit=lLocalMapPlanes.begin(), lend=lLocalMapPlanes.end(); lit!=lend; lit++)
        {
            map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
            for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
                {
                    pKFi->mnBAFixedForKF=pKF->mnId;
                    if(!pKFi->isBad())
                        lFixedCameras.push_back(pKFi);
                }
            }
        }

        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        if(pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        unsigned long maxKFid = 0;

        // Set Local KeyFrame vertices
        for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(pKFi->mnId==0);
            optimizer.addVertex(vSE3);
            if(pKFi->mnId>maxKFid)
                maxKFid=pKFi->mnId;
        }

        // Set Fixed KeyFrame vertices
        for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++)
        {
            KeyFrame* pKFi = *lit;
            g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(true);
            optimizer.addVertex(vSE3);
            if(pKFi->mnId>maxKFid)
                maxKFid=pKFi->mnId;
        }

        // Set MapPoint vertices
        const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

        vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
        vpEdgesMono.reserve(nExpectedSize);

        vector<KeyFrame*> vpEdgeKFMono;
        vpEdgeKFMono.reserve(nExpectedSize);

        vector<MapPoint*> vpMapPointEdgeMono;
        vpMapPointEdgeMono.reserve(nExpectedSize);

        vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
        vpEdgesStereo.reserve(nExpectedSize);

        vector<KeyFrame*> vpEdgeKFStereo;
        vpEdgeKFStereo.reserve(nExpectedSize);

        vector<MapPoint*> vpMapPointEdgeStereo;
        vpMapPointEdgeStereo.reserve(nExpectedSize);

        const float thHuberMono = sqrt(5.991);
        const float thHuberStereo = sqrt(7.815);

        long unsigned int maxMapPointId = maxKFid;

        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            int id = pMP->mnId+maxKFid+1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);
            if (id > maxMapPointId) {
                maxMapPointId = id;
            }

            const map<KeyFrame*,size_t> observations = pMP->GetObservations();

            //Set edges
            for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;

                if(!pKFi->isBad())
                {
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                    // Monocular observation
                    if(pKFi->mvuRight[mit->second]<0)
                    {
                        Eigen::Matrix<double,2,1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        e->fx = pKFi->fx;
                        e->fy = pKFi->fy;
                        e->cx = pKFi->cx;
                        e->cy = pKFi->cy;

                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);
                    }
                    else // Stereo observation
                    {
                        Eigen::Matrix<double,3,1> obs;
                        const float kp_ur = pKFi->mvuRight[mit->second];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                        e->setInformation(Info);

                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);

                        e->fx = pKFi->fx;
                        e->fy = pKFi->fy;
                        e->cx = pKFi->cx;
                        e->cy = pKFi->cy;
                        e->bf = pKFi->mbf;

                        optimizer.addEdge(e);
                        vpEdgesStereo.push_back(e);
                        vpEdgeKFStereo.push_back(pKFi);
                        vpMapPointEdgeStereo.push_back(pMP);
                    }
                }
            }
        }

        const int nLineExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapLines.size();

        vector<EdgeLineProjectXYZ*> vpLineEdgesStart;
        vpLineEdgesStart.reserve(nLineExpectedSize);

        vector<EdgeLineProjectXYZ*> vpLineEdgesEnd;
        vpLineEdgesEnd.reserve(nLineExpectedSize);

        vector<KeyFrame*> vpLineEdgeKF;
        vpLineEdgeKF.reserve(nLineExpectedSize);

        vector<MapLine*> vpMapLineEdge;
        vpMapLineEdge.reserve(nLineExpectedSize);

        long unsigned int maxMapLineId = maxMapPointId;

        for (list<MapLine *>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++) {
            MapLine *pML = *lit;
            g2o::VertexSBAPointXYZ *vStartPoint = new g2o::VertexSBAPointXYZ();
            vStartPoint->setEstimate(pML->GetWorldPos().head(3));
            int id1 = (2 * pML->mnId) + 1 + maxMapPointId;
            vStartPoint->setId(id1);
            vStartPoint->setMarginalized(true);
            optimizer.addVertex(vStartPoint);

            g2o::VertexSBAPointXYZ *vEndPoint = new VertexSBAPointXYZ();
            vEndPoint->setEstimate(pML->GetWorldPos().tail(3));
            int id2 = (2 * (pML->mnId + 1)) + maxMapPointId;
            vEndPoint->setId(id2);
            vEndPoint->setMarginalized(true);
            optimizer.addVertex(vEndPoint);

            if (id2 > maxMapLineId) {
                maxMapLineId = id2;
            }

            const map<KeyFrame *, size_t> observations = pML->GetObservations();

            for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end();
                 mit != mend; mit++) {
                KeyFrame *pKFi = mit->first;

                if (!pKFi->isBad()) {

                    Eigen::Vector3d lineObs = pKF->mvKeyLineFunctions[mit->second];

                    EdgeLineProjectXYZ *es = new EdgeLineProjectXYZ();
                    es->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
                    es->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                    es->setMeasurement(lineObs);
                    es->setInformation(Eigen::Matrix3d::Identity());

                    g2o::RobustKernelHuber *rks = new g2o::RobustKernelHuber;
                    es->setRobustKernel(rks);
                    rks->setDelta(thHuberStereo);

                    es->fx = pKF->fx;
                    es->fy = pKF->fy;
                    es->cx = pKF->cx;
                    es->cy = pKF->cy;

                    optimizer.addEdge(es);

                    EdgeLineProjectXYZ *ee = new EdgeLineProjectXYZ();
                    ee->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
                    ee->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                    ee->setMeasurement(lineObs);
                    ee->setInformation(Eigen::Matrix3d::Identity());

                    g2o::RobustKernelHuber *rke = new g2o::RobustKernelHuber;
                    ee->setRobustKernel(rke);
                    rke->setDelta(thHuberStereo);

                    ee->fx = pKF->fx;
                    ee->fy = pKF->fy;
                    ee->cx = pKF->cx;
                    ee->cy = pKF->cy;

                    optimizer.addEdge(ee);

                    vpLineEdgesStart.push_back(es);
                    vpLineEdgesEnd.push_back(ee);
                    vpLineEdgeKF.push_back(pKFi);
                    vpMapLineEdge.push_back(pML);
                }
            }
        }

        double angleInfo = Config::Get<double>("Plane.AngleInfo");
        angleInfo = 3282.8 / (angleInfo * angleInfo);
        double disInfo = Config::Get<double>("Plane.DistanceInfo");
        disInfo = disInfo * disInfo;
        double parInfo = Config::Get<double>("Plane.ParallelInfo");
        parInfo = 3282.8 / (parInfo * parInfo);
        double verInfo = Config::Get<double>("Plane.VerticalInfo");
        verInfo = 3282.8 / (verInfo * verInfo);
        double planeChi = Config::Get<double>("Plane.Chi");
        const float deltaPlane = sqrt(planeChi);

        double VPplaneChi = Config::Get<double>("Plane.VPChi");
        const float VPdeltaPlane = sqrt(VPplaneChi);

        const int nPlaneExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPlanes.size();

        vector<g2o::EdgePlane*> vpPlaneEdges;
        vpPlaneEdges.reserve(nPlaneExpectedSize);

        vector<g2o::EdgeVerticalPlane*> vpVerPlaneEdges;
        vpVerPlaneEdges.reserve(nPlaneExpectedSize);

        vector<g2o::EdgeParallelPlane*> vpParPlaneEdges;
        vpParPlaneEdges.reserve(nPlaneExpectedSize);

        vector<KeyFrame*> vpPlaneEdgeKF;
        vpLineEdgeKF.reserve(nPlaneExpectedSize);

        vector<KeyFrame*> vpVerPlaneEdgeKF;
        vpVerPlaneEdgeKF.reserve(nPlaneExpectedSize);

        vector<KeyFrame*> vpParPlaneEdgeKF;
        vpParPlaneEdgeKF.reserve(nPlaneExpectedSize);

        vector<MapPlane*> vpMapPlaneEdge;
        vpMapPlaneEdge.reserve(nPlaneExpectedSize);

        vector<MapPlane*> vpVerMapPlaneEdge;
        vpVerMapPlaneEdge.reserve(nPlaneExpectedSize);

        vector<MapPlane*> vpParMapPlaneEdge;
        vpParMapPlaneEdge.reserve(nPlaneExpectedSize);

        long unsigned int maxMapPlaneId = maxMapLineId;

        // Set MapPlane vertices
        for (list<MapPlane *>::iterator lit = lLocalMapPlanes.begin(), lend = lLocalMapPlanes.end(); lit != lend; lit++) {
            MapPlane *pMP = *lit;

            g2o::VertexPlane *vPlane = new g2o::VertexPlane();
            vPlane->setEstimate(Converter::toPlane3D(pMP->GetWorldPos()));
            const int id = pMP->mnId + maxMapLineId + 1;
            vPlane->setId(id);
            vPlane->setMarginalized(true);
            optimizer.addVertex(vPlane);

            Eigen::Matrix3d Info;
            Info << angleInfo, 0, 0,
                    0, angleInfo, 0,
                    0, 0, disInfo;

            Eigen::Matrix2d VPInfo;
            VPInfo << angleInfo, 0,
                    0, angleInfo;

            const map<KeyFrame *, size_t> observations = pMP->GetObservations();
            for (const auto &observation : observations) {

                KeyFrame *pKF = observation.first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                g2o::EdgePlane *e = new g2o::EdgePlane();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setMeasurement(Converter::toPlane3D(pKF->mvPlaneCoefficients[observation.second]));
                e->setInformation(Info);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaPlane);

                e->planePoints = pMP->mvPlanePoints;

                optimizer.addEdge(e);
                vpPlaneEdges.push_back(e);
                vpPlaneEdgeKF.push_back(pKF);
                vpMapPlaneEdge.push_back(pMP);
            }

            const map<KeyFrame *, size_t> verObservations = pMP->GetVerObservations();
            for (const auto &observation : verObservations) {

                KeyFrame *pKF = observation.first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                g2o::EdgeVerticalPlane *e = new g2o::EdgeVerticalPlane();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setMeasurement(Converter::toPlane3D(pKF->mvPlaneCoefficients[observation.second]));
                e->setInformation(VPInfo);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(VPdeltaPlane);

                optimizer.addEdge(e);
                vpVerPlaneEdges.push_back(e);
                vpVerPlaneEdgeKF.push_back(pKF);
                vpVerMapPlaneEdge.push_back(pMP);
            }

            const map<KeyFrame *, size_t> parObservations = pMP->GetParObservations();
            for (const auto &observation : parObservations) {

                KeyFrame *pKF = observation.first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                g2o::EdgeParallelPlane *e = new g2o::EdgeParallelPlane();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                e->setMeasurement(Converter::toPlane3D(pKF->mvPlaneCoefficients[observation.second]));
                e->setInformation(VPInfo);

                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(VPdeltaPlane);

                optimizer.addEdge(e);
                vpParPlaneEdges.push_back(e);
                vpParPlaneEdgeKF.push_back(pKF);
                vpParMapPlaneEdge.push_back(pMP);
            }
        }

        if(pbStopFlag)
            if(*pbStopFlag)
                return;

        optimizer.initializeOptimization();
        optimizer.optimize(5);

        bool bDoMore= true;

        if(pbStopFlag)
            if(*pbStopFlag)
                bDoMore = false;

        if(bDoMore)
        {

            // Check inlier observations
            for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
            {
                g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
                MapPoint* pMP = vpMapPointEdgeMono[i];

                if(pMP->isBad())
                    continue;

                if(e->chi2()>5.991 || !e->isDepthPositive())
                {
                    e->setLevel(1);
                }

                e->setRobustKernel(0);
            }

            for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
            {
                g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
                MapPoint* pMP = vpMapPointEdgeStereo[i];

                if(pMP->isBad())
                    continue;

                if(e->chi2()>7.815 || !e->isDepthPositive())
                {
                    e->setLevel(1);
                }

                e->setRobustKernel(0);
            }

            for (size_t i = 0, iend = vpLineEdgesStart.size(); i < iend; i++) {
                EdgeLineProjectXYZ *es = vpLineEdgesStart[i];
                EdgeLineProjectXYZ *ee = vpLineEdgesEnd[i];
                MapLine *pML = vpMapLineEdge[i];

                if (pML->isBad())
                    continue;

                if (es->chi2() > 7.815 || ee->chi2() > 7.815) {
                    es->setLevel(1);
                    ee->setLevel(1);
                }

                es->setRobustKernel(0);
                ee->setRobustKernel(0);
            }

            for(size_t i=0, iend=vpPlaneEdges.size(); i<iend;i++)
            {
                g2o::EdgePlane* e = vpPlaneEdges[i];
                MapPlane* pMP = vpMapPlaneEdge[i];

                if(pMP->isBad())
                    continue;

                if(e->chi2()>planeChi)
                {
                    e->setLevel(1);
                }

                e->setRobustKernel(0);
            }

            for(size_t i=0, iend=vpVerPlaneEdges.size(); i<iend;i++)
            {
                g2o::EdgeVerticalPlane* e = vpVerPlaneEdges[i];
                MapPlane* pMP = vpVerMapPlaneEdge[i];

                if(pMP->isBad())
                    continue;

                if(e->chi2()>VPplaneChi)
                {
                    e->setLevel(1);
                }

                e->setRobustKernel(0);
            }

            for(size_t i=0, iend=vpParPlaneEdges.size(); i<iend;i++)
            {
                g2o::EdgeParallelPlane* e = vpParPlaneEdges[i];
                MapPlane* pMP = vpParMapPlaneEdge[i];

                if(pMP->isBad())
                    continue;

                if(e->chi2()>VPplaneChi)
                {
                    e->setLevel(1);
                }

                e->setRobustKernel(0);
            }

            // Optimize again without the outliers

            optimizer.initializeOptimization(0);
            optimizer.optimize(10);

        }

        vector<pair<KeyFrame*,MapPoint*> > vToErase;
        vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

        // Check inlier observations
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                KeyFrame* pKFi = vpEdgeKFMono[i];
                vToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>7.815 || !e->isDepthPositive())
            {
                KeyFrame* pKFi = vpEdgeKFStereo[i];
                vToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        vector<pair<KeyFrame *, MapLine *>> vLineToErase;
        vLineToErase.reserve(vpLineEdgesStart.size());

        for (size_t i = 0, iend = vpLineEdgesStart.size(); i < iend; i++) {
            EdgeLineProjectXYZ *es = vpLineEdgesStart[i];
            EdgeLineProjectXYZ *ee = vpLineEdgesEnd[i];
            MapLine *pML = vpMapLineEdge[i];

            if (pML->isBad())
                continue;

            if (es->chi2() > 7.815 || ee->chi2() > 7.815) {
                KeyFrame *pKFi = vpLineEdgeKF[i];
                vLineToErase.push_back(make_pair(pKFi, pML));
            }
        }

        vector<pair<KeyFrame*,MapPlane*> > vPlaneToErase;
        vPlaneToErase.reserve(vpPlaneEdges.size());

        for(size_t i=0, iend=vpPlaneEdges.size(); i<iend;i++)
        {
            g2o::EdgePlane* e = vpPlaneEdges[i];
            MapPlane* pMP = vpMapPlaneEdge[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>planeChi)
            {
                KeyFrame* pKFi = vpPlaneEdgeKF[i];
                vPlaneToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        vector<pair<KeyFrame*,MapPlane*> > vVerPlaneToErase;
        vVerPlaneToErase.reserve(vpVerPlaneEdges.size());

        for(size_t i=0, iend=vpVerPlaneEdges.size(); i<iend;i++)
        {
            g2o::EdgeVerticalPlane* e = vpVerPlaneEdges[i];
            MapPlane* pMP = vpVerMapPlaneEdge[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>VPplaneChi)
            {
                KeyFrame* pKFi = vpVerPlaneEdgeKF[i];
                vVerPlaneToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        vector<pair<KeyFrame*,MapPlane*> > vParPlaneToErase;
        vParPlaneToErase.reserve(vpParPlaneEdges.size());

        for(size_t i=0, iend=vpParPlaneEdges.size(); i<iend;i++)
        {
            g2o::EdgeParallelPlane* e = vpParPlaneEdges[i];
            MapPlane* pMP = vpParMapPlaneEdge[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>VPplaneChi)
            {
                KeyFrame* pKFi = vpParPlaneEdgeKF[i];
                vParPlaneToErase.push_back(make_pair(pKFi,pMP));
            }
        }

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);

        if(!vToErase.empty())
        {
            for(size_t i=0;i<vToErase.size();i++)
            {
                KeyFrame* pKFi = vToErase[i].first;
                MapPoint* pMPi = vToErase[i].second;
                pKFi->EraseMapPointMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }

        if(!vLineToErase.empty())
        {
            for(size_t i=0;i<vLineToErase.size();i++)
            {
                KeyFrame* pKFi = vLineToErase[i].first;
                MapLine* pMLi = vLineToErase[i].second;
                pKFi->EraseMapLineMatch(pMLi);
                pMLi->EraseObservation(pKFi);
            }
        }

        if(!vPlaneToErase.empty())
        {
            for(size_t i=0;i<vPlaneToErase.size();i++)
            {
                KeyFrame* pKFi = vPlaneToErase[i].first;
                MapPlane* pMPi = vPlaneToErase[i].second;
                pKFi->EraseMapPlaneMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }

        if(!vVerPlaneToErase.empty())
        {
            for(size_t i=0;i<vVerPlaneToErase.size();i++)
            {
                KeyFrame* pKFi = vVerPlaneToErase[i].first;
                MapPlane* pMPi = vVerPlaneToErase[i].second;
                pKFi->EraseMapVerticalPlaneMatch(pMPi);
                pMPi->EraseVerObservation(pKFi);
            }
        }

        if(!vParPlaneToErase.empty())
        {
            for(size_t i=0;i<vParPlaneToErase.size();i++)
            {
                KeyFrame* pKFi = vParPlaneToErase[i].first;
                MapPlane* pMPi = vParPlaneToErase[i].second;
                pKFi->EraseMapParallelPlaneMatch(pMPi);
                pMPi->EraseParObservation(pKFi);
            }
        }

        // Recover optimized data

        //Keyframes
        for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
        {
            KeyFrame* pKF = *lit;
            g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }

        //Points
        for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
        {
            MapPoint* pMP = *lit;
            g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }

        // Lines
        for (list<MapLine *>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++) {
            MapLine *pML = *lit;

            g2o::VertexSBAPointXYZ *vStartPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(
                    (2 * pML->mnId) + 1 + maxMapPointId));
            g2o::VertexSBAPointXYZ *vEndPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(
                    (2 * (pML->mnId + 1)) + maxMapPointId));

            Vector6d LinePos;
            LinePos << Converter::toVector3d(Converter::toCvMat(vStartPoint->estimate())), Converter::toVector3d(
                    Converter::toCvMat(vEndPoint->estimate()));
            pML->SetWorldPos(LinePos);
            pML->UpdateAverageDir();
        }

        //Planes
        for (list<MapPlane *>::iterator lit = lLocalMapPlanes.begin(), lend = lLocalMapPlanes.end(); lit != lend; lit++) {
            MapPlane *pMP = *lit;
            g2o::VertexPlane *vPlane = static_cast<g2o::VertexPlane *>(optimizer.vertex(
                    pMP->mnId + maxMapLineId + 1));
            pMP->SetWorldPos(Converter::toCvMat(vPlane->estimate()));
            pMP->UpdateCoefficientsAndPoints();
        }
    }

    void Optimizer::OptimizeEssentialGraph(Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF,
                                           const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                           const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                           const map<KeyFrame *, set<KeyFrame *> > &LoopConnections,
                                           const bool &bFixScale) {
        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        optimizer.setVerbose(false);
        g2o::BlockSolver_7_3::LinearSolverType *linearSolver =
                new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
        g2o::BlockSolver_7_3 *solver_ptr = new g2o::BlockSolver_7_3(linearSolver);
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

        solver->setUserLambdaInit(1e-16);
        optimizer.setAlgorithm(solver);

        const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
        const vector<MapPoint *> vpMPs = pMap->GetAllMapPoints();
        const vector<MapLine *> vpMLs = pMap->GetAllMapLines();
        const vector<MapPlane *> vpMPLs = pMap->GetAllMapPlanes();

        const unsigned int nMaxKFid = pMap->GetMaxKFid();

        vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid + 1);
        vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid + 1);
        vector<g2o::VertexSim3Expmap *> vpVertices(nMaxKFid + 1);

        const int minFeat = 100;

        // Set KeyFrame vertices
        for (auto pKF : vpKFs) {
            if (pKF->isBad())
                continue;
            g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

            const int nIDi = pKF->mnId;

            auto it = CorrectedSim3.find(pKF);

            if (it != CorrectedSim3.end()) {
                vScw[nIDi] = it->second;
                VSim3->setEstimate(it->second);
            } else {
                Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
                Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKF->GetTranslation());
                g2o::Sim3 Siw(Rcw, tcw, 1.0);
                vScw[nIDi] = Siw;
                VSim3->setEstimate(Siw);
            }

            if (pKF == pLoopKF)
                VSim3->setFixed(true);

            VSim3->setId(nIDi);
            VSim3->setMarginalized(false);
            VSim3->_fix_scale = bFixScale;

            optimizer.addVertex(VSim3);

            vpVertices[nIDi] = VSim3;
        }


        set<pair<long unsigned int, long unsigned int> > sInsertedEdges;

        const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();

        // Set Loop edges
        for (const auto &LoopConnection : LoopConnections) {
            KeyFrame *pKF = LoopConnection.first;
            const long unsigned int nIDi = pKF->mnId;
            const set<KeyFrame *> &spConnections = LoopConnection.second;
            const g2o::Sim3 Siw = vScw[nIDi];
            const g2o::Sim3 Swi = Siw.inverse();

            for (auto spConnection : spConnections) {
                const long unsigned int nIDj = spConnection->mnId;
                if ((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) && pKF->GetWeight(spConnection) < minFeat)
                    continue;

                const g2o::Sim3 Sjw = vScw[nIDj];
                const g2o::Sim3 Sji = Sjw * Swi;

                g2o::EdgeSim3 *e = new g2o::EdgeSim3();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                e->setMeasurement(Sji);

                e->information() = matLambda;

                optimizer.addEdge(e);

                sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
            }
        }

        // Set normal edges
        for (auto pKF : vpKFs) {
            const int nIDi = pKF->mnId;

            g2o::Sim3 Swi;

            auto iti = NonCorrectedSim3.find(pKF);

            if (iti != NonCorrectedSim3.end())
                Swi = (iti->second).inverse();
            else
                Swi = vScw[nIDi].inverse();

            KeyFrame *pParentKF = pKF->GetParent();

            // Spanning tree edge
            if (pParentKF) {
                int nIDj = pParentKF->mnId;

                g2o::Sim3 Sjw;

                auto itj = NonCorrectedSim3.find(pParentKF);

                if (itj != NonCorrectedSim3.end())
                    Sjw = itj->second;
                else
                    Sjw = vScw[nIDj];

                g2o::Sim3 Sji = Sjw * Swi;

                g2o::EdgeSim3 *e = new g2o::EdgeSim3();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                e->setMeasurement(Sji);

                e->information() = matLambda;
                optimizer.addEdge(e);
            }

            // Loop edges
            const set<KeyFrame *> sLoopEdges = pKF->GetLoopEdges();
            for (auto pLKF : sLoopEdges) {
                if (pLKF->mnId < pKF->mnId) {
                    g2o::Sim3 Slw;

                    auto itl = NonCorrectedSim3.find(pLKF);

                    if (itl != NonCorrectedSim3.end())
                        Slw = itl->second;
                    else
                        Slw = vScw[pLKF->mnId];

                    g2o::Sim3 Sli = Slw * Swi;
                    g2o::EdgeSim3 *el = new g2o::EdgeSim3();
                    el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->mnId)));
                    el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                    el->setMeasurement(Sli);
                    el->information() = matLambda;
                    optimizer.addEdge(el);
                }
            }

            // Covisibility graph edges
            const vector<KeyFrame *> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
            for (auto pKFn : vpConnectedKFs) {
                if (pKFn && pKFn != pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn)) {
                    if (!pKFn->isBad() && pKFn->mnId < pKF->mnId) {
                        if (sInsertedEdges.count(make_pair(min(pKF->mnId, pKFn->mnId), max(pKF->mnId, pKFn->mnId))))
                            continue;

                        g2o::Sim3 Snw;

                        auto itn = NonCorrectedSim3.find(pKFn);

                        if (itn != NonCorrectedSim3.end())
                            Snw = itn->second;
                        else
                            Snw = vScw[pKFn->mnId];

                        g2o::Sim3 Sni = Snw * Swi;

                        g2o::EdgeSim3 *en = new g2o::EdgeSim3();
                        en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->mnId)));
                        en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                        en->setMeasurement(Sni);
                        en->information() = matLambda;
                        optimizer.addEdge(en);
                    }
                }
            }
        }

        // Optimize!
        optimizer.initializeOptimization();
        optimizer.optimize(20);

        unique_lock<mutex> lock(pMap->mMutexMapUpdate);

        // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
        for (auto pKFi : vpKFs) {
            const int nIDi = pKFi->mnId;

            g2o::VertexSim3Expmap *VSim3 = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(nIDi));
            g2o::Sim3 CorrectedSiw = VSim3->estimate();
            vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
            Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = CorrectedSiw.translation();
            double s = CorrectedSiw.scale();

            eigt *= (1. / s); //[R t/s;0 1]

            cv::Mat Tiw = Converter::toCvSE3(eigR, eigt);

            pKFi->SetPose(Tiw);
        }

        // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
        for (auto pMP : vpMPs) {
            if (pMP->isBad())
                continue;

            int nIDr;
            if (pMP->mnCorrectedByKF == pCurKF->mnId) {
                nIDr = pMP->mnCorrectedReference;
            } else {
                KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
                nIDr = pRefKF->mnId;
            }


            g2o::Sim3 Srw = vScw[nIDr];
            g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

            cv::Mat P3Dw = pMP->GetWorldPos();
            Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
            Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

            cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
            pMP->SetWorldPos(cvCorrectedP3Dw);

            pMP->UpdateNormalAndDepth();
        }

        // Correct lines. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
        for (auto pML : vpMLs) {
            if (pML->isBad())
                continue;

            int nIDr;
            if (pML->mnCorrectedByKF == pCurKF->mnId) {
                nIDr = pML->mnCorrectedReference;
            } else {
                KeyFrame *pRefKF = pML->GetReferenceKeyFrame();
                nIDr = pRefKF->mnId;
            }


            g2o::Sim3 Srw = vScw[nIDr];
            g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

            Eigen::Vector3d eigSP3Dw = pML->mWorldPos.head(3);
            Eigen::Vector3d eigEP3Dw = pML->mWorldPos.tail(3);

            Eigen::Matrix<double, 3, 1> eigCorrectedSP3Dw = correctedSwr.map(Srw.map(eigSP3Dw));
            Eigen::Matrix<double, 3, 1> eigCorrectedEP3Dw = correctedSwr.map(Srw.map(eigEP3Dw));

            Vector6d linePos;
            linePos << eigCorrectedSP3Dw(0), eigCorrectedSP3Dw(1), eigCorrectedSP3Dw(2), eigCorrectedEP3Dw(
                    0), eigCorrectedEP3Dw(1), eigCorrectedEP3Dw(2);
            pML->SetWorldPos(linePos);
            pML->ComputeDistinctiveDescriptors();
            pML->UpdateAverageDir();
        }

        // Correct planes. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
        for (auto pMP : vpMPLs) {
            if (pMP->isBad())
                continue;

            int nIDr;
            if (pMP->mnCorrectedByKF == pCurKF->mnId) {
                nIDr = pMP->mnCorrectedReference;
            } else {
                KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
                nIDr = pRefKF->mnId;
            }

            cv::Mat Srw = Converter::toCvMat(vScw[nIDr]);
            cv::Mat correctedSwr = Converter::toCvMat(vCorrectedSwc[nIDr]);

            cv::Mat sRSrw = Srw.rowRange(0, 3).colRange(0, 3);
            cv::Mat tSrw = Srw.rowRange(0, 3).col(3);

            cv::Mat sRCorrectedSwr = correctedSwr.rowRange(0, 3).colRange(0, 3);
            cv::Mat tCorrectedSwr = correctedSwr.rowRange(0, 3).col(3);

            cv::Mat P3Dw = pMP->GetWorldPos();

            cv::Mat correctedP3Dw = cv::Mat::eye(4, 1, CV_32F);

            correctedP3Dw.rowRange(0, 3).col(0) = sRSrw * P3Dw.rowRange(0, 3).col(0);
            correctedP3Dw.at<float>(3, 0) =
                    P3Dw.at<float>(3, 0) - tSrw.dot(correctedP3Dw.rowRange(0, 3).col(0));
            if (correctedP3Dw.at<float>(3, 0) < 0.0)
                correctedP3Dw = -correctedP3Dw;

            correctedP3Dw.rowRange(0, 3).col(0) = sRCorrectedSwr * correctedP3Dw.rowRange(0, 3).col(0);
            correctedP3Dw.at<float>(3, 0) =
                    correctedP3Dw.at<float>(3, 0) - tCorrectedSwr.dot(correctedP3Dw.rowRange(0, 3).col(0));
            if (correctedP3Dw.at<float>(3, 0) < 0.0)
                correctedP3Dw = -correctedP3Dw;

            pMP->SetWorldPos(correctedP3Dw);
            pMP->UpdateCoefficientsAndPoints();
        }
    }



    int Optimizer::TranslationOptimization(Planar_SLAM::Frame *pFrame) {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        int nInitialCorrespondences = 0;

        // Set Frame vertex
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // Set MapPoint vertices
        const int N = pFrame->N;

        //rotation
        cv::Mat R_cw = pFrame->mTcw.rowRange(0, 3).colRange(0, 3).clone();

        vector<g2o::EdgeSE3ProjectXYZOnlyTranslation *> vpEdgesMono;
        vector<size_t> vnIndexEdgeMono;
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);

        vector<g2o::EdgeStereoSE3ProjectXYZOnlyTranslation *> vpEdgesStereo;
        vector<size_t> vnIndexEdgeStereo;
        vpEdgesStereo.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float deltaMono = sqrt(5.991);
        const float deltaStereo = sqrt(7.815);


        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);

            for (int i = 0; i < N; i++) {
                MapPoint *pMP = pFrame->mvpMapPoints[i];
                if (pMP) {
                    // Monocular observation
                    if (pFrame->mvuRight[i] < 0) {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        Eigen::Matrix<double, 2, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        obs << kpUn.pt.x, kpUn.pt.y;

                        g2o::EdgeSE3ProjectXYZOnlyTranslation *e = new g2o::EdgeSE3ProjectXYZOnlyTranslation();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaMono);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        cv::Mat Xw = pMP->GetWorldPos();
                        cv::Mat Xc = R_cw * Xw;

                        e->Xc[0] = Xc.at<float>(0);
                        e->Xc[1] = Xc.at<float>(1);
                        e->Xc[2] = Xc.at<float>(2);


                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    } else  // Stereo observation
                    {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        //SET EDGE
                        Eigen::Matrix<double, 3, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        const float &kp_ur = pFrame->mvuRight[i];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZOnlyTranslation *e = new g2o::EdgeStereoSE3ProjectXYZOnlyTranslation();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                        e->setInformation(Info);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaStereo);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        e->bf = pFrame->mbf;
                        cv::Mat Xw = pMP->GetWorldPos();
                        cv::Mat Xc = R_cw * Xw;

                        e->Xc[0] = Xc.at<float>(0);
                        e->Xc[1] = Xc.at<float>(1);
                        e->Xc[2] = Xc.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesStereo.push_back(e);
                        vnIndexEdgeStereo.push_back(i);
                    }
                }

            }
        }

        const int NL = pFrame->NL;

        vector<EdgeLineProjectXYZOnlyTranslation *> vpEdgesLineSp;
        vector<size_t> vnIndexLineEdgeSp;
        vpEdgesLineSp.reserve(NL);
        vnIndexLineEdgeSp.reserve(NL);

        vector<EdgeLineProjectXYZOnlyTranslation *> vpEdgesLineEp;
        vpEdgesLineEp.reserve(NL);

        {
            unique_lock<mutex> lock(MapLine::mGlobalMutex);

            for (int i = 0; i < NL; i++) {
                MapLine *pML = pFrame->mvpMapLines[i];
                if (pML) {
                    pFrame->mvbLineOutlier[i] = false;

                    Eigen::Vector3d line_obs;
                    line_obs = pFrame->mvKeyLineFunctions[i];

                    EdgeLineProjectXYZOnlyTranslation *els = new EdgeLineProjectXYZOnlyTranslation();

                    els->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    els->setMeasurement(line_obs);
                    els->setInformation(Eigen::Matrix3d::Identity() * 1);//*vSteroStartPointInfo[i]);

                    g2o::RobustKernelHuber *rk_line_s = new g2o::RobustKernelHuber;
                    els->setRobustKernel(rk_line_s);
                    rk_line_s->setDelta(deltaStereo);

                    els->fx = pFrame->fx;
                    els->fy = pFrame->fy;
                    els->cx = pFrame->cx;
                    els->cy = pFrame->cy;

                    cv::Mat Xw = Converter::toCvVec(pML->mWorldPos.head(3));
                    cv::Mat Xc = R_cw * Xw;
                    els->Xc[0] = Xc.at<float>(0);
                    els->Xc[1] = Xc.at<float>(1);
                    els->Xc[2] = Xc.at<float>(2);

                    optimizer.addEdge(els);

                    vpEdgesLineSp.push_back(els);
                    vnIndexLineEdgeSp.push_back(i);

                    EdgeLineProjectXYZOnlyTranslation *ele = new EdgeLineProjectXYZOnlyTranslation();

                    ele->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    ele->setMeasurement(line_obs);
                    ele->setInformation(Eigen::Matrix3d::Identity() * 1);//vSteroEndPointInfo[i]);

                    g2o::RobustKernelHuber *rk_line_e = new g2o::RobustKernelHuber;
                    ele->setRobustKernel(rk_line_e);
                    rk_line_e->setDelta(deltaStereo);

                    ele->fx = pFrame->fx;
                    ele->fy = pFrame->fy;
                    ele->cx = pFrame->cx;
                    ele->cy = pFrame->cy;

                    Xw = Converter::toCvVec(pML->mWorldPos.tail(3));
                    Xc = R_cw * Xw;
                    ele->Xc[0] = Xc.at<float>(0);
                    ele->Xc[1] = Xc.at<float>(1);
                    ele->Xc[2] = Xc.at<float>(2);


                    optimizer.addEdge(ele);

                    vpEdgesLineEp.push_back(ele);
                }
            }
        }


        if (nInitialCorrespondences < 3) {
            return 0;
        }

        //Set Plane vertices
        const int M = pFrame->mnPlaneNum;
        vector<g2o::EdgePlaneOnlyTranslation *> vpEdgesPlane;
        vector<size_t> vnIndexEdgePlane;
        vpEdgesPlane.reserve(M);
        vnIndexEdgePlane.reserve(M);

//        vector<vector<g2o::EdgePlanePointTranslationOnly *>> vpEdgesPlanePoint;
//        vector<vector<size_t>> vnIndexEdgePlanePoint;
//        vpEdgesPlanePoint = vector<vector<g2o::EdgePlanePointTranslationOnly *>>(M);
//        vnIndexEdgePlanePoint = vector<vector<size_t>>(M);

        vector<g2o::EdgeParallelPlaneOnlyTranslation *> vpEdgesParPlane;
        vector<size_t> vnIndexEdgeParPlane;
        vpEdgesParPlane.reserve(M);
        vnIndexEdgeParPlane.reserve(M);

        vector<g2o::EdgeVerticalPlaneOnlyTranslation *> vpEdgesVerPlane;
        vector<size_t> vnIndexEdgeVerPlane;
        vpEdgesVerPlane.reserve(M);
        vnIndexEdgeVerPlane.reserve(M);

        double angleInfo = Config::Get<double>("Plane.AngleInfo");
        angleInfo = 3282.8 / (angleInfo * angleInfo);
        double disInfo = Config::Get<double>("Plane.DistanceInfo");
        disInfo = disInfo * disInfo;
        double parInfo = Config::Get<double>("Plane.ParallelInfo");
        parInfo = 3282.8 / (parInfo * parInfo);
        double verInfo = Config::Get<double>("Plane.VerticalInfo");
        verInfo = 3282.8 / (verInfo * verInfo);
        double planeChi = Config::Get<double>("Plane.Chi");
        const float deltaPlane = sqrt(planeChi);

        double VPplaneChi = Config::Get<double>("Plane.VPChi");
        const float VPdeltaPlane = sqrt(VPplaneChi);

        {
            unique_lock<mutex> lock(MapPlane::mGlobalMutex);
            int PNum = 0;
            double PEror = 0, PMax = 0;
            for (int i = 0; i < M; ++i) {
                MapPlane *pMP = pFrame->mvpMapPlanes[i];
                if (pMP) {
                    pFrame->mvbPlaneOutlier[i] = false;

                    g2o::EdgePlaneOnlyTranslation *e = new g2o::EdgePlaneOnlyTranslation();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
                    //TODO
                    Eigen::Matrix3d Info;
                    Info << angleInfo, 0, 0,
                            0, angleInfo, 0,
                            0, 0, disInfo;
                    e->setInformation(Info);

                    Plane3D Xw = Converter::toPlane3D(pMP->GetWorldPos());
                    Xw.rotateNormal(Converter::toMatrix3d(R_cw));
                    e->Xc = Xw;

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    //TODO
                    rk->setDelta(deltaPlane);

                    optimizer.addEdge(e);

                    vpEdgesPlane.push_back(e);
                    vnIndexEdgePlane.push_back(i);

//                    int nMatches = pFrame->mvPlanePointMatches[i].size();
//
//                    vector<g2o::EdgePlanePointTranslationOnly *> edgesPlanePoint;
//                    vector<size_t> indexEdgePlanePoint;
//                    for (int j = 0; j < nMatches; j++) {
//                        MapPoint *mapPoint = pFrame->mvPlanePointMatches[i][j];
//                        if (mapPoint) {
//                            g2o::EdgePlanePointTranslationOnly *edge = new g2o::EdgePlanePointTranslationOnly();
//                            edge->setVertex(0, optimizer.vertex(0));
//                            edge->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
//                            edge->setInformation(Eigen::Matrix3d::Identity() * 1);
//
//                            cv::Mat Pw = mapPoint->GetWorldPos();
//                            cv::Mat Pc = R_cw * Pw;
//                            edge->Xc[0] = Pc.at<float>(0);
//                            edge->Xc[1] = Pc.at<float>(1);
//                            edge->Xc[2] = Pc.at<float>(2);
//
//                            g2o::RobustKernelHuber *rkEdge = new g2o::RobustKernelHuber;
//                            edge->setRobustKernel(rkEdge);
//                            rkEdge->setDelta(deltaMono);
//
//                            optimizer.addEdge(edge);
//
//                            edgesPlanePoint.push_back(edge);
//                            indexEdgePlanePoint.push_back(j);
//                        }
//                    }
//
//                    int pointEdges = edgesPlanePoint.size();
//                    int nLineMatches = pFrame->mvPlaneLineMatches[i].size();
//
//                    for (int j = 0, index = pointEdges; j < nLineMatches; j++) {
//                        MapLine *mapLine = pFrame->mvPlaneLineMatches[i][j];
//                        if (mapLine) {
//                            g2o::EdgePlanePointTranslationOnly *edgeStart = new g2o::EdgePlanePointTranslationOnly();
//                            edgeStart->setVertex(0, optimizer.vertex(0));
//                            edgeStart->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
//                            edgeStart->setInformation(Eigen::Matrix3d::Identity() * 1);
//
//                            cv::Mat Xw = Converter::toCvVec(mapLine->mWorldPos.head(3));
//                            cv::Mat Xc = R_cw * Xw;
//                            edgeStart->Xc[0] = Xc.at<float>(0);
//                            edgeStart->Xc[1] = Xc.at<float>(1);
//                            edgeStart->Xc[2] = Xc.at<float>(2);
//
//                            g2o::RobustKernelHuber *rkEdgeStart = new g2o::RobustKernelHuber;
//                            edgeStart->setRobustKernel(rkEdgeStart);
//                            rkEdgeStart->setDelta(deltaMono);
//
//                            optimizer.addEdge(edgeStart);
//
//                            edgesPlanePoint.push_back(edgeStart);
//                            indexEdgePlanePoint.push_back(index++);
//
//                            g2o::EdgePlanePointTranslationOnly *edgeEnd = new g2o::EdgePlanePointTranslationOnly();
//                            edgeEnd->setVertex(0, optimizer.vertex(0));
//                            edgeEnd->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
//                            edgeEnd->setInformation(Eigen::Matrix3d::Identity() * 1);
//
//                            Xw = Converter::toCvVec(mapLine->mWorldPos.tail(3));
//                            Xc = R_cw * Xw;
//                            edgeEnd->Xc[0] = Xc.at<float>(0);
//                            edgeEnd->Xc[1] = Xc.at<float>(1);
//                            edgeEnd->Xc[2] = Xc.at<float>(2);
//
//                            g2o::RobustKernelHuber *rkEdgeEnd = new g2o::RobustKernelHuber;
//                            edgeEnd->setRobustKernel(rkEdgeEnd);
//                            rkEdgeEnd->setDelta(deltaMono);
//
//                            optimizer.addEdge(edgeEnd);
//
//                            edgesPlanePoint.push_back(edgeEnd);
//                            indexEdgePlanePoint.push_back(index++);
//                        }
//                    }
//
//                    vpEdgesPlanePoint[i] = edgesPlanePoint;
//                    vnIndexEdgePlanePoint[i] = indexEdgePlanePoint;


                    e->computeError();
                    double chi = e->chi2();
                    PEror += chi;
                    PMax = PMax > chi ? PMax : chi;
                    PNum++;
//                cout << "  done!" << endl;
                }
            }
            //cout << " Plane: " << PEror / PNum << " ";//" Max: " << PMax << " ";

//            PNum = 0;
//            PEror = 0;
//            PMax = 0;
//            for (int i = 0; i < M; ++i) {
//                // add parallel planes!
//                MapPlane *pMP = pFrame->mvpParallelPlanes[i];
//                if (pMP) {
//                    pFrame->mvbParPlaneOutlier[i] = false;
//
//                    g2o::EdgeParallelPlaneOnlyTranslation *e = new g2o::EdgeParallelPlaneOnlyTranslation();
//                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
//                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
//                    //TODO
//                    Eigen::Matrix2d Info;
//                    Info << parInfo, 0,
//                            0, parInfo;
////                    Info << 0, 0,
////                            0, 0;
//
//                    e->setInformation(Info);
//
//                    Plane3D Xw = Converter::toPlane3D(pMP->GetWorldPos());
//                    Xw.rotateNormal(Converter::toMatrix3d(R_cw));
//                    e->Xc = Xw;
//
//                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
//                    e->setRobustKernel(rk);
//                    //TODO
//                    rk->setDelta(VPdeltaPlane);
//                    optimizer.addEdge(e);
//
//                    vpEdgesParPlane.push_back(e);
//                    vnIndexEdgeParPlane.push_back(i);
//
//                    e->computeError();
//                    double chi = e->chi2();
//                    PEror += chi;
//                    PMax = PMax > chi ? PMax : chi;
//                    PNum++;
//                }
//            }
//            cout << " Par Plane: " << PEror / PNum << " ";//" Max: " << PMax << " ";
//            PNum = 0;
//            PEror = 0;
//            PMax = 0;
//
//            for (int i = 0; i < M; ++i) {
//                // add vertical planes!
//                MapPlane *pMP = pFrame->mvpVerticalPlanes[i];
//                if (pMP) {
//                    pFrame->mvbVerPlaneOutlier[i] = false;
//
//                    g2o::EdgeVerticalPlaneOnlyTranslation *e = new g2o::EdgeVerticalPlaneOnlyTranslation();
//                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
//                    e->setMeasurement(Converter::toPlane3D(pFrame->mvPlaneCoefficients[i]));
//                    //TODO
//                    Eigen::Matrix2d Info;
//                    Info << verInfo, 0,
//                            0, verInfo;
////                    Info << 0, 0,
////                            0, 0;
//
//                    e->setInformation(Info);
//
//                    Plane3D Xw = Converter::toPlane3D(pMP->GetWorldPos());
//                    Xw.rotateNormal(Converter::toMatrix3d(R_cw));
//                    e->Xc = Xw;
//
//                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
//                    e->setRobustKernel(rk);
//                    //TODO
//                    rk->setDelta(VPdeltaPlane);
//                    optimizer.addEdge(e);
//
//                    vpEdgesVerPlane.push_back(e);
//                    vnIndexEdgeVerPlane.push_back(i);
//
//                    e->computeError();
//                    double chi = e->chi2();
//                    PEror += chi;
//                    PMax = PMax > chi ? PMax : chi;
//                    PNum++;
//                }
//            }
//            cout << " Ver Plane: " << PEror / PNum << endl;//" Max: " << PMax << endl;
        }

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
        const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;
        int nLineBad = 0;
        for (size_t it = 0; it < 4; it++) {

            vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad = 0;

            int PNMono = 0;
            double PEMono = 0, PMaxMono = 0;
            for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
                g2o::EdgeSE3ProjectXYZOnlyTranslation *e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                if (pFrame->mvbOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
//                cout<<"optimize mono point chi2, "<<chi2<<endl;
                PNMono++;
                PEMono += chi2;
                PMaxMono = PMaxMono > chi2 ? PMaxMono : chi2;

                if (chi2 > chi2Mono[it]) {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    pFrame->mvbOutlier[idx] = false;
                    e->setLevel(0);
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }
//            if (PNMono == 0)
//                cout << "No mono points " << " ";
//            else
//                cout << " Mono points: " << PEMono / PNMono << " "; //<< " Max: " << PMax << endl;

            int PNStereo = 0;
            double PEStereo = 0, PMaxStereo = 0;

//            cout << "Opti:vpEdgesMono:" << vpEdgesMono.size() << "," << vpEdgesStereo.size() << endl;
            for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
                g2o::EdgeStereoSE3ProjectXYZOnlyTranslation *e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if (pFrame->mvbOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
//                cout<<"optimize stereo point chi2, "<<chi2<<endl;
                PNStereo++;
                PEStereo += chi2;
                PMaxStereo = PMaxStereo > chi2 ? PMaxStereo : chi2;

                if (chi2 > chi2Stereo[it]) {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                } else {
                    e->setLevel(0);
                    pFrame->mvbOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

//            if (PNStereo == 0)
//                cout << "No stereo points " << " ";
//            else
//                cout << " Stereo points: " << PEStereo / PNStereo << endl;

            nLineBad = 0;
            int PNLine= 0;
            double PELine = 0, PMaxLine = 0;
//            cout << "Opti:vpEdgesLine:" << vpEdgesLineSp.size() << endl;
            for (size_t i = 0, iend = vpEdgesLineSp.size(); i < iend; i++) {
                EdgeLineProjectXYZOnlyTranslation *e1 = vpEdgesLineSp[i];  //线段起始点
                EdgeLineProjectXYZOnlyTranslation *e2 = vpEdgesLineEp[i];  //线段终止点

                const size_t idx = vnIndexLineEdgeSp[i];    //线段起始点和终止点的误差边的index一样

                if (pFrame->mvbLineOutlier[idx]) {
                    e1->computeError();
                    e2->computeError();
                }

                const float chi2_s = e1->chiline();//e1->chi2();
                const float chi2_e = e2->chiline();//e2->chi2();
//                cout<<"Optimization: chi2_s "<<chi2_s<<", chi2_e "<<chi2_e<<endl;

                PNLine++;
                PELine += chi2_s + chi2_e;
                PMaxLine = PMaxLine > chi2_s + chi2_e ? PMaxLine : chi2_s + chi2_e;


                if (chi2_s > 2 * chi2Mono[it] || chi2_e > 2 * chi2Mono[it]) {
                    pFrame->mvbLineOutlier[idx] = true;
                    e1->setLevel(1);
                    e2->setLevel(1);
                    nLineBad++;
                } else {
                    pFrame->mvbLineOutlier[idx] = false;
                    e1->setLevel(0);
                    e2->setLevel(0);
                }

                if (it == 2) {
                    e1->setRobustKernel(0);
                    e2->setRobustKernel(0);
                }
            }

//            if (PNLine == 0)
//                cout << "No lines " << " ";
//            else
//                cout << " Lines: " << PELine / PNLine << endl;

            int PN = 0;
            double PE = 0, PMax = 0;

            for (size_t i = 0, iend = vpEdgesPlane.size(); i < iend; i++) {
                g2o::EdgePlaneOnlyTranslation *e = vpEdgesPlane[i];

                const size_t idx = vnIndexEdgePlane[i];

                if (pFrame->mvbPlaneOutlier[idx]) {
                    e->computeError();
                }

                const float chi2 = e->chi2();
                PN++;
                PE += chi2;
                PMax = PMax > chi2 ? PMax : chi2;

                if (chi2 > planeChi) {
                    pFrame->mvbPlaneOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
//                    cout << "planetest bad: " << chi2 << ", id: " << idx << "  Pc : "
//                         << pFrame->ComputePlaneWorldCoeff(idx).t() << "  Pw :"
//                         << pFrame->mvpMapPlanes[idx]->GetWorldPos().t() << endl;
                } else {
                    e->setLevel(0);
                    pFrame->mvbPlaneOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);

//                if (vpEdgesPlanePoint[i].size() > 0) {
//                    int PPN = 0;
//                    double PPE = 0, PPMax = 0;
//                    for (size_t j = 0, jend = vpEdgesPlanePoint[i].size(); j < jend; j++) {
//                        g2o::EdgePlanePointTranslationOnly *edge = vpEdgesPlanePoint[i][j];
//
//                        const size_t index = vnIndexEdgePlanePoint[i][j];
//
//                        const float chi2 = edge->chi2();
////                    cout<<"optimize chi2"<<chi2<<endl;
//                        PPN++;
//                        PPE += chi2;
//                        PPMax = PPMax > chi2 ? PPMax : chi2;
//
//                        if (chi2 > chi2Mono[it]) {
//                            edge->setLevel(1);
//                            nBad++;
//                        } else {
//                            edge->setLevel(0);
//                        }
//
//                        if (it == 2)
//                            edge->setRobustKernel(0);
//                    }
//
//                    if (PPN == 0)
//                        cout << "planetest No plane point matches " << " ";
//                    else
//                        cout << "planetest  Plane point matches: " << PPE / PPN << " "; //<< " Max: " << PMax << endl;
//                }
            }
//            if (PN == 0)
//                cout << "planetest No plane " << " ";
//            else
//                cout << "planetest  Plane: " << PE / PN << " "; //<< " Max: " << PMax << endl;

//            PN = 0;
//            PE = 0;
//            PMax = 0;
//            for (size_t i = 0, iend = vpEdgesParPlane.size(); i < iend; i++) {
//                g2o::EdgeParallelPlaneOnlyTranslation *e = vpEdgesParPlane[i];
//
//                const size_t idx = vnIndexEdgeParPlane[i];
//
//                if (pFrame->mvbParPlaneOutlier[idx]) {
//                    e->computeError();
//                }
//
//                const float chi2 = e->chi2();
//                PN++;
//                PE += chi2;
//                PMax = PMax > chi2 ? PMax : chi2;
//
//                if (chi2 > VPplaneChi) {
//                    pFrame->mvbParPlaneOutlier[idx] = true;
//                    e->setLevel(1);
//                    nBad++;
//                    cout << "planetest bad Par: " << chi2 << ", id: " << idx << "  Pc : "
//                         << pFrame->ComputePlaneWorldCoeff(idx).t() << "  Pw :"
//                         << pFrame->mvpParallelPlanes[idx]->GetWorldPos().t() << endl;
//                } else {
//                    e->setLevel(0);
//                    pFrame->mvbParPlaneOutlier[idx] = false;
//                }
//
//                if (it == 2)
//                    e->setRobustKernel(0);
//            }
//            if (PN == 0)
//                cout << "planetest No par plane " << " ";
//            else
//                cout << "planetest par Plane: " << PE / PN << " "; //<< " Max: " << PMax << endl;
//
//            PN = 0;
//            PE = 0;
//            PMax = 0;
//
//            for (size_t i = 0, iend = vpEdgesVerPlane.size(); i < iend; i++) {
//                g2o::EdgeVerticalPlaneOnlyTranslation *e = vpEdgesVerPlane[i];
//
//                const size_t idx = vnIndexEdgeVerPlane[i];
//
//                if (pFrame->mvbVerPlaneOutlier[idx]) {
//                    e->computeError();
//                }
//
//                const float chi2 = e->chi2();
//                PN++;
//                PE += chi2;
//                PMax = PMax > chi2 ? PMax : chi2;
//
//                if (chi2 > VPplaneChi) {
//                    pFrame->mvbVerPlaneOutlier[idx] = true;
//                    e->setLevel(1);
//                    nBad++;
//                    cout << "planetest bad Ver: " << chi2 << ", id: " << idx << "  Pc : "
//                         << pFrame->ComputePlaneWorldCoeff(idx).t() << "  Pw :"
//                         << pFrame->mvpVerticalPlanes[idx]->GetWorldPos().t() << endl;
//                } else {
//                    e->setLevel(0);
//                    pFrame->mvbVerPlaneOutlier[idx] = false;
//                }
//
//                if (it == 2)
//                    e->setRobustKernel(0);
//            }
//            if (PN == 0)
//                cout << "planetest No Ver plane " << endl;
//            else
//                cout << "planetest Ver Plane: " << PE / PN << endl; //<< " Max: " << PMax << endl;

            if (optimizer.edges().size() < 10)
                break;
        }

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        cv::Mat pose = Converter::toCvMat(SE3quat_recov);
        pFrame->SetPose(pose);

        return nInitialCorrespondences - nBad;
    }

    int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
    {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType * linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

        g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        // Calibration
        const cv::Mat &K1 = pKF1->mK;
        const cv::Mat &K2 = pKF2->mK;

        // Camera poses
        const cv::Mat R1w = pKF1->GetRotation();
        const cv::Mat t1w = pKF1->GetTranslation();
        const cv::Mat R2w = pKF2->GetRotation();
        const cv::Mat t2w = pKF2->GetTranslation();

        // Set Sim3 vertex
        g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();
        vSim3->_fix_scale=bFixScale;
        vSim3->setEstimate(g2oS12);
        vSim3->setId(0);
        vSim3->setFixed(false);
        vSim3->_principle_point1[0] = K1.at<float>(0,2);
        vSim3->_principle_point1[1] = K1.at<float>(1,2);
        vSim3->_focal_length1[0] = K1.at<float>(0,0);
        vSim3->_focal_length1[1] = K1.at<float>(1,1);
        vSim3->_principle_point2[0] = K2.at<float>(0,2);
        vSim3->_principle_point2[1] = K2.at<float>(1,2);
        vSim3->_focal_length2[0] = K2.at<float>(0,0);
        vSim3->_focal_length2[1] = K2.at<float>(1,1);
        optimizer.addVertex(vSim3);

        // Set MapPoint vertices
        const int N = vpMatches1.size();
        const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
        vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
        vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
        vector<size_t> vnIndexEdge;

        vnIndexEdge.reserve(2*N);
        vpEdges12.reserve(2*N);
        vpEdges21.reserve(2*N);

        const float deltaHuber = sqrt(th2);

        int nCorrespondences = 0;

        for(int i=0; i<N; i++)
        {
            if(!vpMatches1[i])
                continue;

            MapPoint* pMP1 = vpMapPoints1[i];
            MapPoint* pMP2 = vpMatches1[i];

            const int id1 = 2*i+1;
            const int id2 = 2*(i+1);

            const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

            if(pMP1 && pMP2)
            {
                if(!pMP1->isBad() && !pMP2->isBad() && i2>=0)
                {
                    g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
                    cv::Mat P3D1w = pMP1->GetWorldPos();
                    cv::Mat P3D1c = R1w*P3D1w + t1w;
                    vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                    vPoint1->setId(id1);
                    vPoint1->setFixed(true);
                    optimizer.addVertex(vPoint1);

                    g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
                    cv::Mat P3D2w = pMP2->GetWorldPos();
                    cv::Mat P3D2c = R2w*P3D2w + t2w;
                    vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                    vPoint2->setId(id2);
                    vPoint2->setFixed(true);
                    optimizer.addVertex(vPoint2);
                }
                else
                    continue;
            }
            else
                continue;

            nCorrespondences++;

            // Set edge x1 = S12*X2
            Eigen::Matrix<double,2,1> obs1;
            const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
            obs1 << kpUn1.pt.x, kpUn1.pt.y;

            g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
            e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
            e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e12->setMeasurement(obs1);
            const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
            e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

            g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
            e12->setRobustKernel(rk1);
            rk1->setDelta(deltaHuber);
            optimizer.addEdge(e12);

            // Set edge x2 = S21*X1
            Eigen::Matrix<double,2,1> obs2;
            const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
            obs2 << kpUn2.pt.x, kpUn2.pt.y;

            g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

            e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
            e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            e21->setMeasurement(obs2);
            float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
            e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

            g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
            e21->setRobustKernel(rk2);
            rk2->setDelta(deltaHuber);
            optimizer.addEdge(e21);

            vpEdges12.push_back(e12);
            vpEdges21.push_back(e21);
            vnIndexEdge.push_back(i);
        }

        // Optimize!
        optimizer.initializeOptimization();
        optimizer.optimize(5);

        // Check inliers
        int nBad=0;
        for(size_t i=0; i<vpEdges12.size();i++)
        {
            g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
            g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
            if(!e12 || !e21)
                continue;

            if(e12->chi2()>th2 || e21->chi2()>th2)
            {
                size_t idx = vnIndexEdge[i];
                vpMatches1[idx]=static_cast<MapPoint*>(NULL);
                optimizer.removeEdge(e12);
                optimizer.removeEdge(e21);
                vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
                vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
                nBad++;
            }
        }

        int nMoreIterations;
        if(nBad>0)
            nMoreIterations=10;
        else
            nMoreIterations=5;

        if(nCorrespondences-nBad<10)
            return 0;

        // Optimize again only with inliers

        optimizer.initializeOptimization();
        optimizer.optimize(nMoreIterations);

        int nIn = 0;
        for(size_t i=0; i<vpEdges12.size();i++)
        {
            g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
            g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
            if(!e12 || !e21)
                continue;

            if(e12->chi2()>th2 || e21->chi2()>th2)
            {
                size_t idx = vnIndexEdge[i];
                vpMatches1[idx]=static_cast<MapPoint*>(NULL);
            }
            else
                nIn++;
        }

        // Recover optimized Sim3
        g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
        g2oS12= vSim3_recov->estimate();

        return nIn;
    }


} //namespace Planar_SLAM
