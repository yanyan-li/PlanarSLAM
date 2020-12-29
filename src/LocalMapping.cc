#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace Planar_SLAM {

    LocalMapping::LocalMapping(Map *pMap, const float bMonocular) :
            mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
            mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true) {
    }

    void LocalMapping::SetLoopCloser(LoopClosing *pLoopCloser) {
        mpLoopCloser = pLoopCloser;
    }

    void LocalMapping::SetTracker(Tracking *pTracker) {
        mpTracker = pTracker;
    }

    void LocalMapping::Run() {

        mbFinished = false;

        while (1) {
            // Tracking will see that Local Mapping is busy
            SetAcceptKeyFrames(false);
            // Check if there are keyframes in the queue
            if (CheckNewKeyFrames()) {
                // BoW conversion and insertion in Map
                // VI-A keyframe insertion
                ProcessNewKeyFrame();

                // Check recent MapPoints
                // VI-B recent map points culling
                thread threadCullPoint(&LocalMapping::MapPointCulling, this);
                thread threadCullLine(&LocalMapping::MapLineCulling, this);
                thread threadCullPlane(&LocalMapping::MapPlaneCulling, this);
                threadCullPoint.join();
                threadCullLine.join();
                threadCullPlane.join();

                // Triangulate new MapPoints
                // VI-C new map points creation

                thread threadCreateP(&LocalMapping::CreateNewMapPoints, this);
                thread threadCreateL(&LocalMapping::CreateNewMapLines2, this);
                threadCreateP.join();
                threadCreateL.join();

                if (!CheckNewKeyFrames()) {
                    // Find more matches in neighbor keyframes and fuse point duplications
                    SearchInNeighbors();
                }

                mbAbortBA = false;

                if (!CheckNewKeyFrames() && !stopRequested()) {
                    // VI-D Local BA
                    if (mpMap->KeyFramesInMap() > 2) {
                        //Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, &mbAbortBA, mpMap);
                    }


                    // Check redundant local Keyframes
                    // VI-E local keyframes culling
                    KeyFrameCulling();
                }

                mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
            } else if (Stop()) {
                // Safe area to stop
                while (isStopped() && !CheckFinish()) {
                    usleep(3000);
                }
                if (CheckFinish())
                    break;
            }

            ResetIfRequested();

            // Tracking will see that Local Mapping is busy
            SetAcceptKeyFrames(true);

            if (CheckFinish())
                break;

            usleep(3000);
        }

        SetFinish();
    }

    void LocalMapping::InsertKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexNewKFs);
        mlNewKeyFrames.push_back(pKF);
        mbAbortBA = true;
    }


    bool LocalMapping::CheckNewKeyFrames() {
        unique_lock<mutex> lock(mMutexNewKFs);
        return (!mlNewKeyFrames.empty());
    }

    void LocalMapping::ProcessNewKeyFrame() {
        {
            unique_lock<mutex> lock(mMutexNewKFs);
            mpCurrentKeyFrame = mlNewKeyFrames.front();
            mlNewKeyFrames.pop_front();
        }

        // Compute Bags of Words structures
        mpCurrentKeyFrame->ComputeBoW();

        // Associate MapPoints to the new keyframe and update normal and descriptor
        const vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

        for (size_t i = 0; i < vpMapPointMatches.size(); i++) {
            MapPoint *pMP = vpMapPointMatches[i];
            if (pMP) {
                if (!pMP->isBad()) {
                    if (!pMP->IsInKeyFrame(mpCurrentKeyFrame)) {
                        pMP->AddObservation(mpCurrentKeyFrame, i);
                        pMP->UpdateNormalAndDepth();
                        pMP->ComputeDistinctiveDescriptors();
                    } else // this can only happen for new stereo points inserted by the Tracking
                    {
                        mlpRecentAddedMapPoints.push_back(pMP);
                    }
                }
            }
        }

        const vector<MapLine *> vpMapLineMatches = mpCurrentKeyFrame->GetMapLineMatches();

        for (size_t i = 0; i < vpMapLineMatches.size(); i++) {
            MapLine *pML = vpMapLineMatches[i];
            if (pML) {
                if (!pML->isBad()) {
                    if (!pML->IsInKeyFrame(mpCurrentKeyFrame)) {
                        pML->AddObservation(mpCurrentKeyFrame, i);
                        pML->UpdateAverageDir();
                        pML->ComputeDistinctiveDescriptors();
                    } else {
                        mlpRecentAddedMapLines.push_back(pML);
                    }
                }
            }
        }

        const vector<MapPlane *> vpMapPlaneMatches = mpCurrentKeyFrame->GetMapPlaneMatches();

        for (size_t i = 0; i < vpMapPlaneMatches.size(); i++) {
            MapPlane *pMP = vpMapPlaneMatches[i];
            if (pMP && !pMP->isBad() && pMP->mnFirstKFid == mpCurrentKeyFrame->mnId) {
                mlpRecentAddedMapPlanes.push_back(pMP);
            }
        }

        // Update links in the Covisibility Graph
        mpCurrentKeyFrame->UpdateConnections();

        // Insert Keyframe in Map
        mpMap->AddKeyFrame(mpCurrentKeyFrame);
    }

    void LocalMapping::MapPointCulling() {
        // Check Recent Added MapPoints
        list<MapPoint *>::iterator lit = mlpRecentAddedMapPoints.begin();
        const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

        int nThObs;
        if (mbMonocular)
            nThObs = 2;
        else
            nThObs = 3;
        const int cnThObs = nThObs;

        while (lit != mlpRecentAddedMapPoints.end()) {
            MapPoint *pMP = *lit;
            if (pMP->isBad()) {
                lit = mlpRecentAddedMapPoints.erase(lit);
            } else if (pMP->GetFoundRatio() < 0.25f) {
                pMP->SetBadFlag();
                lit = mlpRecentAddedMapPoints.erase(lit);
            } else if (((int) nCurrentKFid - (int) pMP->mnFirstKFid) >= 2 && pMP->Observations() <= cnThObs) {
                pMP->SetBadFlag();
                lit = mlpRecentAddedMapPoints.erase(lit);
            } else if (((int) nCurrentKFid - (int) pMP->mnFirstKFid) >= 3)
                lit = mlpRecentAddedMapPoints.erase(lit);
            else
                lit++;
        }
    }

    void LocalMapping::MapLineCulling() {
        // Check Recent Added MapLines
        list<MapLine *>::iterator lit = mlpRecentAddedMapLines.begin();
        const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

        int nThObs;
        if (mbMonocular)
            nThObs = 2;
        else
            nThObs = 3;
        const int cnThObs = nThObs;

        while (lit != mlpRecentAddedMapLines.end()) {
            MapLine *pML = *lit;
            if (pML->isBad()) {
                lit = mlpRecentAddedMapLines.erase(lit);
            } else if (pML->GetFoundRatio() < 0.25f) {
                pML->SetBadFlag();
                lit = mlpRecentAddedMapLines.erase(lit);
            } else if (((int) nCurrentKFid - (int) pML->mnFirstKFid) >= 2 && pML->Observations() <= cnThObs) {
                pML->SetBadFlag();
                lit = mlpRecentAddedMapLines.erase(lit);
            } else if (((int) nCurrentKFid - (int) pML->mnFirstKFid) >= 3)
                lit = mlpRecentAddedMapLines.erase(lit);
            else
                lit++;
        }
    }

    void LocalMapping::MapPlaneCulling() {
        auto dTh = Config::Get<double>("Plane.AssociationDisRef");
        auto aTh = Config::Get<double>("Plane.AssociationAngRef");

        vector<MapPlane *> vpMapPlanes = mpMap->GetAllMapPlanes();

        for (int i = 0; i < vpMapPlanes.size(); i++) {
            int bestIdx = -1;
            float ldTh = dTh;
            for (int j = i + 1; j < vpMapPlanes.size(); j++) {
                MapPlane *pMP1 = vpMapPlanes[i];
                MapPlane *pMP2 = vpMapPlanes[j];

                if (pMP1 && pMP2 && !pMP1->isBad() && !pMP2->isBad()) {
                    cv::Mat p3Dw1 = pMP1->GetWorldPos();
                    cv::Mat p3Dw2 = pMP2->GetWorldPos();

                    float angle = p3Dw1.at<float>(0, 0) * p3Dw2.at<float>(0, 0) +
                                  p3Dw1.at<float>(1, 0) * p3Dw2.at<float>(1, 0) +
                                  p3Dw1.at<float>(2, 0) * p3Dw2.at<float>(2, 0);

                    // associate plane
                    if (angle > aTh || angle < -aTh) {
                        double dis = mpMap->PointDistanceFromPlane(p3Dw1, pMP2->mvPlanePoints);
                        if (dis < ldTh) {
                            ldTh = dis;
                            bestIdx = j;
                        }
                    }
                }
            }

            if (bestIdx != -1) {
                MapPlane *pMP1 = vpMapPlanes[i];
                MapPlane *pMP2 = vpMapPlanes[bestIdx];

                cout << "Match! Plane1: " << pMP1->GetWorldPos().t() << ", Plane2: " << pMP2->GetWorldPos().t() << endl;

                if (pMP1->Observations() > pMP2->Observations())
                    pMP2->Replace(pMP1);
                else
                    pMP1->Replace(pMP2);
            }
        }

        // Check Recent Added MapPlanes
        list<MapPlane *>::iterator lit = mlpRecentAddedMapPlanes.begin();
        const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

        int nThObs;
        if (mbMonocular)
            nThObs = 2;
        else
            nThObs = 2;
        const int cnThObs = nThObs;

        while (lit != mlpRecentAddedMapPlanes.end()) {
            MapPlane *pMP = *lit;
            if (pMP->isBad()) {
                lit = mlpRecentAddedMapPlanes.erase(lit);
            }
            else if (pMP->GetFoundRatio() < 0.25f) {
                pMP->SetBadFlag();
                lit = mlpRecentAddedMapPlanes.erase(lit);
            }
            else if (((int) nCurrentKFid - (int) pMP->mnFirstKFid) >= 2 && pMP->Observations() <= cnThObs) {
                pMP->SetBadFlag();
                lit = mlpRecentAddedMapPlanes.erase(lit);
            }
            else if (((int) nCurrentKFid - (int) pMP->mnFirstKFid) >= 3)
                lit = mlpRecentAddedMapPlanes.erase(lit);
            else
                lit++;
        }
    }

    void LocalMapping::CreateNewMapPoints() {
        // Retrieve neighbor keyframes in covisibility graph
        int nn = 10;
        if (mbMonocular)
            nn = 20;

        const vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

        ORBmatcher matcher(0.6, false);

        cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
        cv::Mat Rwc1 = Rcw1.t();
        cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
        cv::Mat Tcw1(3, 4, CV_32F);
        Rcw1.copyTo(Tcw1.colRange(0, 3));
        tcw1.copyTo(Tcw1.col(3));

        cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

        const float &fx1 = mpCurrentKeyFrame->fx;
        const float &fy1 = mpCurrentKeyFrame->fy;
        const float &cx1 = mpCurrentKeyFrame->cx;
        const float &cy1 = mpCurrentKeyFrame->cy;
        const float &invfx1 = mpCurrentKeyFrame->invfx;
        const float &invfy1 = mpCurrentKeyFrame->invfy;

        const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

        int nnew = 0;

        // Search matches with epipolar restriction and triangulate
        for (size_t i = 0; i < vpNeighKFs.size(); i++) {
            if (i > 0 && CheckNewKeyFrames())
                return;

            KeyFrame *pKF2 = vpNeighKFs[i];

            // Check first that baseline is not too short
            cv::Mat Ow2 = pKF2->GetCameraCenter();
            cv::Mat vBaseline = Ow2 - Ow1;
            const float baseline = cv::norm(vBaseline);

            if (!mbMonocular) {
                if (baseline < pKF2->mb)
                    continue;
            } else {
                const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
                const float ratioBaselineDepth = baseline / medianDepthKF2;

                if (ratioBaselineDepth < 0.01)
                    continue;
            }

            // Compute Fundamental Matrix
            cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

            // Search matches that fullfil epipolar constraint
            vector<pair<size_t, size_t> > vMatchedIndices;
            matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, F12, vMatchedIndices, false);

            cv::Mat Rcw2 = pKF2->GetRotation();
            cv::Mat Rwc2 = Rcw2.t();
            cv::Mat tcw2 = pKF2->GetTranslation();
            cv::Mat Tcw2(3, 4, CV_32F);
            Rcw2.copyTo(Tcw2.colRange(0, 3));
            tcw2.copyTo(Tcw2.col(3));

            const float &fx2 = pKF2->fx;
            const float &fy2 = pKF2->fy;
            const float &cx2 = pKF2->cx;
            const float &cy2 = pKF2->cy;
            const float &invfx2 = pKF2->invfx;
            const float &invfy2 = pKF2->invfy;

            // Triangulate each match
            const int nmatches = vMatchedIndices.size();
            for (int ikp = 0; ikp < nmatches; ikp++) {
                // step6.1：取出匹配的特征点
                const int &idx1 = vMatchedIndices[ikp].first;
                const int &idx2 = vMatchedIndices[ikp].second;

                const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
                const float kp1_ur = mpCurrentKeyFrame->mvuRight[idx1];
                bool bStereo1 = kp1_ur >= 0;

                const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
                const float kp2_ur = pKF2->mvuRight[idx2];
                bool bStereo2 = kp2_ur >= 0;

                // Check parallax between rays
                cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
                cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);

                cv::Mat ray1 = Rwc1 * xn1;
                cv::Mat ray2 = Rwc2 * xn2;
                const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

                float cosParallaxStereo = cosParallaxRays + 1;
                float cosParallaxStereo1 = cosParallaxStereo;
                float cosParallaxStereo2 = cosParallaxStereo;

                if (bStereo1)
                    cosParallaxStereo1 = cos(2 * atan2(mpCurrentKeyFrame->mb / 2, mpCurrentKeyFrame->mvDepth[idx1]));
                else if (bStereo2)
                    cosParallaxStereo2 = cos(2 * atan2(pKF2->mb / 2, pKF2->mvDepth[idx2]));

                cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

                cv::Mat x3D;
                if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 &&
                    (bStereo1 || bStereo2 || cosParallaxRays < 0.9998)) {
                    // Linear Triangulation Method
                    cv::Mat A(4, 4, CV_32F);
                    A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                    A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                    A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                    A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                    cv::Mat w, u, vt;
                    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                    x3D = vt.row(3).t();

                    if (x3D.at<float>(3) == 0)
                        continue;

                    // Euclidean coordinates
                    x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);

                } else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2) {
                    x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);
                } else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1) {
                    x3D = pKF2->UnprojectStereo(idx2);
                } else
                    continue; //No stereo and very low parallax

                cv::Mat x3Dt = x3D.t();

                //Check triangulation in front of cameras
                float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
                if (z1 <= 0)
                    continue;

                float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
                if (z2 <= 0)
                    continue;

                //Check reprojection error in first keyframe
                const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
                const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
                const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
                const float invz1 = 1.0 / z1;

                if (!bStereo1) {
                    float u1 = fx1 * x1 * invz1 + cx1;
                    float v1 = fy1 * y1 * invz1 + cy1;
                    float errX1 = u1 - kp1.pt.x;
                    float errY1 = v1 - kp1.pt.y;
                    if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1)    //5.991是基于卡方检验计算出的阈值，假设测量有一个像素的偏差
                        continue;
                } else {
                    float u1 = fx1 * x1 * invz1 + cx1;
                    float u1_r = u1 - mpCurrentKeyFrame->mbf * invz1;
                    float v1 = fy1 * y1 * invz1 + cy1;
                    float errX1 = u1 - kp1.pt.x;
                    float errY1 = v1 - kp1.pt.y;
                    float errX1_r = u1_r - kp1_ur;
                    if ((errX1 * errX1 + errY1 * errY1 + errX1_r * errX1_r) > 7.8 * sigmaSquare1)
                        continue;
                }

                //Check reprojection error in second keyframe
                const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
                const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
                const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
                const float invz2 = 1.0 / z2;
                if (!bStereo2) {

                    float u2 = fx2 * x2 * invz2 + cx2;
                    float v2 = fy2 * y2 * invz2 + cy2;
                    float errX2 = u2 - kp2.pt.x;
                    float errY2 = v2 - kp2.pt.y;
                    if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2)
                        continue;
                } else {
                    float u2 = fx2 * x2 * invz2 + cx2;
                    float u2_r = u2 - mpCurrentKeyFrame->mbf * invz2;
                    float v2 = fy2 * y2 * invz2 + cy2;
                    float errX2 = u2 - kp2.pt.x;
                    float errY2 = v2 - kp2.pt.y;
                    float errX2_r = u2_r - kp2_ur;
                    if ((errX2 * errX2 + errY2 * errY2 + errX2_r * errX2_r) > 7.8 * sigmaSquare2)  ///7.8和上面的5.991联系？
                        continue;
                }

                //Check scale consistency
                cv::Mat normal1 = x3D - Ow1;
                float dist1 = cv::norm(normal1);

                cv::Mat normal2 = x3D - Ow2;
                float dist2 = cv::norm(normal2);

                if (dist1 == 0 || dist2 == 0)
                    continue;

                const float ratioDist = dist2 / dist1;
                const float ratioOctave =
                        mpCurrentKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];

                if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor)
                    continue;

                // Triangulation is succesfull
                MapPoint *pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpMap);

                pMP->AddObservation(mpCurrentKeyFrame, idx1);
                pMP->AddObservation(pKF2, idx2);

                mpCurrentKeyFrame->AddMapPoint(pMP, idx1);
                pKF2->AddMapPoint(pMP, idx2);

                pMP->ComputeDistinctiveDescriptors();

                pMP->UpdateNormalAndDepth();

                mpMap->AddMapPoint(pMP);
                mlpRecentAddedMapPoints.push_back(pMP);

                nnew++;
            }
        }
    }

    void LocalMapping::CreateNewMapLines1() {
        // Retrieve neighbor keyframes in covisibility graph
        int nn = 10;
        if (mbMonocular)
            nn = 20;
        //step1：
        const vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

        LSDmatcher lmatcher;

        cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
        cv::Mat Rwc1 = Rcw1.t();
        cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
        cv::Mat Tcw1(3, 4, CV_32F);
        Rcw1.copyTo(Tcw1.colRange(0, 3));
        tcw1.copyTo(Tcw1.col(3));

        cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

        const float &fx1 = mpCurrentKeyFrame->fx;
        const float &fy1 = mpCurrentKeyFrame->fy;
        const float &cx1 = mpCurrentKeyFrame->cx;
        const float &cy1 = mpCurrentKeyFrame->cy;
        const float &invfx1 = mpCurrentKeyFrame->invfx;
        const float &invfy1 = mpCurrentKeyFrame->invfy;

        const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

        int nnew = 0;

        // Search matches with epipolar restriction and triangulate
        for (size_t i = 0; i < vpNeighKFs.size(); i++) {
            if (i > 0 && CheckNewKeyFrames())
                return;

            KeyFrame *pKF2 = vpNeighKFs[i];

            // Check first that baseline is not too short
            cv::Mat Ow2 = pKF2->GetCameraCenter();
            cv::Mat vBaseline = Ow2 - Ow1;
            const float baseline = cv::norm(vBaseline);

            // step3：
            if (!mbMonocular) {
                //
                if (baseline < pKF2->mb)
                    continue;
            } else {
                //
                const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
                // baseline
                const float ratioBaselineDepth = baseline / medianDepthKF2;
                //
                if (ratioBaselineDepth < 0.01)
                    continue;
            }

            // Compute Fundamental Matrix
            // step4：
            cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

            // Search matches that fulfill epipolar constraint
            // step5：
            vector<pair<size_t, size_t>> vMatchedIndices;
            lmatcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, vMatchedIndices);

            cv::Mat Rcw2 = pKF2->GetRotation();
            cv::Mat Rwc2 = Rcw2.t();
            cv::Mat tcw2 = pKF2->GetTranslation();
            cv::Mat Tcw2(3, 4, CV_32F);
            Rcw2.copyTo(Tcw2.colRange(0, 3));
            tcw2.copyTo(Tcw2.col(3));

            const float &fx2 = pKF2->fx;
            const float &fy2 = pKF2->fy;
            const float &cx2 = pKF2->cx;
            const float &cy2 = pKF2->cy;
            const float &invfx2 = pKF2->invfx;
            const float &invfy2 = pKF2->invfy;

            // Triangulate each match
            // step6：
            const int nmatches = vMatchedIndices.size();
            for (int ikl = 0; ikl < nmatches; ikl++) {
                // step6.1：
                const int &idx1 = vMatchedIndices[ikl].first;
                const int &idx2 = vMatchedIndices[ikl].second;

                const KeyLine &keyline1 = mpCurrentKeyFrame->mvKeyLines[idx1];
                const KeyLine &keyline2 = pKF2->mvKeyLines[idx2];
                const Vector3d keyline2_function = pKF2->mvKeyLineFunctions[idx2];

                //
                Point2f midP1 = keyline1.pt;
                Point2f midP2 = keyline2.pt;
                // step6.2:
                //
                cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (midP1.x - cx1) * invfx1, (midP1.y - cy1) * invfy1, 1.0);
                cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (midP2.x - cx2) * invfx2, (midP2.y - cy2) * invfy2, 1.0);

                //
                cv::Mat ray1 = Rwc1 * xn1;
                cv::Mat ray2 = Rwc2 * xn2;
                float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));
//            cosParallaxRays = cosParallaxRays + 1;

                // step6.3：
                cv::Mat StartC1, EndC1, StartC2, EndC2;
                StartC1 = (cv::Mat_<float>(3, 1) << (keyline1.startPointX - cx1) * invfx1,
                        (keyline1.startPointY - cy1) * invfy1, 1.0);
                EndC1 = (cv::Mat_<float>(3, 1) << (keyline1.endPointX - cx1) * invfx1, (keyline1.endPointY - cy1) *
                                                                                       invfy1, 1.0);
                StartC2 = (cv::Mat_<float>(3, 1) << (keyline2.startPointX - cx2) * invfx2,
                        (keyline2.startPointY - cy2) * invfy2, 1.0);
                EndC2 = (cv::Mat_<float>(3, 1) << (keyline2.endPointX - cx2) * invfx2, (keyline2.endPointY - cy2) *
                                                                                       invfy2, 1.0);

                // step6.4：
                cv::Mat s3D, e3D;
                if (cosParallaxRays > 0 && cosParallaxRays < 0.9998) {
                    cv::Mat A(4, 4, CV_32F);
                    A.row(0) = StartC1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                    A.row(1) = StartC1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                    A.row(2) = StartC2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                    A.row(3) = StartC2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                    cv::Mat w1, u1, vt1;
                    cv::SVD::compute(A, w1, u1, vt1, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                    s3D = vt1.row(3).t();

                    if (s3D.at<float>(3) == 0)
                        continue;

                    // Euclidean coordinates
                    s3D = s3D.rowRange(0, 3) / s3D.at<float>(3);

                    cv::Mat B(4, 4, CV_32F);
                    B.row(0) = EndC1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                    B.row(1) = EndC1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                    B.row(2) = EndC2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                    B.row(3) = EndC2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                    cv::Mat w2, u2, vt2;
                    cv::SVD::compute(B, w2, u2, vt2, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                    e3D = vt2.row(3).t();

                    if (e3D.at<float>(3) == 0)
                        continue;

                    // Euclidean coordinates
                    e3D = e3D.rowRange(0, 3) / e3D.at<float>(3);
                } else
                    continue;

                cv::Mat s3Dt = s3D.t();
                cv::Mat e3Dt = e3D.t();

                // step6.5：
                float SZC1 = Rcw1.row(2).dot(s3Dt) + tcw1.at<float>(2);   //起始点在C1下的Z坐标值
                if (SZC1 <= 0)
                    continue;

                float SZC2 = Rcw2.row(2).dot(s3Dt) + tcw2.at<float>(2);   //起始点在C2下的Z坐标值
                if (SZC2 <= 0)
                    continue;

                float EZC1 = Rcw1.row(2).dot(e3Dt) + tcw1.at<float>(2);   //终止点在C1下的Z坐标值
                if (EZC1 <= 0)
                    continue;

                float EZC2 = Rcw2.row(2).dot(e3Dt) + tcw2.at<float>(2);   //终止点在C2下的Z坐标值
                if (EZC2 <= 0)
                    continue;

                // step6.6：
                const float &sigmaSquare2 = mpCurrentKeyFrame->mvLevelSigma2[keyline1.octave];
                //
                Vector3d lC1 = mpCurrentKeyFrame->mvKeyLineFunctions[idx1];

                const float x1 = Rcw1.row(0).dot(s3Dt) + tcw1.at<float>(0);
                const float y1 = Rcw1.row(1).dot(s3Dt) + tcw1.at<float>(1);
                float e1 = lC1(0) * x1 + lC1(1) * y1 + lC1(2);

                const float x2 = Rcw1.row(0).dot(e3Dt) + tcw1.at<float>(0);
                const float y2 = Rcw1.row(1).dot(e3Dt) + tcw1.at<float>(1);
                float e2 = lC1(0) * x2 + lC1(1) * y2 + lC1(2);

                float eC1 = e1 + e2;
                if (eC1 > 7.8 * sigmaSquare2)    ///Q:7.8是仿照CreateMapPoints()函数中的双目来的，这里需要重新计算
                    continue;

                // step6.7：
                // -1.
                Vector3d lC2 = pKF2->mvKeyLineFunctions[idx2];

                // -2.
                const float x3 = Rcw2.row(0).dot(s3Dt) + tcw2.at<float>(0);
                const float y3 = Rcw2.row(1).dot(s3Dt) + tcw2.at<float>(1);
                float e3 = lC2(0) * x3 + lC2(1) * y3 + lC2(2);

                // -3.
                const float x4 = Rcw2.row(0).dot(e3Dt) + tcw2.at<float>(0);
                const float y4 = Rcw2.row(1).dot(e3Dt) + tcw2.at<float>(1);
                float e4 = lC2(0) * x4 + lC2(1) * y3 + lC2(2);

                // -4.
                float eC2 = e3 + e4;
                if (eC1 > 7.8 * sigmaSquare2)
                    continue;

                // step6.8:
                cv::Mat middle3D = 0.5 * (s3D + e3D);

                cv::Mat normal1 = middle3D - Ow1;
                float dist1 = cv::norm(normal1);

                cv::Mat normal2 = middle3D - Ow2;
                float dist2 = cv::norm(normal2);

                if (dist1 == 0 || dist2 == 0)
                    continue;

                const float ratioDist = dist2 / dist1;

                const float ratioOctave =
                        mpCurrentKeyFrame->mvScaleFactors[keyline1.octave] / pKF2->mvScaleFactors[keyline2.octave];
                if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor)
                    continue;

                // step6.9:

                Vector6d line3D;
                line3D << s3D.at<float>(0), s3D.at<float>(1), s3D.at<float>(2), e3D.at<float>(0), e3D.at<float>(
                        1), e3D.at<float>(2);
                MapLine *pML = new MapLine(line3D, mpCurrentKeyFrame, mpMap);

                // step6.10：
                pML->AddObservation(mpCurrentKeyFrame, idx1);
                pML->AddObservation(pKF2, idx2);

                mpCurrentKeyFrame->AddMapLine(pML, idx1);
                pKF2->AddMapLine(pML, idx2);

                pML->ComputeDistinctiveDescriptors();
                pML->UpdateAverageDir();
                mpMap->AddMapLine(pML);

                // step6.11：
                mlpRecentAddedMapLines.push_back(pML);

                nnew++;
            }
        }

    }

    void LocalMapping::CreateNewMapLines2() {
        // Retrieve neighbor keyframes in covisibility graph
        int nn = 10;
        if (mbMonocular)
            nn = 20;

        const vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

        LSDmatcher lmatcher;

        cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
        cv::Mat Rwc1 = Rcw1.t();
        cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
        cv::Mat Tcw1(3, 4, CV_32F);
        Rcw1.copyTo(Tcw1.colRange(0, 3));
        tcw1.copyTo(Tcw1.col(3));

        cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

        const Mat &K1 = mpCurrentKeyFrame->mK;
        const float &fx1 = mpCurrentKeyFrame->fx;
        const float &fy1 = mpCurrentKeyFrame->fy;
        const float &cx1 = mpCurrentKeyFrame->cx;
        const float &cy1 = mpCurrentKeyFrame->cy;
        const float &invfx1 = mpCurrentKeyFrame->invfx;
        const float &invfy1 = mpCurrentKeyFrame->invfy;

        const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

        int nnew = 0;

        // Search matches with epipolar restriction and triangulate
        for (size_t i = 0; i < vpNeighKFs.size(); i++) {
            if (i > 0 && CheckNewKeyFrames())
                return;

            KeyFrame *pKF2 = vpNeighKFs[i];

            // Check first that baseline is not too short
            cv::Mat Ow2 = pKF2->GetCameraCenter();
            cv::Mat vBaseline = Ow2 - Ow1;
            const float baseline = cv::norm(vBaseline);

            if (!mbMonocular) {
                if (baseline < pKF2->mb)
                    continue;
            } else {
                const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
                const float ratioBaselineDepth = baseline / medianDepthKF2;
                if (ratioBaselineDepth < 0.01)
                    continue;
            }

            // Compute Fundamental Matrix
            cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

            // Search matches that fulfill epipolar constraint
            vector<pair<size_t, size_t>> vMatchedIndices;
            lmatcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, vMatchedIndices);

            cv::Mat Rcw2 = pKF2->GetRotation();
            cv::Mat Rwc2 = Rcw2.t();
            cv::Mat tcw2 = pKF2->GetTranslation();
            cv::Mat Tcw2(3, 4, CV_32F);
            Rcw2.copyTo(Tcw2.colRange(0, 3));
            tcw2.copyTo(Tcw2.col(3));

            const Mat &K2 = pKF2->mK;
            const float &fx2 = pKF2->fx;
            const float &fy2 = pKF2->fy;
            const float &cx2 = pKF2->cx;
            const float &cy2 = pKF2->cy;
            const float &invfx2 = pKF2->invfx;
            const float &invfy2 = pKF2->invfy;

            // Triangulate each matched line Segment
            const int nmatches = vMatchedIndices.size();
            for (int ikl = 0; ikl < nmatches; ikl++) {
                const int &idx1 = vMatchedIndices[ikl].first;
                const int &idx2 = vMatchedIndices[ikl].second;

                const KeyLine &kl1 = mpCurrentKeyFrame->mvKeyLines[idx1];
                bool bStereo1 = mpCurrentKeyFrame->mvDepthLine[idx1] > 0;

                const KeyLine &kl2 = pKF2->mvKeyLines[idx2];
                bool bStereo2 = mpCurrentKeyFrame->mvDepthLine[idx2] > 0;

                cv::Mat kl1sp, kl1ep, kl2sp, kl2ep;
                kl1sp = (cv::Mat_<float>(3, 1) << (kl1.startPointX - cx1) * invfx1, (kl1.startPointY - cy1) *
                                                                                    invfy1, 1.0);
                kl1ep = (cv::Mat_<float>(3, 1) << (kl1.endPointX - cx1) * invfx1, (kl1.endPointY - cy1) * invfy1, 1.0);
                kl2sp = (cv::Mat_<float>(3, 1) << (kl2.startPointX - cx2) * invfx2, (kl2.startPointY - cy2) *
                                                                                    invfy2, 1.0);
                kl2ep = (cv::Mat_<float>(3, 1) << (kl2.endPointX - cx2) * invfx2, (kl2.endPointY - cy2) * invfy2, 1.0);

                cv::Mat sp3D, ep3D;
                if (bStereo1) {
                    Vector6d line3D = mpCurrentKeyFrame->obtain3DLine(idx1);
                    sp3D = cv::Mat::eye(3, 1, CV_32F);
                    ep3D = cv::Mat::eye(3, 1, CV_32F);
                    sp3D.at<float>(0) = line3D(0);
                    sp3D.at<float>(1) = line3D(1);
                    sp3D.at<float>(2) = line3D(2);
                    ep3D.at<float>(0) = line3D(3);
                    ep3D.at<float>(1) = line3D(4);
                    ep3D.at<float>(2) = line3D(5);
                } else if (bStereo2) {
                    Vector6d line3D = pKF2->obtain3DLine(idx2);
                    sp3D = cv::Mat::eye(3, 1, CV_32F);
                    ep3D = cv::Mat::eye(3, 1, CV_32F);
                    sp3D.at<float>(0) = line3D(0);
                    sp3D.at<float>(1) = line3D(1);
                    sp3D.at<float>(2) = line3D(2);
                    ep3D.at<float>(0) = line3D(3);
                    ep3D.at<float>(1) = line3D(4);
                    ep3D.at<float>(2) = line3D(5);
                } else
                    continue; //No stereo and very low parallax

                cv::Mat sp3Dt = sp3D.t();
                cv::Mat ep3Dt = ep3D.t();


                //Check triangulation in front of cameras
                float zsp1 = Rcw1.row(2).dot(sp3Dt) + tcw1.at<float>(2);
                if (zsp1 <= 0)
                    continue;

                float zep1 = Rcw1.row(2).dot(ep3Dt) + tcw1.at<float>(2);
                if (zep1 <= 0)
                    continue;

                float zsp2 = Rcw2.row(2).dot(sp3Dt) + tcw2.at<float>(2);
                if (zsp2 <= 0)
                    continue;

                float zep2 = Rcw2.row(2).dot(ep3Dt) + tcw2.at<float>(2);
                if (zep2 <= 0)
                    continue;

                //Check reprojection error in first keyframe
                const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kl1.octave];
                const float xsp1 = Rcw1.row(0).dot(sp3Dt) + tcw1.at<float>(0);
                const float ysp1 = Rcw1.row(1).dot(sp3Dt) + tcw1.at<float>(1);
                const float invzsp1 = 1.0 / zsp1;

                float usp1 = fx1 * xsp1 * invzsp1 + cx1;
                float vsp1 = fy1 * ysp1 * invzsp1 + cy1;

                float errXsp1 = usp1 - kl1.startPointX;
                float errYsp1 = vsp1 - kl1.startPointY;
                if ((errXsp1 * errXsp1 + errYsp1 * errYsp1) > 5.991 * sigmaSquare1)
                    continue;

                const float xep1 = Rcw1.row(0).dot(ep3Dt) + tcw1.at<float>(0);
                const float yep1 = Rcw1.row(1).dot(ep3Dt) + tcw1.at<float>(1);
                const float invzep1 = 1.0 / zep1;

                float uep1 = fx1 * xep1 * invzep1 + cx1;
                float vep1 = fy1 * yep1 * invzep1 + cy1;

                float errXep1 = uep1 - kl1.endPointX;
                float errYep1 = vep1 - kl1.endPointY;
                if ((errXep1 * errXep1 + errYep1 * errYep1) > 5.991 * sigmaSquare1)
                    continue;

                //Check reprojection error in second keyframe
                const float sigmaSquare2 = pKF2->mvLevelSigma2[kl2.octave];
                const float xsp2 = Rcw2.row(0).dot(sp3Dt) + tcw2.at<float>(0);
                const float ysp2 = Rcw2.row(1).dot(sp3Dt) + tcw2.at<float>(1);
                const float invzsp2 = 1.0 / zsp2;

                float usp2 = fx2 * xsp2 * invzsp2 + cx2;
                float vsp2 = fy2 * ysp2 * invzsp2 + cy2;
                float errXsp2 = usp2 - kl2.startPointX;
                float errYsp2 = vsp2 - kl2.startPointY;
                if ((errXsp2 * errXsp2 + errYsp2 * errYsp2) > 5.991 * sigmaSquare2)
                    continue;

                const float xep2 = Rcw2.row(0).dot(ep3Dt) + tcw2.at<float>(0);
                const float yep2 = Rcw2.row(1).dot(ep3Dt) + tcw2.at<float>(1);
                const float invzep2 = 1.0 / zep2;

                float uep2 = fx2 * xep2 * invzep2 + cx2;
                float vep2 = fy2 * yep2 * invzep2 + cy2;
                float errXep2 = uep2 - kl2.endPointX;
                float errYep2 = vep2 - kl2.endPointY;
                if ((errXep2 * errXep2 + errYep2 * errYep2) > 5.991 * sigmaSquare2)
                    continue;

                //Check scale consistency
                cv::Mat normalsp1 = sp3D - Ow1;
                float distsp1 = cv::norm(normalsp1);

                cv::Mat normalep1 = ep3D - Ow1;
                float distep1 = cv::norm(normalep1);

                cv::Mat normalsp2 = sp3D - Ow2;
                float distsp2 = cv::norm(normalsp2);

                cv::Mat normalep2 = ep3D - Ow2;
                float distep2 = cv::norm(normalep2);

                if (distsp1 == 0 || distep1 == 0 || distsp2 == 0 || distep2 == 0)
                    continue;

                const float ratioDistsp = distsp2 / distsp1;
                const float ratioDistep = distep2 / distep1;
                const float ratioOctave =
                        mpCurrentKeyFrame->mvScaleFactors[kl1.octave] / pKF2->mvScaleFactors[kl2.octave];

                if (ratioDistsp * ratioFactor < ratioOctave || ratioDistsp > ratioOctave * ratioFactor ||
                    ratioDistep * ratioFactor < ratioOctave || ratioDistep > ratioOctave * ratioFactor)
                    continue;

                Vector6d line3D;
                line3D << sp3Dt.at<float>(0), sp3Dt.at<float>(1), sp3Dt.at<float>(2), ep3Dt.at<float>(0),
                        ep3Dt.at<float>(1), ep3Dt.at<float>(2);

                MapLine *pML = new MapLine(line3D, mpCurrentKeyFrame, mpMap);

                pML->AddObservation(mpCurrentKeyFrame, idx1);
                pML->AddObservation(pKF2, idx2);

                mpCurrentKeyFrame->AddMapLine(pML, idx1);
                pKF2->AddMapLine(pML, idx2);

                pML->ComputeDistinctiveDescriptors();
                pML->UpdateAverageDir();
                mpMap->AddMapLine(pML);

                mlpRecentAddedMapLines.push_back(pML);

                nnew++;
            }

        }
    }

    void LocalMapping::SearchInNeighbors() {
        // Retrieve neighbor keyframes
        int nn = 10;
        if (mbMonocular)
            nn = 20;
        const vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
        vector<KeyFrame *> vpTargetKFs;
        for (auto pKFi : vpNeighKFs) {
            if (pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi);
            pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

            // Extend to some second neighbors
            const vector<KeyFrame *> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
            for (auto pKFi2 : vpSecondNeighKFs) {
                if (pKFi2->isBad() || pKFi2->mnFuseTargetForKF == mpCurrentKeyFrame->mnId ||
                    pKFi2->mnId == mpCurrentKeyFrame->mnId)
                    continue;
                vpTargetKFs.push_back(pKFi2);
            }
        }

        // Search matches by projection from current KF in target KFs
        ORBmatcher matcher;
        vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
        for (auto pKFi : vpTargetKFs) {
            matcher.Fuse(pKFi, vpMapPointMatches);
        }

        // Search matches by projection from target KFs in current KF
        vector<MapPoint *> vpFuseCandidates;
        vpFuseCandidates.reserve(vpTargetKFs.size() * vpMapPointMatches.size());

        for (auto pKFi : vpTargetKFs) {
            vector<MapPoint *> vpMapPointsKFi = pKFi->GetMapPointMatches();

            for (auto pMP : vpMapPointsKFi) {
                if (!pMP)
                    continue;
                if (pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                    continue;
                pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
                vpFuseCandidates.push_back(pMP);
            }
        }

        matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);


        // Update points
        vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
        for (auto pMP : vpMapPointMatches) {
            if (pMP) {
                if (!pMP->isBad()) {
                    pMP->ComputeDistinctiveDescriptors();
                    pMP->UpdateNormalAndDepth();
                }
            }
        }

        LSDmatcher lineMatcher;
        vector<MapLine *> vpMapLineMatches = mpCurrentKeyFrame->GetMapLineMatches();
        for (auto pKFi : vpTargetKFs) {
            lineMatcher.Fuse(pKFi, vpMapLineMatches);
        }

        vector<MapLine *> vpLineFuseCandidates;
        vpLineFuseCandidates.reserve(vpTargetKFs.size() * vpMapLineMatches.size());

        for (auto pKFi : vpTargetKFs) {
            vector<MapLine *> vpMapLinesKFi = pKFi->GetMapLineMatches();

            for (auto pML : vpMapLinesKFi) {
                if (!pML)
                    continue;

                if (pML->isBad() || pML->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                    continue;

                pML->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
                vpLineFuseCandidates.push_back(pML);
            }
        }

        lineMatcher.Fuse(mpCurrentKeyFrame, vpLineFuseCandidates);

        // Update Lines
        vpMapLineMatches = mpCurrentKeyFrame->GetMapLineMatches();
        for (auto pML : vpMapLineMatches) {
            if (pML) {
                if (!pML->isBad()) {
                    pML->ComputeDistinctiveDescriptors();
                    pML->UpdateAverageDir();
                }
            }
        }

        // Update connections in covisibility graph
        mpCurrentKeyFrame->UpdateConnections();
    }

    cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2) {
        cv::Mat R1w = pKF1->GetRotation();
        cv::Mat t1w = pKF1->GetTranslation();
        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();

        cv::Mat R12 = R1w * R2w.t();
        cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

        cv::Mat t12x = SkewSymmetricMatrix(t12);

        const cv::Mat &K1 = pKF1->mK;
        const cv::Mat &K2 = pKF2->mK;


        return K1.t().inv() * t12x * R12 * K2.inv();
    }

    void LocalMapping::RequestStop() {
        unique_lock<mutex> lock(mMutexStop);
        mbStopRequested = true;
        unique_lock<mutex> lock2(mMutexNewKFs);
        mbAbortBA = true;
    }

    bool LocalMapping::Stop() {
        unique_lock<mutex> lock(mMutexStop);
        if (mbStopRequested && !mbNotStop) {
            mbStopped = true;
            cout << "Local Mapping STOP" << endl;
            return true;
        }

        return false;
    }

    bool LocalMapping::isStopped() {
        unique_lock<mutex> lock(mMutexStop);
        return mbStopped;
    }

    bool LocalMapping::stopRequested() {
        unique_lock<mutex> lock(mMutexStop);
        return mbStopRequested;
    }

    void LocalMapping::Release() {
        unique_lock<mutex> lock(mMutexStop);
        unique_lock<mutex> lock2(mMutexFinish);
        if (mbFinished)
            return;
        mbStopped = false;
        mbStopRequested = false;
        for (list<KeyFrame *>::iterator lit = mlNewKeyFrames.begin(), lend = mlNewKeyFrames.end(); lit != lend; lit++)
            delete *lit;
        mlNewKeyFrames.clear();

        cout << "Local Mapping RELEASE" << endl;
    }

    bool LocalMapping::AcceptKeyFrames() {
        unique_lock<mutex> lock(mMutexAccept);
        return mbAcceptKeyFrames;
    }

    void LocalMapping::SetAcceptKeyFrames(bool flag) {
        unique_lock<mutex> lock(mMutexAccept);
        mbAcceptKeyFrames = flag;
    }

    bool LocalMapping::SetNotStop(bool flag) {
        unique_lock<mutex> lock(mMutexStop);

        if (flag && mbStopped)
            return false;

        mbNotStop = flag;

        return true;
    }

    void LocalMapping::InterruptBA() {
        mbAbortBA = true;
    }

    void LocalMapping::KeyFrameCulling() {
        // Check redundant keyframes (only local keyframes)
        // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
        // in at least other 3 keyframes (in the same or finer scale)
        // We only consider close stereo points
        if (mpCurrentKeyFrame->mbNewPlane)
            return;

        vector<KeyFrame *> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

        for (vector<KeyFrame *>::iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end();
             vit != vend; vit++) {
            KeyFrame *pKF = *vit;
            if (pKF->mnId == 0)
                continue;
            const vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();

            int nObs = 3;
            const int thObs = nObs;
            int nRedundantObservations = 0;
            int nMPs = 0;
            for (size_t i = 0, iend = vpMapPoints.size(); i < iend; i++) {
                MapPoint *pMP = vpMapPoints[i];
                if (pMP) {
                    if (!pMP->isBad()) {
                        if (!mbMonocular) {
                            if (pKF->mvDepth[i] > pKF->mThDepth || pKF->mvDepth[i] < 0)
                                continue;
                        }

                        nMPs++;
                        if (pMP->Observations() > thObs) {
                            const int &scaleLevel = pKF->mvKeysUn[i].octave;
                            const map<KeyFrame *, size_t> observations = pMP->GetObservations();
                            int nObs = 0;
                            for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end();
                                 mit != mend; mit++) {
                                KeyFrame *pKFi = mit->first;
                                if (pKFi == pKF)
                                    continue;
                                const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                                if (scaleLeveli <= scaleLevel + 1) {
                                    nObs++;
                                    if (nObs >= thObs)
                                        break;
                                }
                            }
                            if (nObs >= thObs) {
                                nRedundantObservations++;
                            }
                        }
                    }
                }
            }

            if (nRedundantObservations > 0.9 * nMPs)
                pKF->SetBadFlag();
        }
    }

    cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v) {
        return (cv::Mat_<float>(3, 3) << 0, -v.at<float>(2), v.at<float>(1),
                v.at<float>(2), 0, -v.at<float>(0),
                -v.at<float>(1), v.at<float>(0), 0);
    }

    void LocalMapping::RequestReset() {
        {
            unique_lock<mutex> lock(mMutexReset);
            mbResetRequested = true;
        }

        while (1) {
            {
                unique_lock<mutex> lock2(mMutexReset);
                if (!mbResetRequested)
                    break;
            }
            usleep(3000);
        }
    }

    void LocalMapping::ResetIfRequested() {
        unique_lock<mutex> lock(mMutexReset);
        if (mbResetRequested) {
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();
            mlpRecentAddedMapLines.clear();
            mbResetRequested = false;
        }
    }

    void LocalMapping::RequestFinish() {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }

    bool LocalMapping::CheckFinish() {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

    void LocalMapping::SetFinish() {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinished = true;
        unique_lock<mutex> lock2(mMutexStop);
        mbStopped = true;
    }

    bool LocalMapping::isFinished() {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinished;
    }

} //namespace Planar_SLAM
