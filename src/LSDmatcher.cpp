//
// Created by lan on 17-12-26.
//
#include "LSDmatcher.h"

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace Planar_SLAM
{
    const int LSDmatcher::TH_HIGH = 100;
    const int LSDmatcher::TH_LOW = 50;

    LSDmatcher::LSDmatcher(float nnratio, bool checkOri):mfNNratio(nnratio), mbCheckOrientation(checkOri)
    {
    }

    int LSDmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono) {
        int nmatches = 0;

        const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0, 3).col(3);

        const cv::Mat twc = -Rcw.t()*tcw;

        const cv::Mat Rlw = LastFrame.mTcw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tlw = LastFrame.mTcw.rowRange(0, 3).col(3);

        const cv::Mat tlc = Rlw*twc+tlw;

        const bool bForward = tlc.at<float>(2)>CurrentFrame.mb && !bMono;
        const bool bBackward = -tlc.at<float>(2)>CurrentFrame.mb && !bMono;

        for (int i = 0; i < LastFrame.NL; i++) {
            MapLine *pML = LastFrame.mvpMapLines[i];

            if (!pML || pML->isBad() || LastFrame.mvbLineOutlier[i]) {
                continue;
            }

            Vector6d P = pML->GetWorldPos();

            cv::Mat SP = (Mat_<float>(3, 1) << P(0), P(1), P(2));
            cv::Mat EP = (Mat_<float>(3, 1) << P(3), P(4), P(5));

            const cv::Mat SPc = Rcw * SP + tcw;
            const auto &SPcX = SPc.at<float>(0);
            const auto &SPcY = SPc.at<float>(1);
            const auto &SPcZ = SPc.at<float>(2);

            const cv::Mat EPc = Rcw * EP + tcw;
            const auto &EPcX = EPc.at<float>(0);
            const auto &EPcY = EPc.at<float>(1);
            const auto &EPcZ = EPc.at<float>(2);

            if (SPcZ < 0.0f || EPcZ < 0.0f)
                continue;

            const float invz1 = 1.0f / SPcZ;
            const float u1 = CurrentFrame.fx * SPcX * invz1 + CurrentFrame.cx;
            const float v1 = CurrentFrame.fy * SPcY * invz1 + CurrentFrame.cy;

            if (u1 < CurrentFrame.mnMinX || u1 > CurrentFrame.mnMaxX)
                continue;
            if (v1 < CurrentFrame.mnMinY || v1 > CurrentFrame.mnMaxY)
                continue;

            const float invz2 = 1.0f / EPcZ;
            const float u2 = CurrentFrame.fx * EPcX * invz2 + CurrentFrame.cx;
            const float v2 = CurrentFrame.fy * EPcY * invz2 + CurrentFrame.cy;

            if (u2 < CurrentFrame.mnMinX || u2 > CurrentFrame.mnMaxX)
                continue;
            if (v2 < CurrentFrame.mnMinY || v2 > CurrentFrame.mnMaxY)
                continue;

            int nLastOctave = LastFrame.mvKeylinesUn[i].octave;

            float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

            vector<size_t> vIndices;

            if(bForward)
                vIndices = CurrentFrame.GetLinesInArea(u1, v1, u2, v2, radius, nLastOctave);
            else if(bBackward)
                vIndices = CurrentFrame.GetLinesInArea(u1, v1, u2, v2, radius, 0, nLastOctave);
            else
                vIndices = CurrentFrame.GetLinesInArea(u1, v1, u2, v2,radius, nLastOctave-1, nLastOctave+1);

            if(vIndices.empty())
                continue;

            const cv::Mat desc = pML->GetDescriptor();

            int bestDist=256;
            int bestLevel= -1;
            int bestDist2=256;
            int bestLevel2 = -1;
            int bestIdx =-1 ;

            for(unsigned long idx : vIndices)
            {
                if( CurrentFrame.mvpMapLines[idx])
                    if( CurrentFrame.mvpMapLines[idx]->Observations()>0)
                        continue;

                const cv::Mat &d =  CurrentFrame.mLdesc.row(idx);

                const int dist = DescriptorDistance(desc, d);

                if(dist<bestDist)
                {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestLevel2 = bestLevel;
                    bestLevel =  CurrentFrame.mvKeylinesUn[idx].octave;
                    bestIdx = idx;
                }
                else if(dist < bestDist2)
                {
                    bestLevel2 =  CurrentFrame.mvKeylinesUn[idx].octave;
                    bestDist2 = dist;
                }
            }

            if(bestDist <= TH_HIGH)
            {
                if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                    continue;

                CurrentFrame.mvpMapLines[bestIdx]=pML;
                nmatches++;
            }
        }

        return nmatches;
    }

    int LSDmatcher::SearchByProjection(Frame &F, const std::vector<MapLine *> &vpMapLines, const float th)
    {
        int nmatches = 0;

        const bool bFactor = th!=1.0;

        for(auto pML : vpMapLines)
        {
            if(!pML || pML->isBad() || !pML->mbTrackInView)
                continue;

            const int &nPredictLevel = pML->mnTrackScaleLevel;

            float r = RadiusByViewingCos(pML->mTrackViewCos);

            if(bFactor)
                r*=th;

            vector<size_t> vIndices =
                    F.GetLinesInArea(pML->mTrackProjX1, pML->mTrackProjY1, pML->mTrackProjX2, pML->mTrackProjY2,
                                     r*F.mvScaleFactors[nPredictLevel], nPredictLevel-1, nPredictLevel);

            if(vIndices.empty())
                continue;

            const cv::Mat MLdescriptor = pML->GetDescriptor();

            int bestDist=256;
            int bestLevel= -1;
            int bestDist2=256;
            int bestLevel2 = -1;
            int bestIdx =-1 ;

            for(unsigned long idx : vIndices)
            {
                if(F.mvpMapLines[idx])
                    if(F.mvpMapLines[idx]->Observations()>0)
                        continue;

                const cv::Mat &d = F.mLdesc.row(idx);

                const int dist = DescriptorDistance(MLdescriptor, d);

                // 根据描述子寻找描述子距离最小和次小的特征点
                if(dist<bestDist)
                {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestLevel2 = bestLevel;
                    bestLevel = F.mvKeylinesUn[idx].octave;
                    bestIdx = idx;
                }
                else if(dist < bestDist2)
                {
                    bestLevel2 = F.mvKeylinesUn[idx].octave;
                    bestDist2 = dist;
                }
            }

            // Apply ratio to second match (only if best and second are in the same scale level)
            if(bestDist <= TH_HIGH)
            {
                if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                    continue;

                F.mvpMapLines[bestIdx]=pML;
                nmatches++;
            }
        }
        return nmatches;
    }

    int LSDmatcher::SerachForInitialize(Frame &InitialFrame, Frame &CurrentFrame, vector<pair<int, int>> &LineMatches)
    {
        LineMatches.clear();
        int nmatches = 0;
        BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);
        Mat ldesc1, ldesc2;
        vector<vector<DMatch>> lmatches;
        ldesc1 = InitialFrame.mLdesc;
        ldesc2 = CurrentFrame.mLdesc;
        bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

        double nn_dist_th, nn12_dist_th;
        CurrentFrame.lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
        nn12_dist_th = nn12_dist_th*0.5;
        sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());
        for(int i=0; i<lmatches.size(); i++)
        {
            int qdx = lmatches[i][0].queryIdx;
            int tdx = lmatches[i][0].trainIdx;
            double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
            if(dist_12>nn12_dist_th)
            {
                LineMatches.push_back(make_pair(qdx, tdx));
                nmatches++;
            }
        }
        return nmatches;
    }

    int LSDmatcher::SearchByDescriptor(KeyFrame* pKF, Frame &currentF, vector<MapLine*> &vpMapLineMatches)
    {
        const vector<MapLine*> vpMapLinesKF = pKF->GetMapLineMatches();

        vpMapLineMatches = vector<MapLine*>(currentF.NL,static_cast<MapLine*>(NULL));

        int nmatches = 0;
        BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);
        Mat ldesc1, ldesc2;
        vector<vector<DMatch>> lmatches;
        ldesc1 = pKF->mLineDescriptors;
        ldesc2 = currentF.mLdesc;
        bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

        double nn_dist_th, nn12_dist_th;
        const float minRatio=1.0f/1.5f;
        currentF.lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
        nn12_dist_th = nn12_dist_th*0.5;
        sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());
        for(int i=0; i<lmatches.size(); i++)
        {
            int qdx = lmatches[i][0].queryIdx;
            int tdx = lmatches[i][0].trainIdx;
            double dist_12 = lmatches[i][0].distance/lmatches[i][1].distance;
            if(dist_12<minRatio)
            {
                MapLine* mapLine = vpMapLinesKF[qdx];

                if(mapLine)
                {
                    vpMapLineMatches[tdx]=mapLine;
                    nmatches++;
                }

            }
        }
        return nmatches;
    }

    int LSDmatcher::SearchByDescriptor(KeyFrame* pKF, KeyFrame *pKF2, vector<MapLine*> &vpMapLineMatches)
    {
        const vector<MapLine*> vpMapLinesKF = pKF->GetMapLineMatches();
        const vector<MapLine*> vpMapLinesKF2 = pKF2->GetMapLineMatches();

        vpMapLineMatches = vector<MapLine*>(vpMapLinesKF.size(),static_cast<MapLine*>(NULL));
        int nmatches = 0;
        BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);
        Mat ldesc1, ldesc2;
        vector<vector<DMatch>> lmatches;
        ldesc1 = pKF->mLineDescriptors;
        ldesc2 = pKF2->mLineDescriptors;
        bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

        double nn_dist_th, nn12_dist_th;
        pKF2->lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
        nn12_dist_th = nn12_dist_th*0.5;
        sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());
        for(int i=0; i<lmatches.size(); i++)
        {
            int qdx = lmatches[i][0].queryIdx;
            int tdx = lmatches[i][0].trainIdx;
            double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
            if(dist_12>nn12_dist_th)
            {
                MapLine* mapLine = vpMapLinesKF2[tdx];
                if(mapLine) {
                    vpMapLineMatches[qdx] = mapLine;
                    nmatches++;
                }
            }
        }
        return nmatches;
    }

    int LSDmatcher::DescriptorDistance(const Mat &a, const Mat &b)
    {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist=0;

        for(int i=0; i<8; i++, pa++, pb++)
        {
            unsigned  int v = *pa ^ *pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }

    int LSDmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2,
                                           vector<pair<size_t, size_t>> &vMatchedPairs)
    {
        vMatchedPairs.clear();
        int nmatches = 0;
        BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);
        Mat ldesc1, ldesc2;
        vector<vector<DMatch>> lmatches;
        ldesc1 = pKF1->mLineDescriptors;
        ldesc2 = pKF2->mLineDescriptors;
        bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

        double nn_dist_th, nn12_dist_th;
        pKF1->lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
        nn12_dist_th = nn12_dist_th*0.1;
        sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());
        for(int i=0; i<lmatches.size(); i++)
        {
            int qdx = lmatches[i][0].queryIdx;
            int tdx = lmatches[i][0].trainIdx;

            if (pKF1->GetMapLine(qdx) || pKF2->GetMapLine(tdx)) {
                continue;
            }

            double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
            if(dist_12>nn12_dist_th)
            {
                vMatchedPairs.push_back(make_pair(qdx, tdx));
                nmatches++;
            }
        }
        return nmatches;
    }

    float LSDmatcher::RadiusByViewingCos(const float &viewCos)
    {
        if(viewCos>0.998)
            return 5.0;
        else
            return 8.0;
    }

    int LSDmatcher::SearchByProjection(KeyFrame *pKF, cv::Mat Scw, const std::vector<MapLine *> &vpLines,
                                       std::vector<MapLine *> &vpMatched, int th) {
        // Get Calibration Parameters for later projection
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        // Decompose Scw
        cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
        const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
        cv::Mat Rcw = sRcw/scw;
        cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
        cv::Mat Ow = -Rcw.t()*tcw;

        // Set of MapLines already found in the KeyFrame
        set<MapLine*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(static_cast<MapLine*>(NULL));

        int nmatches=0;

        // For each Candidate MapLine Project and Match
        for(int iML=0, iendML=vpLines.size(); iML<iendML; iML++)
        {
            MapLine* pML = vpLines[iML];

            // Discard Bad MapLines and already found
            if(!pML || pML->isBad() || spAlreadyFound.count(pML))
                continue;


            Vector6d P = pML->GetWorldPos();

            cv::Mat SP = (Mat_<float>(3, 1) << P(0), P(1), P(2));
            cv::Mat EP = (Mat_<float>(3, 1) << P(3), P(4), P(5));

            const cv::Mat SPc = Rcw * SP + tcw;
            const auto &SPcX = SPc.at<float>(0);
            const auto &SPcY = SPc.at<float>(1);
            const auto &SPcZ = SPc.at<float>(2);

            const cv::Mat EPc = Rcw * EP + tcw;
            const auto &EPcX = EPc.at<float>(0);
            const auto &EPcY = EPc.at<float>(1);
            const auto &EPcZ = EPc.at<float>(2);

            if (SPcZ < 0.0f || EPcZ < 0.0f)
                continue;

            const float invz1 = 1.0f / SPcZ;
            const float u1 = fx * SPcX * invz1 + cx;
            const float v1 = fy * SPcY * invz1 + cy;

            if (u1 < pKF->mnMinX || u1 > pKF->mnMaxX)
                continue;
            if (v1 < pKF->mnMinY || v1 > pKF->mnMaxY)
                continue;

            const float invz2 = 1.0f / EPcZ;
            const float u2 = fx * EPcX * invz2 + cx;
            const float v2 = fy * EPcY * invz2 + cy;

            if (u2 < pKF->mnMinX || u2 > pKF->mnMaxX)
                continue;
            if (v2 < pKF->mnMinY || v2 > pKF->mnMaxY)
                continue;

            const float maxDistance = pML->GetMaxDistanceInvariance();
            const float minDistance = pML->GetMinDistanceInvariance();

            const cv::Mat OM = 0.5 * (SP + EP) - Ow;
            const float dist = cv::norm(OM);

            if (dist < minDistance || dist > maxDistance)
                continue;

            Vector3d Pn = pML->GetNormal();
            cv::Mat pn = (Mat_<float>(3, 1) << Pn(0), Pn(1), Pn(2));

            if(OM.dot(pn)<0.5*dist)
                continue;

            const int nPredictedLevel = pML->PredictScale(dist, pKF->mfLogScaleFactor);

            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetLinesInArea(u1,v1, u2, v2, radius);

            if(vIndices.empty())
                continue;

            const cv::Mat dML = pML->GetDescriptor();

            int bestDist=256;
            int bestIdx =-1 ;

            for(unsigned long idx : vIndices)
            {
                if(vpMatched[idx])
                    continue;

                const int &klLevel = pKF->mvKeyLines[idx].octave;

                if(klLevel<nPredictedLevel-1 || klLevel>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF->mLineDescriptors.row(idx);

                const int dist = DescriptorDistance(dML,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_LOW)
            {
                vpMatched[bestIdx]=pML;
                nmatches++;
            }
        }

        return nmatches;
    }

    int LSDmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapLine *> &vpMatches12, const float &s12,
                                 const cv::Mat &R12, const cv::Mat &t12, const float th) {
        const float &fx = pKF1->fx;
        const float &fy = pKF1->fy;
        const float &cx = pKF1->cx;
        const float &cy = pKF1->cy;

        // Camera 1 from world
        cv::Mat R1w = pKF1->GetRotation();
        cv::Mat t1w = pKF1->GetTranslation();

        //Camera 2 from world
        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();

        //Transformation between cameras
        cv::Mat sR12 = s12*R12;
        cv::Mat sR21 = (1.0/s12)*R12.t();
        cv::Mat t21 = -sR21*t12;

        const vector<MapLine*> vpMapLines1 = pKF1->GetMapLineMatches();
        const int N1 = vpMapLines1.size();

        const vector<MapLine*> vpMapLines2 = pKF2->GetMapLineMatches();
        const int N2 = vpMapLines2.size();

        vector<bool> vbAlreadyMatched1(N1,false);
        vector<bool> vbAlreadyMatched2(N2,false);

        for(int i=0; i<N1; i++)
        {
            MapLine* pML = vpMatches12[i];
            if(pML)
            {
                vbAlreadyMatched1[i]=true;
                int idx2 = pML->GetIndexInKeyFrame(pKF2);
                if(idx2>=0 && idx2<N2)
                    vbAlreadyMatched2[idx2]=true;
            }
        }

        vector<int> vnMatch1(N1,-1);
        vector<int> vnMatch2(N2,-1);

        // Transform from KF1 to KF2 and search
        for(int i1=0; i1<N1; i1++)
        {
            MapLine* pML = vpMapLines1[i1];

            if(!pML || pML->isBad() || vbAlreadyMatched1[i1])
                continue;

            Vector6d P = pML->GetWorldPos();

            cv::Mat SP = (Mat_<float>(3, 1) << P(0), P(1), P(2));
            cv::Mat EP = (Mat_<float>(3, 1) << P(3), P(4), P(5));

            const cv::Mat SPc1 = R1w * SP + t1w;
            const cv::Mat SPc2 = sR21 * SPc1 + t21;
            const auto &SPcX = SPc2.at<float>(0);
            const auto &SPcY = SPc2.at<float>(1);
            const auto &SPcZ = SPc2.at<float>(2);

            const cv::Mat EPc1 = R1w * EP + t1w;
            const cv::Mat EPc2 = sR21 * EPc1 + t21;
            const auto &EPcX = EPc2.at<float>(0);
            const auto &EPcY = EPc2.at<float>(1);
            const auto &EPcZ = EPc2.at<float>(2);

            if (SPcZ < 0.0f || EPcZ < 0.0f)
                continue;

            const float invz1 = 1.0f / SPcZ;
            const float u1 = fx * SPcX * invz1 + cx;
            const float v1 = fy * SPcY * invz1 + cy;

            if(!pKF2->IsInImage(u1,v1))
                continue;

            const float invz2 = 1.0f / EPcZ;
            const float u2 = fx * EPcX * invz2 + cx;
            const float v2 = fy * EPcY * invz2 + cy;

            if(!pKF2->IsInImage(u2,v2))
                continue;

            const float maxDistance = pML->GetMaxDistanceInvariance();
            const float minDistance = pML->GetMinDistanceInvariance();

            const float dist3D = cv::norm(0.5 * (SPc2 + EPc2));

            if (dist3D < minDistance || dist3D > maxDistance) {
                continue;
            }

            // Compute predicted octave
            const int nPredictedLevel = pML->PredictScale(dist3D, pKF2->mfLogScaleFactor);

            // Search in a radius
            const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF2->GetLinesInArea(u1,v1,u2,v2,radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dML = pML->GetDescriptor();

            int bestDist = INT_MAX;
            int bestIdx = -1;
            for(unsigned long idx : vIndices)
            {
                const int &klLevel = pKF2->mvKeyLines[idx].octave;

                if(klLevel<nPredictedLevel-1 || klLevel>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF2->mLineDescriptors.row(idx);

                const int dist = DescriptorDistance(dML, dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_HIGH)
            {
                vnMatch1[i1]=bestIdx;
            }
        }

        // Transform from KF2 to KF1 and search
        for(int i2=0; i2<N2; i2++)
        {
            MapLine* pML = vpMapLines2[i2];

            if(!pML || pML->isBad() || vbAlreadyMatched2[i2])
                continue;

            Vector6d P = pML->GetWorldPos();

            cv::Mat SP = (Mat_<float>(3, 1) << P(0), P(1), P(2));
            cv::Mat EP = (Mat_<float>(3, 1) << P(3), P(4), P(5));

            const cv::Mat SPc2 = R2w * SP + t2w;
            const cv::Mat SPc1 = sR12 * SPc2 + t12;
            const auto &SPcX = SPc1.at<float>(0);
            const auto &SPcY = SPc1.at<float>(1);
            const auto &SPcZ = SPc1.at<float>(2);

            const cv::Mat EPc2 = R2w * EP + t2w;
            const cv::Mat EPc1 = sR12 * EPc2 + t12;
            const auto &EPcX = EPc1.at<float>(0);
            const auto &EPcY = EPc1.at<float>(1);
            const auto &EPcZ = EPc1.at<float>(2);

            if (SPcZ < 0.0f || EPcZ < 0.0f)
                continue;

            const float invz1 = 1.0f / SPcZ;
            const float u1 = fx * SPcX * invz1 + cx;
            const float v1 = fy * SPcY * invz1 + cy;

            if(!pKF1->IsInImage(u1,v1))
                continue;

            const float invz2 = 1.0f / EPcZ;
            const float u2 = fx * EPcX * invz2 + cx;
            const float v2 = fy * EPcY * invz2 + cy;

            if(!pKF1->IsInImage(u2,v2))
                continue;

            const float maxDistance = pML->GetMaxDistanceInvariance();
            const float minDistance = pML->GetMinDistanceInvariance();

            const float dist3D = cv::norm(0.5 * (SPc1 + EPc1));

            if (dist3D < minDistance || dist3D > maxDistance)
                continue;

            // Compute predicted octave
            const int nPredictedLevel = pML->PredictScale(dist3D, pKF1->mfLogScaleFactor);

            // Search in a radius
            const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF1->GetLinesInArea(u1,v1,u2,v2,radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dML = pML->GetDescriptor();

            int bestDist = INT_MAX;
            int bestIdx = -1;
            for(unsigned long idx : vIndices)
            {
                const int &klLevel = pKF1->mvKeyLines[idx].octave;

                if(klLevel<nPredictedLevel-1 || klLevel>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF1->mLineDescriptors.row(idx);

                const int dist = DescriptorDistance(dML, dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_HIGH)
            {
                vnMatch2[i2]=bestIdx;
            }
        }

        // Check agreement
        int nFound = 0;

        for(int i1=0; i1<N1; i1++)
        {
            int idx2 = vnMatch1[i1];

            if(idx2>=0)
            {
                int idx1 = vnMatch2[idx2];
                if(idx1==i1)
                {
                    vpMatches12[i1] = vpMapLines2[idx2];
                    nFound++;
                }
            }
        }

        return nFound;
    }

    int LSDmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapLine *> &vpLines, float th,
                         vector<MapLine *> &vpReplaceLine) {
        // Get Calibration Parameters for later projection
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        // Decompose Scw
        cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
        const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
        cv::Mat Rcw = sRcw/scw;
        cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
        cv::Mat Ow = -Rcw.t()*tcw;

        // Set of MapPoints already found in the KeyFrame
        const set<MapLine*> spAlreadyFound = pKF->GetMapLines();

        int nFused=0;

        const int nLines = vpLines.size();

        // For each candidate MapPoint project and match
        for(int iML=0; iML<nLines; iML++)
        {
            MapLine* pML = vpLines[iML];

            // Discard Bad MapPoints and already found
            if(!pML || pML->isBad() || spAlreadyFound.count(pML))
                continue;

            Vector6d P = pML->GetWorldPos();

            cv::Mat SP = (Mat_<float>(3, 1) << P(0), P(1), P(2));
            cv::Mat EP = (Mat_<float>(3, 1) << P(3), P(4), P(5));

            const cv::Mat SPc = Rcw * SP + tcw;
            const auto &SPcX = SPc.at<float>(0);
            const auto &SPcY = SPc.at<float>(1);
            const auto &SPcZ = SPc.at<float>(2);

            const cv::Mat EPc = Rcw * EP + tcw;
            const auto &EPcX = EPc.at<float>(0);
            const auto &EPcY = EPc.at<float>(1);
            const auto &EPcZ = EPc.at<float>(2);

            if (SPcZ < 0.0f || EPcZ < 0.0f)
                continue;

            const float invz1 = 1.0f / SPcZ;
            const float u1 = fx * SPcX * invz1 + cx;
            const float v1 = fy * SPcY * invz1 + cy;

            if (u1 < pKF->mnMinX || u1 > pKF->mnMaxX)
                continue;
            if (v1 < pKF->mnMinY || v1 > pKF->mnMaxY)
                continue;

            const float invz2 = 1.0f / EPcZ;
            const float u2 = fx * EPcX * invz2 + cx;
            const float v2 = fy * EPcY * invz2 + cy;

            if (u2 < pKF->mnMinX || u2 > pKF->mnMaxX)
                continue;
            if (v2 < pKF->mnMinY || v2 > pKF->mnMaxY)
                continue;

            const float maxDistance = pML->GetMaxDistanceInvariance();
            const float minDistance = pML->GetMinDistanceInvariance();

            const cv::Mat OM = 0.5 * (SP + EP) - Ow;
            const float dist = cv::norm(OM);

            if (dist < minDistance || dist > maxDistance)
                continue;

            Vector3d Pn = pML->GetNormal();
            cv::Mat pn = (Mat_<float>(3, 1) << Pn(0), Pn(1), Pn(2));

            if(OM.dot(pn)<0.5*dist)
                continue;

            const int nPredictedLevel = pML->PredictScale(dist, pKF->mfLogScaleFactor);

            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetLinesInArea(u1,v1, u2, v2, radius);

            if(vIndices.empty())
                continue;

            const cv::Mat dML = pML->GetDescriptor();

            int bestDist=INT_MAX;
            int bestIdx =-1 ;

            for(unsigned long idx : vIndices)
            {
                const int &klLevel = pKF->mvKeyLines[idx].octave;

                if(klLevel<nPredictedLevel-1 || klLevel>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF->mLineDescriptors.row(idx);

                const int dist = DescriptorDistance(dML,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_LOW)
            {
                MapLine* pMLinKF = pKF->GetMapLine(bestIdx);
                if(pMLinKF)
                {
                    if(!pMLinKF->isBad())
                        vpReplaceLine[iML] = pMLinKF;
                }
                else
                {
                    pML->AddObservation(pKF,bestIdx);
                    pKF->AddMapLine(pML,bestIdx);
                }
                nFused++;
            }
        }

        return nFused;
    }

    int LSDmatcher::Fuse(KeyFrame *pKF, const vector<MapLine *> &vpMapLines, const float th)
    {
        cv::Mat Rcw = pKF->GetRotation();
        cv::Mat tcw = pKF->GetTranslation();

        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;
        const float &bf = pKF->mbf;

        cv::Mat Ow = pKF->GetCameraCenter();

        int nFused=0;

        const int nLines = vpMapLines.size();

        // For each candidate MapPoint project and match
        for(int iML=0; iML<nLines; iML++)
        {
            MapLine* pML = vpMapLines[iML];

            // Discard Bad MapLines and already found
            if(!pML || pML->isBad())
                continue;

            Vector6d P = pML->GetWorldPos();

            cv::Mat SP = (Mat_<float>(3, 1) << P(0), P(1), P(2));
            cv::Mat EP = (Mat_<float>(3, 1) << P(3), P(4), P(5));

            const cv::Mat SPc = Rcw * SP + tcw;
            const auto &SPcX = SPc.at<float>(0);
            const auto &SPcY = SPc.at<float>(1);
            const auto &SPcZ = SPc.at<float>(2);

            const cv::Mat EPc = Rcw * EP + tcw;
            const auto &EPcX = EPc.at<float>(0);
            const auto &EPcY = EPc.at<float>(1);
            const auto &EPcZ = EPc.at<float>(2);

            if (SPcZ < 0.0f || EPcZ < 0.0f)
                continue;

            const float invz1 = 1.0f / SPcZ;
            const float u1 = fx * SPcX * invz1 + cx;
            const float v1 = fy * SPcY * invz1 + cy;

            if (u1 < pKF->mnMinX || u1 > pKF->mnMaxX)
                continue;
            if (v1 < pKF->mnMinY || v1 > pKF->mnMaxY)
                continue;

            const float invz2 = 1.0f / EPcZ;
            const float u2 = fx * EPcX * invz2 + cx;
            const float v2 = fy * EPcY * invz2 + cy;

            if (u2 < pKF->mnMinX || u2 > pKF->mnMaxX)
                continue;
            if (v2 < pKF->mnMinY || v2 > pKF->mnMaxY)
                continue;

            const float maxDistance = pML->GetMaxDistanceInvariance();
            const float minDistance = pML->GetMinDistanceInvariance();

            const cv::Mat OM = 0.5 * (SP + EP) - Ow;
            const float dist = cv::norm(OM);

            if (dist < minDistance || dist > maxDistance)
                continue;

            Vector3d Pn = pML->GetNormal();
            cv::Mat pn = (Mat_<float>(3, 1) << Pn(0), Pn(1), Pn(2));

            if(OM.dot(pn)<0.5*dist)
                continue;

            const int nPredictedLevel = pML->PredictScale(dist, pKF->mfLogScaleFactor);

            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetLinesInArea(u1,v1, u2, v2, radius);

            if(vIndices.empty())
                continue;

            const cv::Mat dML = pML->GetDescriptor();

            int bestDist=INT_MAX;
            int bestIdx =-1 ;

            for(unsigned long idx : vIndices)
            {
                const int &klLevel = pKF->mvKeyLines[idx].octave;

                if(klLevel<nPredictedLevel-1 || klLevel>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF->mLineDescriptors.row(idx);

                const int dist = DescriptorDistance(dML,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_LOW)
            {
                MapLine* pMLinKF = pKF->GetMapLine(bestIdx);
                if(pMLinKF)
                {
                    if(!pMLinKF->isBad()) {
                        if(pMLinKF->Observations()>pML->Observations())
                            pML->Replace(pMLinKF);
                        else
                            pMLinKF->Replace(pML);
                    }
                }
                else
                {
                    pML->AddObservation(pKF,bestIdx);
                    pKF->AddMapLine(pML,bestIdx);
                }
                nFused++;
            }
        }

        return nFused;
    }

//    int LSDmatcher::Fuse(KeyFrame *pKF, const vector<MapLine *> &vpMapLines)
//    {
//        cv::Mat Rcw = pKF->GetRotation();
//        cv::Mat tcw = pKF->GetTranslation();
//
//        const float &fx = pKF->fx;
//        const float &fy = pKF->fy;
//        const float &cx = pKF->cx;
//        const float &cy = pKF->cy;
//        const float &bf = pKF->mbf;
//
//        cv::Mat Ow = pKF->GetCameraCenter();
//
//        int nFused=0;
//
//        Mat lineDesc = pKF->mLineDescriptors;   //待Fuse的关键帧的描述子
//        const int nMLs = vpMapLines.size();
//
//        //遍历所有的MapLines
//        for(int i=0; i<nMLs; i++)
//        {
//            MapLine* pML = vpMapLines[i];
//
//            if(!pML)
//                continue;
//
//            if(pML->isBad() || pML->IsInKeyFrame(pKF))
//                continue;
//#if 0
//            Vector6d LineW = pML->GetWorldPos();
//            cv::Mat LineSW = (Mat_<float>(3,1) << LineW(0), LineW(1), LineW(2));
//            cv::Mat LineSC = Rcw*LineSW + tcw;
//            cv::Mat LineEW = (Mat_<float>(3,1) << LineW(3), LineW(4), LineW(5));
//            cv::Mat LineEC = Rcw*LineEW + tcw;
//
//            //Depth must be positive
//            if(LineSC.at<float>(2)<0.0f || LineEC.at<float>(2)<0.0f)
//                continue;
//
//            // 获取起始点在图像上的投影坐标
//            const float invz1 = 1/LineSC.at<float>(2);
//            const float x1 = LineSC.at<float>(0)*invz1;
//            const float y1 = LineSC.at<float>(1)*invz1;
//
//            const float u1 = fx*x1 + cx;
//            const float v1 = fy*y1 + cy;
//
//            // 获取终止点在图像上的投影坐标
//            const float invz2 = 1/LineEC.at<float>(2);
//            const float x2 = LineEC.at<float>(0)*invz2;
//            const float y2 = LineEC.at<float>(1)*invz2;
//
//            const float u2 = fx*x2 + cx;
//            const float v2 = fy*y2 + cy;
//#endif
//            Mat CurrentLineDesc = pML->mLDescriptor;        //MapLine[i]对应的线特征描述子
//
//#if 0
//            // 采用暴力匹配法,knnMatch
//            BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);
//            vector<vector<DMatch>> lmatches;
//            bfm->knnMatch(CurrentLineDesc, lineDesc, lmatches, 2);
//            double nn_dist_th, nn12_dist_th;
//            pKF->lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);
//            nn12_dist_th = nn12_dist_th*0.1;
//            sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());
//            for(int i=0; i<lmatches.size(); i++)
//            {
//                int tdx = lmatches[i][0].trainIdx;
//                double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
//                if(dist_12>nn12_dist_th)    //找到了pKF中对应ML
//                {
//                    MapLine* pMLinKF = pKF->GetMapLine(tdx);
//                    if(pMLinKF)
//                    {
//                        if(!pMLinKF->isBad())
//                        {
//                            if(pMLinKF->Observations()>pML->Observations())
//                                pML->Replace(pMLinKF);
//                            else
//                                pMLinKF->Replace(pML);
//                        }
//                    }
//                    nFused++;
//                }
//            }
//#elif 1
//            // 使用暴力匹配法
//            Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
//            vector<DMatch> lmatches;
//            matcher->match ( CurrentLineDesc, lineDesc, lmatches );
//
//            double max_dist = 0;
//            double min_dist = 100;
//
//            //-- Quick calculation of max and min distances between keypoints
//            for( int i = 0; i < CurrentLineDesc.rows; i++ )
//            {
//                double dist = lmatches[i].distance;
//                if( dist < min_dist ) min_dist = dist;
//                if( dist > max_dist ) max_dist = dist;
//            }
//
//            // "good" matches (i.e. whose distance is less than 2*min_dist )
//            std::vector< DMatch > good_matches;
//            for( int i = 0; i < CurrentLineDesc.rows; i++ )
//            {
//                if( lmatches[i].distance < 1.5*min_dist )
//                {
//                    int tdx = lmatches[i].trainIdx;
//                    MapLine* pMLinKF = pKF->GetMapLine(tdx);
//                    if(pMLinKF)
//                    {
//                        if(!pMLinKF->isBad())
//                        {
//                            if(pMLinKF->Observations()>pML->Observations())
//                                pML->Replace(pMLinKF);
//                            else
//                                pMLinKF->Replace(pML);
//                        }
//                    }
//                    nFused++;
//                }
//            }
//
//#else
//            cout << "CurrentLineDesc.empty() = " << CurrentLineDesc.empty() << endl;
//            cout << "lineDesc.empty() = " << lineDesc.empty() << endl;
//            cout << CurrentLineDesc << endl;
//            if(CurrentLineDesc.empty() || lineDesc.empty())
//                continue;
//
//            // 采用Flann方法
//            FlannBasedMatcher flm;
//            vector<DMatch> lmatches;
//            flm.match(CurrentLineDesc, lineDesc, lmatches);
//
//            double max_dist = 0;
//            double min_dist = 100;
//
//            //-- Quick calculation of max and min distances between keypoints
//            cout << "CurrentLineDesc.rows = " << CurrentLineDesc.rows << endl;
//            for( int i = 0; i < CurrentLineDesc.rows; i++ )
//            { double dist = lmatches[i].distance;
//                if( dist < min_dist ) min_dist = dist;
//                if( dist > max_dist ) max_dist = dist;
//            }
//
//            // "good" matches (i.e. whose distance is less than 2*min_dist )
//            std::vector< DMatch > good_matches;
//            for( int i = 0; i < CurrentLineDesc.rows; i++ )
//            {
//                if( lmatches[i].distance < 2*min_dist )
//                {
//                    int tdx = lmatches[i].trainIdx;
//                    MapLine* pMLinKF = pKF->GetMapLine(tdx);
//                    if(pMLinKF)
//                    {
//                        if(!pMLinKF->isBad())
//                        {
//                            if(pMLinKF->Observations()>pML->Observations())
//                                pML->Replace(pMLinKF);
//                            else
//                                pMLinKF->Replace(pML);
//                        }
//                    }
//                    nFused++;
//                }
//            }
//#endif
//        }
//        return nFused;
//    }
}
