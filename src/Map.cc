#include "Map.h"

#include<mutex>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace Planar_SLAM {

    Map::Map() : mnMaxKFid(0), mnBigChangeIdx(0) {
    }

    void Map::AddKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexMap);
        mspKeyFrames.insert(pKF);
        if (pKF->mnId > mnMaxKFid)
            mnMaxKFid = pKF->mnId;
    }

    void Map::AddMapPoint(MapPoint *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPoints.insert(pMP);
    }

    void Map::EraseMapPoint(MapPoint *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPoints.erase(pMP);

        // TODO: This only erase the pointer.
        // Delete the MapPoint
    }

    void Map::EraseKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexMap);
        mspKeyFrames.erase(pKF);

        // TODO: This only erase the pointer.
        // Delete the MapPoint
    }

    void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs) {
        unique_lock<mutex> lock(mMutexMap);
        mvpReferenceMapPoints = vpMPs;
    }

    void Map::InformNewBigChange() {
        unique_lock<mutex> lock(mMutexMap);
        mnBigChangeIdx++;
    }

    int Map::GetLastBigChangeIdx() {
        unique_lock<mutex> lock(mMutexMap);
        return mnBigChangeIdx;
    }

    vector<KeyFrame *> Map::GetAllKeyFrames() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<KeyFrame *>(mspKeyFrames.begin(), mspKeyFrames.end());
    }

    vector<MapPoint *> Map::GetAllMapPoints() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapPoint *>(mspMapPoints.begin(), mspMapPoints.end());
    }

    long unsigned int Map::MapPointsInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return mspMapPoints.size();
    }

    long unsigned int Map::KeyFramesInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return mspKeyFrames.size();
    }

    vector<MapPoint *> Map::GetReferenceMapPoints() {
        unique_lock<mutex> lock(mMutexMap);
        return mvpReferenceMapPoints;
    }

    long unsigned int Map::GetMaxKFid() {
        unique_lock<mutex> lock(mMutexMap);
        return mnMaxKFid;
    }

    void Map::clear() {
        for (auto mspMapPoint : mspMapPoints)
            delete mspMapPoint;
        for (auto mspMapLine : mspMapLines)
            delete mspMapLine;
        for (auto mspMapPlane : mspMapPlanes)
            delete mspMapPlane;

        for (auto mspKeyFrame : mspKeyFrames)
            delete mspKeyFrame;

        mspMapPlanes.clear();
        mspMapPoints.clear();
        mspKeyFrames.clear();
        mspMapLines.clear();
        mnMaxKFid = 0;
        mvpReferenceMapPoints.clear();
        mvpReferenceMapLines.clear();
        mvpKeyFrameOrigins.clear();
    }

    void Map::AddMapLine(MapLine *pML) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapLines.insert(pML);
    }

    void Map::EraseMapLine(MapLine *pML) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapLines.erase(pML);
    }

    //
    void Map::SetReferenceMapLines(const std::vector<MapLine *> &vpMLs) {
        unique_lock<mutex> lock(mMutexMap);
        mvpReferenceMapLines = vpMLs;
    }

    vector<MapLine *> Map::GetAllMapLines() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapLine *>(mspMapLines.begin(), mspMapLines.end());
    }

    vector<MapLine *> Map::GetReferenceMapLines() {
        unique_lock<mutex> lock(mMutexMap);
        return mvpReferenceMapLines;
    }

    long unsigned int Map::MapLinesInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return mspMapLines.size();
    }

    void Map::AddMapPlane(MapPlane *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPlanes.insert(pMP);
    }

    void Map::EraseMapPlane(MapPlane *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPlanes.erase(pMP);
    }

    vector<MapPlane *> Map::GetAllMapPlanes() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapPlane *>(mspMapPlanes.begin(), mspMapPlanes.end());
    }

    long unsigned int Map::MapPlanesInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return mspMapPlanes.size();
    }

    cv::Mat Map::FindManhattan(Frame &pF, const float &verTh, bool out) {
        cv::Mat bestP1, bestP2;
        float lverTh = verTh;
        int maxSize = 0;

        if(out)
            cout << "Matching planes..." << endl;

        for (int i = 0; i < pF.mnPlaneNum; ++i) {
            cv::Mat p1 = pF.mvPlaneCoefficients[i];
            if(out)
                cout << " plane  " << i << ": " << endl;

            if(out)
                cout << " p1  " << p1.t() << ": " << endl;

            for (int j = i+1;j < pF.mnPlaneNum; ++j) {
                cv::Mat p2 = pF.mvPlaneCoefficients[j];

                float angle = p1.at<float>(0) * p2.at<float>(0) +
                              p1.at<float>(1) * p2.at<float>(1) +
                              p1.at<float>(2) * p2.at<float>(2);

                if(out)
                    cout << j << ", p2 : " << p2.t() << endl;

                if(out)
                    cout << j << ", angle : " << angle << endl;

                // vertical planes
                if (angle < lverTh && angle > -lverTh && (pF.mvPlanePoints[i].size() + pF.mvPlanePoints[j].size()) > maxSize) {
                    if(out)
                        cout << "  vertical!" << endl;
                    maxSize = pF.mvPlanePoints[i].size() + pF.mvPlanePoints[j].size();

                    if (bestP1.empty() || bestP2.empty()) {
                        bestP1 = cv::Mat::eye(cv::Size(1, 3), CV_32F);
                        bestP2 = cv::Mat::eye(cv::Size(1, 3), CV_32F);
                    }

                    bestP1.at<float>(0, 0) = p1.at<float>(0, 0);
                    bestP1.at<float>(1, 0) = p1.at<float>(1, 0);
                    bestP1.at<float>(2, 0) = p1.at<float>(2, 0);

                    bestP2.at<float>(0, 0) = p2.at<float>(0, 0);
                    bestP2.at<float>(1, 0) = p2.at<float>(1, 0);
                    bestP2.at<float>(2, 0) = p2.at<float>(2, 0);
                }
            }
        }

        if (bestP1.empty() || bestP2.empty()) {
            if(out)
                cout << "Matching planes and lines..." << endl;

            for (int i = 0; i < pF.mnPlaneNum; ++i) {
                cv::Mat p = pF.ComputePlaneWorldCoeff(i);
                if(out)
                    cout << " plane  " << i << ": " << endl;

                for (int j = 0; j < pF.mvLines3D.size(); ++j) {
                    Vector6d lineVector = pF.obtain3DLine(j);

                    cv::Mat startPoint = cv::Mat::eye(cv::Size(1, 3), CV_32F);
                    cv::Mat endPoint = cv::Mat::eye(cv::Size(1, 3), CV_32F);

                    startPoint.at<float>(0, 0) = lineVector[0];
                    startPoint.at<float>(1, 0) = lineVector[1];
                    startPoint.at<float>(2, 0) = lineVector[2];
                    endPoint.at<float>(0, 0) = lineVector[3];
                    endPoint.at<float>(1, 0) = lineVector[4];
                    endPoint.at<float>(2, 0) = lineVector[5];

                    cv::Mat line = startPoint - endPoint;
                    line /= cv::norm(line);

                    if(out)
                        cout << "line: " << line << endl;

                    float angle = p.at<float>(0, 0) * line.at<float>(0, 0) +
                                  p.at<float>(1, 0) * line.at<float>(1, 0) +
                                  p.at<float>(2, 0) * line.at<float>(2, 0);

                    if(out)
                        cout << j << ", angle : " << angle << endl;

                    if (angle < lverTh && angle > -lverTh) {
                        if(out)
                            cout << "  vertical!" << endl;
                        lverTh = abs(angle);

                        if (bestP1.empty() || bestP2.empty()) {
                            bestP1 = cv::Mat::eye(cv::Size(1, 3), CV_32F);
                            bestP2 = cv::Mat::eye(cv::Size(1, 3), CV_32F);
                        }

                        bestP1.at<float>(0, 0) = p.at<float>(0, 0);
                        bestP1.at<float>(1, 0) = p.at<float>(1, 0);
                        bestP1.at<float>(2, 0) = p.at<float>(2, 0);

                        bestP2.at<float>(0, 0) = line.at<float>(0, 0);
                        bestP2.at<float>(1, 0) = line.at<float>(1, 0);
                        bestP2.at<float>(2, 0) = line.at<float>(2, 0);
                    }
                }
            }
        }

        if(out)
            cout << "Matching done" << endl;

        cv::Mat Rotation_cm;
        Rotation_cm = cv::Mat::eye(cv::Size(3, 3), CV_32F);

        if (!bestP1.empty() && !bestP2.empty()) {

            int loc1;
            float max1 = 0;
            for (int i = 0; i < 3; i++) {
                float val = bestP1.at<float>(i);
                if (val < 0)
                    val = -val;
                if (val > max1) {
                    loc1 = i;
                    max1 = val;
                }
            }

            if (bestP1.at<float>(loc1) < 0) {
                bestP1 = -bestP1;
            }

            int loc2;
            float max2 = 0;
            for (int i = 0; i < 3; i++) {
                float val = bestP2.at<float>(i);
                if (val < 0)
                    val = -val;
                if (val > max2) {
                    loc2 = i;
                    max2 = val;
                }
            }

            if (bestP2.at<float>(loc2) < 0) {
                bestP2 = -bestP2;
            }

            cv::Mat p3;

            p3 = bestP1.cross(bestP2);

            int loc3;
            float max3 = 0;
            for (int i = 0; i < 3; i++) {
                float val = p3.at<float>(i);
                if (val < 0)
                    val = -val;
                if (val > max3) {
                    loc3 = i;
                    max3 = val;
                }
            }

            if (p3.at<float>(loc3) < 0) {
                p3 = -p3;
            }

            if(out) {
                cout << "p1: " << bestP1 << endl;
                cout << "p2: " << bestP2 << endl;
                cout << "p3: " << p3 << endl;
            }

            cv::Mat first, second, third;

            std::map<int, cv::Mat> sort;
            sort[loc1] = bestP1;
            sort[loc2] = bestP2;
            sort[loc3] = p3;

            first = sort[0];
            second = sort[1];
            third = sort[2];

            // todo: refine this part
            Rotation_cm.at<float>(0, 0) = first.at<float>(0, 0);
            Rotation_cm.at<float>(1, 0) = first.at<float>(1, 0);
            Rotation_cm.at<float>(2, 0) = first.at<float>(2, 0);
            Rotation_cm.at<float>(0, 1) = second.at<float>(0, 0);
            Rotation_cm.at<float>(1, 1) = second.at<float>(1, 0);
            Rotation_cm.at<float>(2, 1) = second.at<float>(2, 0);
            Rotation_cm.at<float>(0, 2) = third.at<float>(0, 0);
            Rotation_cm.at<float>(1, 2) = third.at<float>(1, 0);
            Rotation_cm.at<float>(2, 2) = third.at<float>(2, 0);

            cv::Mat U, W, VT;

            cv::SVD::compute(Rotation_cm, W, U, VT);

            Rotation_cm = U * VT;
        }

        return Rotation_cm;
    }

    void Map::FlagMatchedPlanePoints(Planar_SLAM::Frame &pF, const float &dTh) {

        unique_lock<mutex> lock(mMutexMap);
        int nMatches = 0;

        for (int i = 0; i < pF.mnPlaneNum; ++i) {

            cv::Mat pM = pF.ComputePlaneWorldCoeff(i);

            if (pF.mvpMapPlanes[i]) {
                for (auto mapPoint : mspMapPoints) {
                    cv::Mat pW = mapPoint->GetWorldPos();

                    double dis = abs(pM.at<float>(0, 0) * pW.at<float>(0, 0) +
                                     pM.at<float>(1, 0) * pW.at<float>(1, 0) +
                                     pM.at<float>(2, 0) * pW.at<float>(2, 0) +
                                     pM.at<float>(3, 0));

                    if (dis < 0.5) {
                        mapPoint->SetAssociatedWithPlaneFlag(true);
                        nMatches++;
                    }
                }
            }
        }
    }



    double Map::PointDistanceFromPlane(const cv::Mat &plane, PointCloud::Ptr boundry, bool out) {
        double res = 100;
        if (out)
            cout << " compute dis: " << endl;
        for (auto p : boundry->points) {
            double dis = abs(plane.at<float>(0, 0) * p.x +
                             plane.at<float>(1, 0) * p.y +
                             plane.at<float>(2, 0) * p.z +
                             plane.at<float>(3, 0));
            if (dis < res)
                res = dis;
        }
        if (out)
            cout << endl << "ave : " << res << endl;
        return res;
    }

} //namespace Planar_SLAM
