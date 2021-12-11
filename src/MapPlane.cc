#include "MapPlane.h"
#include<mutex>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace Planar_SLAM{
    long unsigned int MapPlane::nNextId = 0;
    mutex MapPlane::mGlobalMutex;

    MapPlane::MapPlane(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
            mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), mpRefKF(pRefKF), mnVisible(1), mnFound(1),
            mnBALocalForKF(0), mnBAGlobalForKF(0), mvPlanePoints(new PointCloud()), mpMap(pMap), nObs(0),
            mbBad(false), mnFuseCandidateForKF(0), mpReplaced(static_cast<MapPlane*>(NULL)), mnLoopPlaneForKF(0),
            mnLoopVerticalPlaneForKF(0), mnLoopParallelPlaneForKF(0), mnCorrectedByKF(0),
            mnCorrectedReference(0) {
        mnId = nNextId++;

        Pos.copyTo(mWorldPos);

        mRed = rand() % 256;
        mBlue = rand() % 256;
        mGreen = rand() % 256;
    }

    void MapPlane::AddObservation(KeyFrame *pKF, int idx) {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
            return;
        mObservations[pKF] = idx;
        nObs++;
    }

    void MapPlane::AddVerObservation(KeyFrame *pKF, int idx) {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mVerObservations.count(pKF))
            return;
        mVerObservations[pKF] = idx;
    }

    void MapPlane::AddParObservation(KeyFrame *pKF, int idx) {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mParObservations.count(pKF))
            return;
        mParObservations[pKF] = idx;
    }

    void MapPlane::EraseObservation(KeyFrame *pKF) {
        bool bBad = false;
        {
            unique_lock<mutex> lock(mMutexFeatures);
            if (mObservations.count(pKF)) {
                mObservations.erase(pKF);
                nObs--;

                if (mpRefKF == pKF)
                    mpRefKF = mObservations.begin()->first;

                if (nObs <= 2)
                    bBad = true;
            }
        }

        if (bBad) {
            SetBadFlag();
        }
    }

    void MapPlane::EraseVerObservation(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mVerObservations.count(pKF)){
            mVerObservations.erase(pKF);
        }
    }

    void MapPlane::EraseParObservation(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mParObservations.count(pKF)){
            mParObservations.erase(pKF);
        }
    }

    map<KeyFrame*, size_t> MapPlane::GetObservations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mObservations;
    }

    map<KeyFrame*, size_t> MapPlane::GetVerObservations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mVerObservations;
    }

    map<KeyFrame*, size_t> MapPlane::GetParObservations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mParObservations;
    }

    int MapPlane::Observations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return nObs;
    }

    KeyFrame* MapPlane::GetReferenceKeyFrame()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mpRefKF;
    }

    int MapPlane::GetIndexInKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
            return mObservations[pKF];
        else
            return -1;
    }

    int MapPlane::GetIndexInVerticalKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mVerObservations.count(pKF))
            return mVerObservations[pKF];
        else
            return -1;
    }

    int MapPlane::GetIndexInParallelKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mParObservations.count(pKF))
            return mParObservations[pKF];
        else
            return -1;
    }

    bool MapPlane::IsInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return (mObservations.count(pKF));
    }

    void MapPlane::SetWorldPos(const cv::Mat &Pos)
    {
        unique_lock<mutex> lock2(mGlobalMutex);
        unique_lock<mutex> lock(mMutexPos);
        Pos.copyTo(mWorldPos);
    }

    cv::Mat MapPlane::GetWorldPos(){
        unique_lock<mutex> lock(mMutexPos);
        return mWorldPos.clone();
    }

    bool MapPlane::isBad()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mbBad;
    }

    void MapPlane::SetBadFlag()
    {
        map<KeyFrame*,size_t> obs, verObs, parObs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            mbBad=true;
            obs = mObservations;
            mObservations.clear();
            verObs = mVerObservations;
            mVerObservations.clear();
            parObs = mParObservations;
            mParObservations.clear();
        }
        for(auto & ob : obs)
        {
            KeyFrame* pKF = ob.first;
            pKF->EraseMapPlaneMatch(ob.second);
        }
        for(auto & verOb : verObs)
        {
            KeyFrame* pKF = verOb.first;
            pKF->EraseMapVerticalPlaneMatch(verOb.second);
        }
        for(auto & parOb : parObs)
        {
            KeyFrame* pKF = parOb.first;
            pKF->EraseMapParallelPlaneMatch(parOb.second);
        }

        mpMap->EraseMapPlane(this);
    }

    void MapPlane::Replace(MapPlane* pMP) {
        if(pMP->mnId==this->mnId)
            return;

        int nvisible, nfound;
        map<KeyFrame*,size_t> obs, verObs, parObs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            mbBad=true;
            obs = mObservations;
            mObservations.clear();
            verObs = mVerObservations;
            mVerObservations.clear();
            parObs = mParObservations;
            mParObservations.clear();
            nvisible = mnVisible;
            nfound = mnFound;
            mpReplaced = pMP;
        }

        for(auto & ob : obs)
        {
            // Replace measurement in keyframe
            KeyFrame* pKF = ob.first;

            *pMP->mvPlanePoints += pKF->mvPlanePoints[ob.second];

            if(!pMP->IsInKeyFrame(pKF))
            {
                pKF->ReplaceMapPlaneMatch(ob.second, pMP);
                pMP->AddObservation(pKF,ob.second);
            }
            else
            {
                pKF->EraseMapPlaneMatch(ob.second);
            }
        }
        for(auto & ob : verObs)
        {
            // Replace measurement in keyframe
            KeyFrame* pKF = ob.first;

            if(!pMP->IsInKeyFrame(pKF))
            {
                pKF->ReplaceMapVerticalPlaneMatch(ob.second, pMP);
                pMP->AddVerObservation(pKF,ob.second);
            }
            else
            {
                pKF->EraseMapVerticalPlaneMatch(ob.second);
            }
        }
        for(auto & ob : parObs)
        {
            // Replace measurement in keyframe
            KeyFrame* pKF = ob.first;

            if(!pMP->IsInKeyFrame(pKF))
            {
                pKF->ReplaceMapParallelPlaneMatch(ob.second, pMP);
                pMP->AddParObservation(pKF,ob.second);
            }
            else
            {
                pKF->EraseMapParallelPlaneMatch(ob.second);
            }
        }

        pMP->IncreaseFound(nfound);
        pMP->IncreaseVisible(nvisible);
        pMP->UpdateCoefficientsAndPoints();

        mpMap->EraseMapPlane(this);
    }

    MapPlane* MapPlane::GetReplaced()
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mpReplaced;
    }

    void MapPlane::IncreaseVisible(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnVisible+=n;
    }

    void MapPlane::IncreaseFound(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnFound+=n;
    }

    float MapPlane::GetFoundRatio()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return static_cast<float>(mnFound)/mnVisible;
    }

    void MapPlane::UpdateCoefficientsAndPoints() {
        PointCloud::Ptr combinedPoints (new PointCloud());
        map<KeyFrame*, size_t> observations = GetObservations();
        for(auto & observation : observations){
            KeyFrame* frame = observation.first;
            int id = observation.second;

            PointCloud::Ptr points (new PointCloud());
            pcl::transformPointCloud(frame->mvPlanePoints[id], *points, Converter::toMatrix4d(frame->GetPoseInverse()));

            *combinedPoints += *points;
        }

        pcl::VoxelGrid<PointT>  voxel;
        voxel.setLeafSize( 0.1, 0.1, 0.1);

        PointCloud::Ptr coarseCloud(new PointCloud());
        voxel.setInputCloud(combinedPoints);
        voxel.filter(*coarseCloud);

        mvPlanePoints = coarseCloud;

        if (mvPlanePoints->points.size() > 4) {
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::SACSegmentation<PointT> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(0.01);

            seg.setInputCloud(mvPlanePoints);
            seg.segment(*inliers, *coefficients);

        }
    }

    void MapPlane::UpdateCoefficientsAndPoints(const Planar_SLAM::Frame &pF, int id) {

        PointCloud::Ptr combinedPoints (new PointCloud());

        Eigen::Isometry3d T = Planar_SLAM::Converter::toSE3Quat(pF.mTcw );
        pcl::transformPointCloud(pF.mvPlanePoints[id], *combinedPoints, T.inverse().matrix());


        *combinedPoints += *mvPlanePoints;

        pcl::VoxelGrid<PointT>  voxel;
        voxel.setLeafSize( 0.1, 0.1, 0.1);

        PointCloud::Ptr coarseCloud(new PointCloud());
        voxel.setInputCloud(combinedPoints);
        voxel.filter(*coarseCloud);

        mvPlanePoints = coarseCloud;

        if (mvPlanePoints->points.size() > 4) {
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::SACSegmentation<PointT> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(0.01);

            seg.setInputCloud(mvPlanePoints);
            seg.segment(*inliers, *coefficients);
        }
    }
}
