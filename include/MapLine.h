//
// Created by lan on 17-12-20.
//

#ifndef ORB_SLAM2_MAPLINE_H
#define ORB_SLAM2_MAPLINE_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/core/core.hpp>
#include <mutex>
#include <eigen3/Eigen/Core>
#include <map>

namespace Planar_SLAM
{
    class KeyFrame;
    class Map;
    class Frame;

    typedef Eigen::Matrix<double,6,1> Vector6d;
    class MapLine
    {
    public:
        MapLine(Vector6d &Pos, KeyFrame* pRefKF, Map* pMap);
        MapLine(Vector6d &Pos, Map* pMap, Frame* pFrame, const int &idxF);

        void SetWorldPos(const Vector6d &Pos);
        Vector6d GetWorldPos();

        Eigen::Vector3d GetNormal();
        KeyFrame* GetReferenceKeyFrame();

        map<KeyFrame*, size_t> GetObservations();
        int Observations();


        void AddObservation(KeyFrame* pKF, size_t idx);
        void EraseObservation(KeyFrame* pKF);

        int GetIndexInKeyFrame(KeyFrame* pKF);
        bool IsInKeyFrame(KeyFrame* pKF);

        void SetBadFlag();
        bool isBad();

        void Replace(MapLine* pML);
        MapLine* GetReplaced();

        void IncreaseVisible(int n=1);
        void IncreaseFound(int n=1);
        float GetFoundRatio();
        inline int GetFound(){
            return mnFound;
        }

        void ComputeDistinctiveDescriptors();

        cv::Mat GetDescriptor();

        void UpdateAverageDir();

        float GetMinDistanceInvariance();
        float GetMaxDistanceInvariance();
        int PredictScale(const float &currentDist, const float &logScaleFactor);

    public:
        long unsigned int mnId; //Global ID for MapLine
        static long unsigned int nNextId;
        const long int mnFirstKFid;
        const long int mnFirstFrame;
        int nObs;

        // Variables used by the tracking
        float mTrackProjX1;
        float mTrackProjY1;
        float mTrackProjX2;
        float mTrackProjY2;
        int mnTrackScaleLevel;
        float mTrackViewCos;
        bool mbTrackInView;

        long unsigned int mnTrackReferenceForFrame;

        long unsigned int mnLastFrameSeen;

        // Variables used by local mapping
        long unsigned int mnBALocalForKF;
        long unsigned int mnFuseCandidateForKF;

        // Variables used by loop closing
        long unsigned int mnLoopLineForKF;
        long unsigned int mnCorrectedByKF;
        long unsigned int mnCorrectedReference;
        cv::Mat mPosGBA;
        long unsigned int mnBAGlobalForKF;

        static std::mutex mGlobalMutex;

    public:
        Vector6d mWorldPos;
        Eigen::Vector3d mStart3D;
        Eigen::Vector3d mEnd3D;

        // KeyFrames observing the line and associated index in keyframe
        std::map<KeyFrame*, size_t> mObservations;

        Eigen::Vector3d mNormalVector;

        cv::Mat mLDescriptor;

        KeyFrame* mpRefKF;

        std::vector<cv::Mat> mvDesc_list;
        std::vector<Eigen::Vector3d> mvdir_list;

        //Tracking counters
        int mnVisible;
        int mnFound;

        // Bad flag , we don't currently erase MapPoint from memory
        bool mbBad;
        MapLine* mpReplaced;

        float mfMinDistance;
        float mfMaxDistance;

        Map* mpMap;

        mutex mMutexPos;
        mutex mMutexFeatures;
    };

}


#endif //ORB_SLAM2_MAPLINE_H
