#ifndef MAPPLANE_H
#define MAPPLANE_H

#include"KeyFrame.h"
#include"Frame.h"
#include"Map.h"
#include "Converter.h"

#include <opencv2/core/core.hpp>
#include <mutex>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/exceptions.h>


namespace Planar_SLAM {
    class KeyFrame;
    class Frame;
    class Map;
    class MapPlane {
        typedef pcl::PointXYZRGB PointT;
        typedef pcl::PointCloud <PointT> PointCloud;
    public:
        MapPlane(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap);

        void SetWorldPos(const cv::Mat &Pos);
        cv::Mat GetWorldPos();

        void Replace(MapPlane* pMP);
        void ReplaceVerticalObservations(MapPlane* pMP);
        void ReplaceParallelObservations(MapPlane* pMP);
        MapPlane* GetReplaced();

        void IncreaseVisible(int n=1);
        void IncreaseFound(int n=1);
        float GetFoundRatio();
        inline int GetFound(){
            return mnFound;
        }

        void SetBadFlag();
        bool isBad();

        KeyFrame* GetReferenceKeyFrame();

        void AddObservation(KeyFrame* pKF, int idx);
        void AddParObservation(KeyFrame* pKF, int idx);
        void AddVerObservation(KeyFrame* pKF, int idx);

        void EraseObservation(KeyFrame* pKF);
        void EraseVerObservation(KeyFrame* pKF);
        void EraseParObservation(KeyFrame* pKF);

        std::map<KeyFrame*, size_t> GetObservations();
        std::map<KeyFrame*, size_t> GetParObservations();
        std::map<KeyFrame*, size_t> GetVerObservations();
        int Observations();
        int GetIndexInKeyFrame(KeyFrame *pKF);
        int GetIndexInVerticalKeyFrame(KeyFrame *pKF);
        int GetIndexInParallelKeyFrame(KeyFrame *pKF);
        bool IsInKeyFrame(KeyFrame *pKF);
        void UpdateCoefficientsAndPoints();
        void UpdateCoefficientsAndPoints(const Frame& pF, int id);

    public:
        long unsigned int mnId; ///< Global ID for MapPlane;
        static long unsigned int nNextId;
        long int mnFirstKFid;
        long int mnFirstFrame;
        int nObs;

        static std::mutex mGlobalMutex;

        long unsigned int mnBALocalForKF; //used in local BA

        long unsigned int mnFuseCandidateForKF;

        // Variables used by loop closing
        long unsigned int mnLoopPlaneForKF;
        long unsigned int mnLoopVerticalPlaneForKF;
        long unsigned int mnLoopParallelPlaneForKF;
        long unsigned int mnCorrectedByKF;
        long unsigned int mnCorrectedReference;
        cv::Mat mPosGBA;
        long unsigned int mnBAGlobalForKF;

        //used for visualization
        int mRed;
        int mGreen;
        int mBlue;

        PointCloud::Ptr mvPlanePoints;

        //Tracking counters
        int mnVisible;
        int mnFound;

    protected:
        cv::Mat mWorldPos; ///< Position in absolute coordinates

        std::map<KeyFrame*, size_t> mObservations;
        std::map<KeyFrame*, size_t> mParObservations;
        std::map<KeyFrame*, size_t> mVerObservations;

        std::mutex mMutexPos;
        std::mutex mMutexFeatures;

        KeyFrame* mpRefKF;

        bool mbBad;
        MapPlane* mpReplaced;

        Map* mpMap;
    };
} //namespace Planar_SLAM
#endif
