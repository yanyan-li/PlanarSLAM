#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"
#include <set>
#include <unordered_map>
#include <tuple>

#include <mutex>


#include "MapLine.h"

#include "MapPlane.h"
#include <eigen3/Eigen/Core>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/ModelCoefficients.h>

namespace Planar_SLAM
{

    class MapPoint;
    class KeyFrame;
    class MapLine;
    class MapPlane;
    class Frame;

    class Map
    {
    public:
        typedef pcl::PointXYZRGB PointT;
        typedef pcl::PointCloud <PointT> PointCloud;

        Map();

        void AddKeyFrame(KeyFrame* pKF);
        void AddMapPoint(MapPoint* pMP);
        void EraseMapPoint(MapPoint* pMP);
        void EraseKeyFrame(KeyFrame* pKF);
        void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);

        void InformNewBigChange();
        int GetLastBigChangeIdx();
        void AddMapLine(MapLine* pML);
        void EraseMapLine(MapLine* pML);
        void SetReferenceMapLines(const std::vector<MapLine*> &vpMLs);

        std::vector<KeyFrame*> GetAllKeyFrames();
        std::vector<MapPoint*> GetAllMapPoints();
        std::vector<MapPoint*> GetReferenceMapPoints();

        std::vector<MapLine*> GetAllMapLines();
        std::vector<MapLine*> GetReferenceMapLines();
        long unsigned int MapLinesInMap();

        long unsigned int MapPointsInMap();
        long unsigned  KeyFramesInMap();

        long unsigned int GetMaxKFid();

        void clear();

        vector<KeyFrame*> mvpKeyFrameOrigins;

        std::mutex mMutexMapUpdate;

        // This avoid that two points are created simultaneously in separate threads (id conflict)
        std::mutex mMutexPointCreation;
        std::mutex mMutexLineCreation;

        void AddMapPlane(MapPlane* pMP);
        void EraseMapPlane(MapPlane *pMP);
        std::vector<MapPlane*> GetAllMapPlanes();
        long unsigned int MapPlanesInMap();

        cv::Mat FindManhattan(Frame &pF, const float &verTh, bool out = false);

        void FlagMatchedPlanePoints(Planar_SLAM::Frame &pF, const float &dTh);

        double PointDistanceFromPlane(const cv::Mat& plane, PointCloud::Ptr boundry, bool out = false);

    protected:
        std::set<MapPoint*> mspMapPoints;

        std::set<MapLine*> mspMapLines;

        std::set<MapPlane*> mspMapPlanes;

        std::set<KeyFrame*> mspKeyFrames;

        std::vector<MapPoint*> mvpReferenceMapPoints;
        std::vector<MapLine*> mvpReferenceMapLines;
        long unsigned int mnMaxKFid;

        // Index related to a big change in the map (loop closure, global BA)
        int mnBigChangeIdx;
        std::mutex mMutexMap;
    };

} //namespace Planar_SLAM

#endif // MAP_H
