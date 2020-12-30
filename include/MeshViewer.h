#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include "System.h"
#include "KeyFrame.h"
#include "Map.h"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <condition_variable>
#include <pcl/ModelCoefficients.h>
#include <pcl/visualization/cloud_viewer.h>


namespace Planar_SLAM {
class KeyFrame;
class Map;
    class MeshViewer {
    public:
        typedef pcl::PointXYZRGBA PointT;
        typedef pcl::PointCloud<PointT> PointCloud;

        MeshViewer(Map* map);

        // 插入一个keyframe，会更新一次地图
        void insertKeyFrame(KeyFrame *kf, cv::Mat &color, cv::Mat &depth);

        void shutdown();

        void viewer();

        void print();
        void SaveMeshModel(const string &filename);

    protected:
        Map* mMap;

        void AddKFPointCloud(KeyFrame *pKF);

        PointCloud::Ptr mAllCloudPoints;

        shared_ptr<thread> viewerThread;

        bool shutDownFlag = false;
        mutex shutDownMutex;

        bool printFlag = false;
        mutex printMutex;

        condition_variable keyFrameUpdated;
        mutex keyFrameUpdateMutex;

        // data to generate point clouds
        vector<KeyFrame *> mvKeyframes;

        mutex keyframeMutex;
        uint16_t lastKeyframeSize = 0;

    };
}
#endif // POINTCLOUDMAPPING_H
