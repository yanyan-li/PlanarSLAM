#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"
#include <opencv2/opencv.hpp>
//#include "LSDextractor"
#include "MapLine.h"
#include "LSDextractor.h"
#include "auxiliar.h"
#include <fstream>
#include "Config.h"
#include "MapPlane.h"
#include "PlaneExtractor.h"

#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/features/integral_image_normal.h>

namespace Planar_SLAM
{
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

    class MapPoint;
    class KeyFrame;
    class MapLine;
    class MapPlane;
    class Frame
    {

    public:
        typedef pcl::PointXYZRGB PointT;
        typedef pcl::PointCloud <PointT> PointCloud;

        Frame();

        // Copy constructor.
        Frame(const Frame &frame);

        // Constructor for RGB-D cameras.
        Frame(const cv::Mat &imRGB, const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, const float &depthMapFactor);

        // Extract ORB on the image. 0 for left image and 1 for right image.
        void ExtractORB(int flag, const cv::Mat &im);

        // extract line feature
        void ExtractLSD(const cv::Mat &im, const cv::Mat &depth,cv::Mat K);

        void isLineGood(const cv::Mat &imGray,  const cv::Mat &imDepth,cv::Mat K);

        void lineDescriptorMAD( std::vector<std::vector<cv::DMatch>> matches, double &nn_mad, double &nn12_mad) const;

        // Compute Bag of Words representation.
        void ComputeBoW();

        // Set the camera pose.
        void SetPose(cv::Mat Tcw);

        // Computes rotation, translation and camera center matrices from the camera pose.
        void UpdatePoseMatrices();

        // Returns the camera center.
        inline cv::Mat GetCameraCenter(){
            return mOw.clone();
        }

        // Returns inverse of rotation
        inline cv::Mat GetRotationInverse(){
            return mRwc.clone();
        }

        // Check if a MapPoint is in the frustum of the camera
        // and fill variables of the MapPoint to be used by the tracking
        bool isInFrustum(MapPoint* pMP, float viewingCosLimit);
        bool isInFrustum(MapLine* pML, float viewingCosLimit);
        // Compute the cell of a keypoint (return false if outside the grid)
        bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

        vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1) const;
        vector<size_t> GetLinesInArea(const float &x1, const float &y1, const float &x2, const float &y2,
                                      const float &r, const int minLevel=-1, const int maxLevel=-1) const;

        // Search a match for each keypoint in the left image to a keypoint in the right image.
        // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
        void ComputeStereoMatches();

        // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
        void ComputeStereoFromRGBD(const cv::Mat &imDepth);

        // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
        cv::Mat UnprojectStereo(const int &i);
        Vector6d obtain3DLine(const int &i);
        void EstimateSurfaceNormalGradient( const cv::Mat &imDepth, const cv::Mat &imGraphMask,const double &timeStamp,cv::Mat &K);
        void EstimateVanihsingDirection();

        cv::Mat ComputePlaneWorldCoeff(const int &idx);

        // remove useless planes
        bool MaxPointDistanceFromPlane(cv::Mat &plane, PointCloud::Ptr pointCloud);
    public:
        // Vocabulary used for relocalization.
        ORBVocabulary* mpORBvocabulary;

        // Feature extractor. The right is used only in the stereo case.
        ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
        // line feature extractor, 自己添加的
        LineSegment* mpLineSegment;
        // Frame timestamp.
        double mTimeStamp;

        // Calibration matrix and OpenCV distortion parameters.
        cv::Mat mK;
        static float fx;
        static float fy;
        static float cx;
        static float cy;
        static float invfx;
        static float invfy;
        cv::Mat mDistCoef;

        // Stereo baseline multiplied by fx.
        float mbf;

        // Stereo baseline in meters.
        float mb;

        // Threshold close/far points. Close points are inserted from 1 view.
        // Far points are inserted as in the monocular case from 2 views.
        float mThDepth;

        // Number of KeyPoints.
        int N;
        int NL;
        bool dealWithLine;
        float blurNumber;
        // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
        // In the stereo case, mvKeysUn is redundant as images must be rectified.
        // In the RGB-D case, RGB images can be distorted.
        std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
        std::vector<cv::KeyPoint> mvKeysUn;

        // Corresponding stereo coordinate and depth for each keypoint.
        // "Monocular" keypoints have a negative value.
        std::vector<float> mvuRight;
        std::vector<float> mvDepth;
        std::vector<float>  mvDepthLine;
        // Bag of Words Vector structures.
        DBoW2::BowVector mBowVec;
        DBoW2::FeatureVector mFeatVec;

        // ORB descriptor, each row associated to a keypoint.
        cv::Mat mDescriptors, mDescriptorsRight;

        // MapPoints associated to keypoints, NULL pointer if no association.
        std::vector<MapPoint*> mvpMapPoints;

        // Flag to identify outlier associations.
        std::vector<bool> mvbOutlier;

        SurfaceNormal surfacenomal;
        std::vector<SurfaceNormal> vSurfaceNormal;
        std::vector<VanishingDirection> vVanishingDirection;

        std::vector<cv::Point2i> vSurfaceNormalx;std::vector<cv::Point2i> vSurfaceNormaly;std::vector<cv::Point2i> vSurfaceNormalz;

        std::vector<cv::Point3f> vSurfacePointx;std::vector<cv::Point3f> vSurfacePointy;std::vector<cv::Point3f> vSurfacePointz;

        std::vector<vector<cv::Point2d>> vVanishingLinex;std::vector<vector<cv::Point2d>> vVanishingLiney;std::vector<vector<cv::Point2d>> vVanishingLinez;
        //2D endpoints, 3D LinesPC
        std::vector<RandomPoint3d> vVaishingLinePCx;std::vector<RandomPoint3d> vVaishingLinePCy;std::vector<RandomPoint3d> vVaishingLinePCz;

        cv::Mat mLdesc;
        std::vector<cv::line_descriptor::KeyLine> mvKeylinesUn;
        std::vector<Eigen::Vector3d> mvKeyLineFunctions;
        std::vector<bool> mvbLineOutlier;
        std::vector<MapLine*> mvpMapLines;

        vector<cv::Point3d> mv3DLineforMap;
        vector<Vector6d > mvLines3D;
        vector<FrameLine> mVF3DLines;

        // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
        static float mfGridElementWidthInv;
        static float mfGridElementHeightInv;
        std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

        // Camera pose.
        cv::Mat mTcw;
        cv::Mat mTwc;

        // Rotation, translation and camera center
        cv::Mat mRcw;
        cv::Mat mtcw;
        cv::Mat mRwc;
        cv::Mat mOw; //==mtwc
        
        // Current and Next Frame id.
        static long unsigned int nNextId;
        long unsigned int mnId;

        // Reference Keyframe.
        KeyFrame* mpReferenceKF;

        // Scale pyramid info.
        int mnScaleLevels;
        float mfScaleFactor;
        float mfLogScaleFactor;
        std::vector<float> mvScaleFactors;
        std::vector<float> mvInvScaleFactors;
        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSigma2;

        // Undistorted Image Bounds (computed once).
        static float mnMinX;
        static float mnMaxX;
        static float mnMinY;
        static float mnMaxY;

        static bool mbInitialComputations;

        std::vector<PointCloud> mvPlanePoints;
        std::vector<cv::Mat> mvPlaneCoefficients;
        std::vector<MapPlane*> mvpMapPlanes;
        std::vector<MapPlane*> mvpParallelPlanes;
        std::vector<MapPlane*> mvpVerticalPlanes;
        std::vector<std::vector<MapPoint*>> mvPlanePointMatches;
        std::vector<std::vector<MapLine*>> mvPlaneLineMatches;
        // Flag to identify outlier planes new planes.
        std::vector<bool> mvbPlaneOutlier;
        std::vector<bool> mvbParPlaneOutlier;
        std::vector<bool> mvbVerPlaneOutlier;
        int mnPlaneNum;
        bool mbNewPlane; // used to determine a keyframe

        PlaneDetection planeDetector;


    private:

        // Undistort keypoints given OpenCV distortion parameters.
        // Only for the RGB-D case. Stereo must be already rectified!
        // (called in the constructor).
        void UndistortKeyPoints();

        // Computes image bounds for the undistorted image (called in the constructor).
        void ComputeImageBounds(const cv::Mat &imLeft);

        // Assign keypoints to the grid for speed up feature matching (called in the constructor).
        void AssignFeaturesToGrid();

        void ComputePlanes(const cv::Mat &imDepth, const cv::Mat &depth, const cv::Mat &imGrey, cv::Mat K, float depthMapFactor);
    };

}// namespace Planar_SLAM

#endif // FRAME_H