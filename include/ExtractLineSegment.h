//
// Created by lan on 17-12-13.
//

#ifndef ORB_SLAM2_LINEFEATURE_H
#define ORB_SLAM2_LINEFEATURE_H

#include <iostream>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <eigen3/Eigen/Core>
#include "auxiliar.h"

namespace Planar_SLAM
{
    class LineSegment
    {
    public:
        LineSegment();

        ~LineSegment()= default;

        void ExtractLineSegment(const cv::Mat &img, std::vector<cv::line_descriptor::KeyLine> &keylines, cv::Mat &ldesc, std::vector<Eigen::Vector3d> &keylineFunctions, float scale = 1.2, int numOctaves = 1);
    };
}


#endif //ORB_SLAM2_LINEFEATURE_H
