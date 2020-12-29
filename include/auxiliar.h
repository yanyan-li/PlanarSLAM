//
// Created by lan on 17-12-18.
//

#pragma once

#include <iostream>

#include <cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

#include <vector>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>

typedef Eigen::Matrix<double,6,1> Vector6d;
typedef Eigen::Matrix<double,6,6> Matrix6d;

struct compare_descriptor_by_NN_dist
{
    inline bool operator()(const std::vector<cv::DMatch>& a, const std::vector<cv::DMatch>& b){
        return ( a[0].distance < b[0].distance);
    }
};

struct conpare_descriptor_by_NN12_dist
{
    inline bool operator()(const std::vector<cv::DMatch>& a, const std::vector<cv::DMatch>& b){
        return ((a[1].distance - a[0].distance) > (b[1].distance - b[0].distance));
    }
};

struct sort_descriptor_by_queryIdx
{
    inline bool operator()(const std::vector<cv::DMatch>& a, const std::vector<cv::DMatch>& b){
        return ( a[0].queryIdx < b[0].queryIdx );
    }
};

struct sort_lines_by_response
{
    inline bool operator()(const cv::line_descriptor::KeyLine& a, const cv::line_descriptor::KeyLine& b){
        return ( a.response > b.response );
    }
};

inline cv::Mat SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<  0, -v.at<float>(2), v.at<float>(1),
                        v.at<float>(2),               0,-v.at<float>(0),
                       -v.at<float>(1),  v.at<float>(0),             0);
}

inline double vector_mad(std::vector<double> residues)
{
    if(residues.size()!=0)
    {
        // Return the standard deviation of vector with MAD estimation
        int n_samples = residues.size();
        sort(residues.begin(), residues.end());
        double median = residues[n_samples/2];
        for(int i=0; i<n_samples; i++)
            residues[i] = fabs(residues[i]-median);
        std::sort(residues.begin(), residues.end());
        double MAD = residues[n_samples/2];
        return 1.4826*MAD;
    } else
        return 0.0;
}
