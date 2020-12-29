#ifndef PLANEDETECTION_H
#define PLANEDETECTION_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include <string>
#include <fstream>
#include <Eigen/Eigen>
#include "include/peac/AHCPlaneFitter.hpp"
#include <unordered_map>

typedef Eigen::Vector3d VertexType;

const int kScaleFactor = 5000;

const int kDepthWidth = 640;
const int kDepthHeight = 480;

#ifdef __linux__
#define _isnan(x) isnan(x)
#endif

struct ImagePointCloud
{
    std::vector<VertexType> vertices; // 3D vertices
	int w, h;

	inline int width() const { return w; }
	inline int height() const { return h; }
	inline bool get(const int row, const int col, double &x, double &y, double &z) const {
		const int pixIdx = row * w + col;
		z = vertices[pixIdx][2];
		// Remove points with 0 or invalid depth in case they are detected as a plane
		if (z == 0 || std::_isnan(z)) return false;
		x = vertices[pixIdx][0];
		y = vertices[pixIdx][1];
		return true;
	}
};

class PlaneDetection
{
public:
	ImagePointCloud cloud;
	ahc::PlaneFitter< ImagePointCloud > plane_filter;
    std::vector<std::vector<int>> plane_vertices_; // vertex indices each plane contains
	cv::Mat seg_img_; // segmentation image
	cv::Mat color_img_; // input color image
	int plane_num_;

public:
	PlaneDetection();
	~PlaneDetection();

	bool readColorImage(cv::Mat RGBImg);

	bool readDepthImage(cv::Mat depthImg, cv::Mat &K);

	void runPlaneDetection();

};


#endif