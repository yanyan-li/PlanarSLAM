#include "PlaneExtractor.h"
using namespace std;
using namespace cv;
using namespace Eigen;

PlaneDetection::PlaneDetection()
{
	cloud.vertices.resize(kDepthHeight * kDepthWidth);
	cloud.w = kDepthWidth;
	cloud.h = kDepthHeight;
}

PlaneDetection::~PlaneDetection()
{
	cloud.vertices.clear();
	seg_img_.release();
	color_img_.release();
}

bool PlaneDetection::readColorImage(cv::Mat RGBImg)
{
	color_img_ = RGBImg;
	if (color_img_.empty() || color_img_.depth() != CV_8U)
	{
		cout << "ERROR: cannot read color image. No such a file, or the image format is not 8UC3" << endl;
		return false;
	}
	return true;
}

bool PlaneDetection::readDepthImage(cv::Mat depthImg, cv::Mat &K)
{
	cv::Mat depth_img = depthImg;
	if (depth_img.empty() || depth_img.depth() != CV_16U)
	{
		cout << "WARNING: cannot read depth image. No such a file, or the image format is not 16UC1" << endl;
		return false;
	}
	int rows = depth_img.rows, cols = depth_img.cols;
	int vertex_idx = 0;
	for (int i = 0; i < rows; i+=1)
	{
		for (int j = 0; j < cols; j+=1)
		{
			double z = (double)(depth_img.at<unsigned short>(i, j)) / kScaleFactor;
			if (_isnan(z))
			{
				cloud.vertices[vertex_idx++] = VertexType(0, 0, z);
				continue;
			}
			double x = ((double)j - K.at<float>(0, 2)) * z / K.at<float>(0, 0);
			double y = ((double)i - K.at<float>(1, 2)) * z / K.at<float>(1, 1);
			cloud.vertices[vertex_idx++] = VertexType(x, y, z);
		}
	}
	return true;
}

void PlaneDetection::runPlaneDetection()
{
	seg_img_ = cv::Mat(kDepthHeight, kDepthWidth, CV_8UC3);
	plane_filter.run(&cloud, &plane_vertices_, &seg_img_);

	plane_num_ = (int)plane_vertices_.size();
}
