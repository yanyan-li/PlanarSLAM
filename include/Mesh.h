//
// Created by raza on 17.02.20.
//

#ifndef ORB_SLAM2_MESH_H
#define ORB_SLAM2_MESH_H

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/file_io.h>
#include <pcl/io/ply/ply_parser.h>
#include <pcl/io/ply/ply.h>

#include <pcl/point_types.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>

#include <pcl/range_image/range_image.h>

#include <pcl/common/transforms.h>
#include <pcl/common/geometry.h>
#include <pcl/common/common.h>
#include <pcl/common/common_headers.h>

#include <pcl/ModelCoefficients.h>

#include <pcl/features/normal_3d.h>
//#include <pcl/features/gasd.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/filters/crop_box.h>
#include <pcl/filters/crop_hull.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/surface/poisson.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/simplification_remove_unused_vertices.h>
#include <pcl/surface/vtk_smoothing/vtk_utils.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/convex_hull.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>

#include <boost/filesystem.hpp>
#include <boost/algorithm/algorithm.hpp>
#include <boost/thread/thread.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include "Thirdparty/triangle/triangle.h"

class Mesh {

public:
    float static distance(cv::Point3f &P1, cv::Point3f &P2, cv::Point3f &P3, float threshold)
    {
       float d1 =  sqrt((P1.x-P2.x)*(P1.x-P2.x)+ (P1.y-P2.y)*(P1.y-P2.y)+ (P1.z-P2.z)*(P1.z-P2.z));
       float d2 =  sqrt((P1.x-P3.x)*(P1.x-P3.x)+ (P1.y-P3.y)*(P1.y-P3.y)+ (P1.z-P3.z)*(P1.z-P3.z));
       float d3 =  sqrt((P3.x-P2.x)*(P3.x-P2.x)+ (P3.y-P2.y)*(P3.y-P2.y)+ (P3.z-P2.z)*(P3.z-P2.z));

       if(d1>threshold||d2>threshold||d3>threshold)
       {
           return -1;
       }

        return (d1+d2+d3)/3;
    }

    void static visualizeMesh(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,pcl::PolygonMesh &mesh){

        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("MAP3D MESH"));

        int PORT1 = 0;
        viewer->createViewPort(0.0, 0.0, 0.5, 1.0, PORT1);
        viewer->setBackgroundColor (0, 0, 0, PORT1);
        viewer->addText("ORIGINAL", 10, 10, "PORT1", PORT1);

        int PORT2 = 0;
        viewer->createViewPort(0.5, 0.0, 1.0, 1.0, PORT2);
        viewer->setBackgroundColor (0, 0, 0, PORT2);
        viewer->addText("MESH", 10, 10, "PORT2", PORT2);
        viewer->addPolygonMesh(mesh,"mesh",PORT2);

        viewer->addCoordinateSystem();
        pcl::PointXYZ p1, p2, p3;

        p1.getArray3fMap() << 1, 0, 0;
        p2.getArray3fMap() << 0, 1, 0;
        p3.getArray3fMap() << 0,0.1,1;

        viewer->addText3D("x", p1, 0.2, 1, 0, 0, "x_");
        viewer->addText3D("y", p2, 0.2, 0, 1, 0, "y_");
        viewer->addText3D ("z", p3, 0.2, 0, 0, 1, "z_");

        if(cloud->points[0].r <= 0 and cloud->points[0].g <= 0 and cloud->points[0].b<= 0 ){
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> color_handler(cloud,255,255,0);
            viewer->removeAllPointClouds(0);
            viewer->addPointCloud(cloud,color_handler,"original_cloud",PORT1);
        }else{
            viewer->addPointCloud(cloud,"original_cloud",PORT1);
        }

        viewer->initCameraParameters ();
        viewer->resetCamera();

        std::cout << "Press [q] to exit!" << std::endl;
        while (!viewer->wasStopped ()){
            viewer->spin();
        }
    }

    void static create_mesh_tri(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PolygonMesh& triangles)
    {
        triangulateio in, out;
        // inputs
        in.numberofpoints = cloud->points.size();
        in.pointlist = (float *)malloc(in.numberofpoints * 2 * sizeof(float));
        int32_t k = 0;
        for (auto &p : cloud->points)
        {
            in.pointlist[k++] = p.x;
            in.pointlist[k++] = p.y;
        }
        in.numberofpointattributes = 0;
        in.pointattributelist = nullptr;
        in.pointmarkerlist = nullptr;
        in.numberofholes = 0;
        in.holelist = nullptr;
        in.numberofregions = 0;
        in.regionlist = nullptr;
        // outputs
        out.pointlist = nullptr;
        out.pointattributelist = nullptr;
        out.pointmarkerlist = nullptr;
        out.trianglelist = nullptr;
        out.triangleattributelist = nullptr;
        out.neighborlist = nullptr;
        out.segmentlist = nullptr;
        out.segmentmarkerlist = nullptr;
        out.edgelist = nullptr;
        out.edgemarkerlist = nullptr;

        // char parameters[] = "pnezcQ";
        char parameters[] = "zVQ";
        triangulate(parameters, &in, &out, nullptr);

        pcl::toPCLPointCloud2(*cloud, triangles.cloud);
        k = 0;

        float averageDistance = 0;
        int i =0;

//        //double
//        for (int index = 0; index < out.numberoftriangles; ++index)
//        {
//            cv::Point3f P1 = cv::Point3f(cloud->points[ out.trianglelist[k]].x, cloud->points[ out.trianglelist[k]].y, cloud->points[ out.trianglelist[k]].y );
//            cv::Point3f P2 = cv::Point3f(cloud->points[ out.trianglelist[k+1]].x, cloud->points[ out.trianglelist[k+1]].y, cloud->points[ out.trianglelist[k+1]].y );
//            cv::Point3f P3 = cv::Point3f(cloud->points[ out.trianglelist[k+2]].x, cloud->points[ out.trianglelist[k+2]].y, cloud->points[ out.trianglelist[k+2]].y );
//            float goodTriangl = distance(P1, P2,P3,0.25);
//
//            cout<< goodTriangl<<endl;
//            if(goodTriangl>0)
//            {
//                averageDistance += goodTriangl;
//                i++;
//
//            }
//        }
//
//        if(i>0)
//        {
//            averageDistance /= i;
//            cout<< "here "<< averageDistance<<endl;
//        }

        for (int index = 0; index < out.numberoftriangles; ++index)
        {
            pcl::Vertices vtx;

            cv::Point3f P1 = cv::Point3f(cloud->points[ out.trianglelist[k]].x, cloud->points[ out.trianglelist[k]].y, cloud->points[ out.trianglelist[k]].y );
            cv::Point3f P2 = cv::Point3f(cloud->points[ out.trianglelist[k+1]].x, cloud->points[ out.trianglelist[k+1]].y, cloud->points[ out.trianglelist[k+1]].y );
            cv::Point3f P3 = cv::Point3f(cloud->points[ out.trianglelist[k+2]].x, cloud->points[ out.trianglelist[k+2]].y, cloud->points[ out.trianglelist[k+2]].y );
            float goodTriangl = distance(P1, P2,P3, 0.25);
            if(goodTriangl>0)
                vtx.vertices.push_back(out.trianglelist[k]); // trianglelist为vertex的索引，一次索引3个就是组成一个三角面片
            k++;
            if(goodTriangl>0)
                vtx.vertices.push_back(out.trianglelist[k]);
            k++;
            if(goodTriangl>0)
                vtx.vertices.push_back(out.trianglelist[k]);
            k++;

            //<<endl<<out.edgelist[k]<<endl;
            if(goodTriangl>0)
                triangles.polygons.push_back(vtx);
        }
    }

    void static create_mesh(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int surface_mode, int normal_method, pcl::PolygonMesh& triangles){
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        normalEstimation.setInputCloud(cloud);
        normalEstimation.setSearchMethod(tree);

        normalEstimation.setViewPoint(0,0,0);
        normalEstimation.setKSearch(8);
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        //normalEstimation.setRadiusSearch(0.3);
        normalEstimation.compute(*normals);

        std::cout << "begin  mesh..." << std::endl;

        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
        pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
        pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
        tree2->setInputCloud(cloud_with_normals);
        pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;   //

        float search_radius = 20;
        float setMU = 5;
        int maxiNearestNeighbors = 100;
        bool normalConsistency = false;
        gp3.setSearchRadius(search_radius);//It was 0.025
        gp3.setMu(setMU); //It was 2.5
        gp3.setMaximumNearestNeighbors(maxiNearestNeighbors);    //It was 100
        gp3.setNormalConsistency(normalConsistency); //It was false
        gp3.setInputCloud(cloud_with_normals);
        gp3.setSearchMethod(tree2);
        gp3.reconstruct(triangles);
    }





    void static SaveMeshModel(pcl::PolygonMesh& triangles, const string&filename ){
        vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
        pcl::PolygonMesh mesh_pcl;
        pcl::VTKUtils::convertToVTK(triangles,polydata);
        pcl::VTKUtils::convertToPCL(polydata,mesh_pcl);
        pcl::io::savePolygonFilePLY(filename, mesh_pcl);
    }
};

#endif //Planar_SLAM
