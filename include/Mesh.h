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

class Mesh {

public:
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

    void static create_mesh(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int surface_mode, int normal_method, pcl::PolygonMesh& triangles){

         /* ****Translated point cloud to origin**** */
        // 2,1


        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;                   
        normalEstimation.setInputCloud(cloud);                                    
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);          
        normalEstimation.setSearchMethod(tree);
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);     

        normalEstimation.setKSearch(10);                    
        //normalEstimation.setRadiusSearch(0.3);            
        normalEstimation.compute(*normals);




        pcl::StopWatch time;

        std::cout << "begin  mesh..." << std::endl;

        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
        pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

        
        pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
        tree2->setInputCloud(cloud_with_normals);

        // 
        pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;   // 

        gp3.setSearchRadius(0.1);  
        gp3.setMu(2.5);  
        gp3.setMaximumNearestNeighbors(100);    

        gp3.setNormalConsistency(false);  

        gp3.setInputCloud(cloud_with_normals);     
        gp3.setSearchMethod(tree2);   //
        gp3.reconstruct(triangles);  //
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
