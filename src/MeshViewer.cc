/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "MeshViewer.h"
#include <KeyFrame.h>
#include <opencv2/highgui/highgui.hpp>
#include "Converter.h"
#include "Mesh.h"
namespace Planar_SLAM {
    MeshViewer::MeshViewer(Map *map): mMap(map), printFlag(true) {

        mAllCloudPoints = boost::make_shared<PointCloud>();

        viewerThread = make_shared<thread>(bind(&MeshViewer::viewer, this));
    }

    void MeshViewer::shutdown() {
        {
            unique_lock<mutex> lck(shutDownMutex);
            shutDownFlag = true;
            keyFrameUpdated.notify_one();
        }
        viewerThread->join();
    }

    void MeshViewer::insertKeyFrame(KeyFrame *kf, cv::Mat &color, cv::Mat &depth) {
        cout << "receive a keyframe, id = " << kf->mnId << endl;
        unique_lock<mutex> lck(keyframeMutex);
        mvKeyframes.push_back(kf);
        keyFrameUpdated.notify_one();
    }

    void MeshViewer::print() {
        unique_lock<mutex> lck_print(printMutex);
        printFlag = true;
    }

    void MeshViewer::viewer() {
        boost::shared_ptr<pcl::visualization::PCLVisualizer> meshViewer (new pcl::visualization::PCLVisualizer ("Planar Mesh"));
        meshViewer->setBackgroundColor (255,255,255);
        int i = 0;
        int cnt = 0;
        char buffer[256];
        while(1)
        {
            {
                unique_lock<mutex> lck_shutdown(shutDownMutex);
                if (shutDownFlag) {
                    break;
                }

            }

            {
                unique_lock<mutex> lck_print(printMutex);
                if (!printFlag) {
                    continue;
                }
            }

            for (int j = 0; j < i; j++) {
                meshViewer->removePolygonMesh(to_string(j));
            }


            std::vector<Planar_SLAM::MapPlane*> PlaneMap = mMap->GetAllMapPlanes();
            i = 0;
            for(auto pMP : PlaneMap){
                int ir = pMP->mRed;
                int ig = pMP->mGreen;
                int ib = pMP->mBlue;

                pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud(new pcl::PointCloud<pcl::PointXYZ>());
                for(auto& planePoint : pMP->mvPlanePoints.get()->points){
                    int i = 0;
                    pcl::PointXYZ p;
                    p.x = planePoint.x;
                    p.y = planePoint.y;
                    p.z = planePoint.z;
                    meshCloud->points.push_back(p);
                }


                if (meshCloud->points.size() > 0) {
                    pcl::PolygonMesh cloud_mesh;
                    string id = std::to_string(i++);
                    Mesh::create_mesh(meshCloud, 2, 1, cloud_mesh);
                    meshViewer->addPolygonMesh(cloud_mesh, id);
                    meshViewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,(float)ir/256, (float)ig/256,(float)ib/256,id);

                }
            }

            meshViewer->spinOnce();

            {
                unique_lock<mutex> lck_print(printMutex);
                printFlag = false;
            }

            sprintf(buffer, "mesh/%06d.png", cnt++);
            meshViewer->saveScreenshot(buffer);

        }
    }

}



















