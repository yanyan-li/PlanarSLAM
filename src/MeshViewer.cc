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

    void MeshViewer::SaveMeshModel(const string &filename) {
        std::vector<Planar_SLAM::MapPlane *> PlaneMap = mMap->GetAllMapPlanes();

        pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud(new pcl::PointCloud<pcl::PointXYZ>());

        // 在
        //std::ofstream planes("planePC.txt");
        int num =0;
        for (auto pMP : PlaneMap) {

            // obtain parameters of the plane
            //pMP->GetWorldPos();
            cv::Mat parameter = pMP->GetWorldPos();
            float nx = parameter.at<float>(0,0);
            float ny = parameter.at<float>(1,0);
            float nz = parameter.at<float>(2,0);
            float d = parameter.at<float>(3,0);

            pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud_part(new pcl::PointCloud<pcl::PointXYZ>());
            for (auto &planePoint : pMP->mvPlanePoints.get()->points) {

                double tempx = planePoint.x;  double tempy =planePoint.y; double tempz =planePoint.z;
                double distance = d+nx*tempx+nz*tempz+ny*tempy;

                //int i = 0;
                pcl::PointXYZ p;
                //cout<<nx*distance<<", "<<ny*distance<<", "<<nz*distance<<endl;
                // outliers
                if(nx*distance>0.1||ny*distance>0.1||nz*distance>0.1)
                    continue;
                p.x = planePoint.x-nx*distance;
                p.y = planePoint.y-ny*distance;
                p.z = planePoint.z-nz*distance;
                meshCloud->points.push_back(p);
                meshCloud_part->points.push_back(p);
                //planes << p.x<<" "<<p.y<<" "<<p.z<<" "<<ir<<" "<<ig<<" "<<ib<<endl;  //把字符串内容"This is a Test!"，写入Test.txt文件
            }

            // draw each instance
            if (meshCloud_part->points.size() > 0&&false) {
                pcl::PolygonMesh cloud_mesh;
                Mesh::create_mesh(meshCloud_part, 2, 1, cloud_mesh);
                Mesh::SaveMeshModel(cloud_mesh,to_string(num)+filename);
            }
            num++;
        }

        //planes.close();

        if (meshCloud->points.size() > 0) {
            pcl::PolygonMesh cloud_mesh;
            Mesh::create_mesh(meshCloud, 2, 1, cloud_mesh);
            Mesh::SaveMeshModel(cloud_mesh,filename);
        }
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

                cv::Mat parameter = pMP->GetWorldPos();
                float nx = parameter.at<float>(0,0);
                float ny = parameter.at<float>(1,0);
                float nz = parameter.at<float>(2,0);
                float d = parameter.at<float>(3,0);

                pcl::PointCloud<pcl::PointXYZ>::Ptr meshCloud(new pcl::PointCloud<pcl::PointXYZ>());
                for(auto& planePoint : pMP->mvPlanePoints.get()->points){

                    double tempx = planePoint.x;  double tempy =planePoint.y; double tempz =planePoint.z;
                    double distance = d+nx*tempx+nz*tempz+ny*tempy;

                    //int i = 0;
                    pcl::PointXYZ p;
                    //cout<<nx*distance<<", "<<ny*distance<<", "<<nz*distance<<endl;
                    // outliers
                    if(nx*distance>0.1||ny*distance>0.1||nz*distance>0.1)
                        continue;
                    p.x = planePoint.x-nx*distance;
                    p.y = planePoint.y-ny*distance;
                    p.z = planePoint.z-nz*distance;
                    //int i = 0;
//                    pcl::PointXYZ p;
//                    p.x = planePoint.x;
//                    p.y = planePoint.y;
//                    p.z = planePoint.z;

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



















