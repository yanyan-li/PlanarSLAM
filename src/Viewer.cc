#include "Viewer.h"
#include <pangolin/pangolin.h>

#include <mutex>

using namespace std;

namespace Planar_SLAM
{

Viewer::Viewer(System* pSystem, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Tracking *pTracking, const string &strSettingPath):
    mpSystem(pSystem), mpFrameDrawer(pFrameDrawer),mpMapDrawer(pMapDrawer), mpTracker(pTracking),
    mbFinishRequested(false), mbFinished(true), mbStopped(false), mbStopRequested(false)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    float fps = fSettings["Camera.fps"];
    if(fps<1)
        fps=30;
    mT = 1e3/fps;

    mImageWidth = fSettings["Camera.width"];
    mImageHeight = fSettings["Camera.height"];
    if(mImageWidth<1 || mImageHeight<1)
    {
        mImageWidth = 640;
        mImageHeight = 480;
    }

    mViewpointX = fSettings["Viewer.ViewpointX"];
    mViewpointY = fSettings["Viewer.ViewpointY"];
    mViewpointZ = fSettings["Viewer.ViewpointZ"];
    mViewpointF = fSettings["Viewer.ViewpointF"];
}


void Viewer::RunWithPLP()
{
        mbFinished = false;
        mbStopped = false;

        pangolin::CreateWindowAndBind("StructureSLAM: 3D Map", 1024, 768);  //用Pangolin创建显示窗口

        // 3D Mouse handler requires depth testing to be enabled
        glEnable(GL_DEPTH_TEST);

        // Issue specific OpenGL we might need
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
        pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
        pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);     //显示特征点
        pangolin::Var<bool> menuShowLines("menu.Show Lines", true, true);       //显示特征线
        pangolin::Var<bool> menuShowPlanes("menu.Show Planes",true,true);
        pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
        pangolin::Var<bool> menuShowGraph("menu.Show Graph",true,true);
        pangolin::Var<bool> menuVideo("menu.Save 2D Frames",false,true);
        pangolin::Var<bool> menuVideo3D("menu.Save Sparse Map",false,true);
        pangolin::Var<bool> menuScreenshot("menu.Screenshot 2D Frame",false, false);
        pangolin::Var<bool> menuScreenshotMesh("menu.Screenshot Mesh",false, false);

        pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000),
                pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0)
        );

        // Add named OpenGL viewport to window and provide 3D Handler
        pangolin::View& d_cam = pangolin::CreateDisplay()
                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
                .SetHandler(new pangolin::Handler3D(s_cam));

        pangolin::OpenGlMatrix Twc;
        Twc.SetIdentity();
        bool bFollow = true;
        bool bLocalizationMode = false;

        int cnt_2d = 0;
        int cnt_3d = 0;
        char buffer[256];
        while(1)
        {

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc);

            if(menuFollowCamera && bFollow)
            {
                s_cam.Follow(Twc);
            }
            else if(menuFollowCamera && !bFollow)
            {
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
                s_cam.Follow(Twc);
                bFollow = true;
            }
            else if(!menuFollowCamera && bFollow)
            {
                bFollow = false;
            }



            d_cam.Activate(s_cam);
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            mpMapDrawer->DrawCurrentCamera(Twc);
            if(menuShowKeyFrames || menuShowGraph)
                mpMapDrawer->DrawKeyFrames(menuShowKeyFrames, menuShowGraph);
            if(menuShowPoints)
                mpMapDrawer->DrawMapPoints();
            if(menuShowLines)
                mpMapDrawer->DrawMapLines();
            if(menuShowPlanes)
                mpMapDrawer->DrawMapPlanes();

            pangolin::FinishFrame();

            cv::Mat im = mpFrameDrawer->DrawFrame();
            cv::imshow("Current 2D Frame", im);
            cv::waitKey(mT);



            if(menuScreenshot)
            {
                cv::imwrite("Screenshot.png", im);
                menuScreenshot = false;
            }

            if(menuScreenshotMesh)
            {
                mpTracker->SaveMesh("Screenshot.ply");
                menuScreenshot = false;
            }

            if (menuVideo) {
                // Save video
                sprintf(buffer, "2d/%06d.png", cnt_2d);
                cv::imwrite(buffer, im);
                cnt_2d++;
            }
            if(menuVideo3D)
            {
                sprintf(buffer,"3d/%04d",cnt_3d);
                pangolin::SaveWindowOnRender(buffer);
                cnt_3d++;

            }

            if(Stop())
            {
                while(isStopped())
                {
                    usleep(3000);
                }
            }

            if(CheckFinish())
                break;
        }

        SetFinish();

    }


void Viewer::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Viewer::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool Viewer::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void Viewer::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(!mbStopped)
        mbStopRequested = true;
}

bool Viewer::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool Viewer::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);

    if(mbFinishRequested)
        return false;
    else if(mbStopRequested)
    {
        mbStopped = true;
        mbStopRequested = false;
        return true;
    }

    return false;

}

void Viewer::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
}

} //namespace Planar_SLAM
