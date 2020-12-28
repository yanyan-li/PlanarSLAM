#include "FrameDrawer.h"
#include "Tracking.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<mutex>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace Planar_SLAM
{

    FrameDrawer::FrameDrawer(Map* pMap):mpMap(pMap)
    {
        mState=Tracking::SYSTEM_NOT_READY;
        mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
    }

    cv::Mat FrameDrawer::DrawFrame()
    {
        cv::Mat im;
        vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
        vector<int> vMatches; // Initialization: correspondeces with reference keypoints
        vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
        vector<bool> vbVO, vbMap; // Tracked MapPoints in current frame
        int state; // Tracking state

        vector<KeyLine> vCurrentKeyLines;
        vector<KeyLine> vIniKeyLines;
        vector<bool> vbLineVO, vbLineMap;

        //Copy variables within scoped mutex
        {
            unique_lock<mutex> lock(mMutex);
            state=mState;
            if(mState==Tracking::SYSTEM_NOT_READY)
                mState=Tracking::NO_IMAGES_YET;

            mIm.copyTo(im);

            // points and lines for the initialized situation
            if(mState==Tracking::NOT_INITIALIZED)
            {

                vCurrentKeys = mvCurrentKeys;
                vIniKeys = mvIniKeys;
                vMatches = mvIniMatches;
                vCurrentKeyLines = mvCurrentKeyLines;
                vIniKeyLines = mvIniKeyLines;
            }
            // points and lines for the tracking situation
            else if(mState==Tracking::OK)
            {
                vCurrentKeys = mvCurrentKeys;
                vbVO = mvbVO;
                vbMap = mvbMap;
                vCurrentKeyLines = mvCurrentKeyLines;
                vbLineVO = mvbLineVO;
                vbLineMap = mvbLineMap;
            }
            else if(mState==Tracking::LOST)
            {
                vCurrentKeys = mvCurrentKeys;
                vCurrentKeyLines = mvCurrentKeyLines;
            }
        } // destroy scoped mutex -> release mutex

        if(im.channels()<3) //this should be always true
            cvtColor(im,im,CV_GRAY2BGR);

        //Draw
        if(state==Tracking::NOT_INITIALIZED) //INITIALIZING
        {
            for(unsigned int i=0; i<vMatches.size(); i++)
            {
                if(vMatches[i]>=0)
                {
                    cv::line(im,vIniKeys[i].pt,vCurrentKeys[vMatches[i]].pt,cv::Scalar(0,255,0));
                }
            }
        }
        else if(state==Tracking::OK) //TRACKING
        {
            mnTracked=0;
            mnTrackedVO=0;
            const float r = 5;
            const int n = vCurrentKeys.size();


            if(1) // visualize 2D points and lines
            {
                //visualize points
                for(int j=0;j<NSNx;j+=1)
                {
                    int u=mvSurfaceNormalx[j].x;
                    int v=mvSurfaceNormalx[j].y;
                    if(u>0&&v>0) {
                        cv::circle(im, cv::Point2f(u, v), 1, cv::Scalar(0, 0, 100), -1);
                    }
                }
                for(int j=0;j<NSNy;j+=1)
                {
                    int u=mvSurfaceNormaly[j].x;
                    int v=mvSurfaceNormaly[j].y;
                    if(u>0&&v>0) {
                        cv::circle(im,cv::Point2f(u,v),1,cv::Scalar(0,100,0),-1);
                    }
                }
                for(int j=0;j<NSNz;j+=1)
                {
                    int u=mvSurfaceNormalz[j].x;
                    int v=mvSurfaceNormalz[j].y;
                    if(u>0&&v>0) {
                        cv::circle(im,cv::Point2f(u,v),1,cv::Scalar(100,0,0),-1);
                    }
                }

                //visualize segmented Manhattan Lines
                // Three colors for three directions
                for(size_t j=0;j<NSLx;j++)
                {
                    int u1 = mvStructLinex[j][2].x; int v1 = mvStructLinex[j][2].y;
                    int u2 = mvStructLinex[j][3].x; int v2 = mvStructLinex[j][3].y;
                    cv::line(im, cv::Point2f(u1,v1),cv::Point2f(u2,v2), cv::Scalar(255, 0, 255),4);
                }
                for(size_t j=0;j<NSLy;j++)
                {
                    int u1 = mvStructLiney[j][2].x; int v1 = mvStructLiney[j][2].y;
                    int u2 = mvStructLiney[j][3].x; int v2 = mvStructLiney[j][3].y;
                    cv::line(im, cv::Point2f(u1,v1),cv::Point2f(u2,v2), cv::Scalar(0, 255, 0),4);
                }
                for(size_t j=0;j<NSLz;j++)
                {

                    int u1 = mvStructLinez[j][2].x; int v1 = mvStructLinez[j][2].y;
                    int u2 = mvStructLinez[j][3].x; int v2 = mvStructLinez[j][3].y;
                    cv::line(im, cv::Point2f(u1,v1),cv::Point2f(u2,v2), cv::Scalar(255, 0, 0),4);
                }
            }

            for(int i=0;i<n;i++)
            {
                if(vbVO[i] || vbMap[i])
                {
                    cv::Point2f pt1,pt2;
                    pt1.x=vCurrentKeys[i].pt.x-r;
                    pt1.y=vCurrentKeys[i].pt.y-r;
                    pt2.x=vCurrentKeys[i].pt.x+r;
                    pt2.y=vCurrentKeys[i].pt.y+r;

                    // This is a match to a MapPoint in the map
                    if(vbMap[i])
                    {
                        //cv::rectangle(im,pt1,pt2,cv::Scalar(155,255,155));
                        cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(155,255,155),-1);
                        mnTracked++;
                    }
                    else // This is match to a "visual odometry" MapPoint created in the last frame
                    {
                        //cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                        cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(155,255,155),-1);
                        mnTrackedVO++;
                    }
                }
            }
        }

        cv::Mat imWithInfo;
        DrawTextInfo(im,state, imWithInfo);

        return imWithInfo;
    }


    void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
    {
        stringstream s;
        if(nState==Tracking::NO_IMAGES_YET)
            s << " WAITING FOR IMAGES";
        else if(nState==Tracking::NOT_INITIALIZED)
            s << " TRYING TO INITIALIZE ";
        else if(nState==Tracking::OK)
        {
            if(!mbOnlyTracking)
                s << "SLAM MODE |  ";
            else
                s << "LOCALIZATION | ";
            int nKFs = mpMap->KeyFramesInMap();
            int nMPs = mpMap->MapPointsInMap();
            s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
            if(mnTrackedVO>0)
                s << ", + VO matches: " << mnTrackedVO;
        }
        else if(nState==Tracking::LOST)
        {
            s << " TRACK LOST. TRYING TO RELOCALIZE ";
        }
        else if(nState==Tracking::SYSTEM_NOT_READY)
        {
            s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
        }

        int baseline=0;
        cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);

        imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());
        im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
        imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
        cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);

    }

    void FrameDrawer::Update(Tracking *pTracker)
    {
        unique_lock<mutex> lock(mMutex);
        pTracker->mImGray.copyTo(mIm);
        mvCurrentKeys=pTracker->mCurrentFrame.mvKeys;
        N = mvCurrentKeys.size();
        mvbVO = vector<bool>(N,false);
        mvbMap = vector<bool>(N,false);
        mbOnlyTracking = pTracker->mbOnlyTracking;

        mvSurfaceNormalx=pTracker->mCurrentFrame.vSurfaceNormalx;
        mvSurfaceNormaly=pTracker->mCurrentFrame.vSurfaceNormaly;
        mvSurfaceNormalz=pTracker->mCurrentFrame.vSurfaceNormalz;
        NSNx=mvSurfaceNormalx.size();
        NSNy=mvSurfaceNormaly.size();
        NSNz=mvSurfaceNormalz.size();

        mvStructLinex = pTracker->mCurrentFrame.vVanishingLinex;
        mvStructLiney = pTracker->mCurrentFrame.vVanishingLiney;
        mvStructLinez = pTracker->mCurrentFrame.vVanishingLinez;
        NSLx = mvStructLinex.size();
        NSLy = mvStructLiney.size();
        NSLz = mvStructLinez.size();

        mvCurrentKeyLines = pTracker->mCurrentFrame.mvKeylinesUn;
        NL = mvCurrentKeyLines.size();  //自己添加的
        mvbLineVO = vector<bool>(NL, false);
        mvbLineMap = vector<bool>(NL, false);

        if(pTracker->mLastProcessedState==Tracking::NOT_INITIALIZED)
        {
            mvIniKeys=pTracker->mInitialFrame.mvKeys;
            mvIniMatches=pTracker->mvIniMatches;
            //线特征的
            mvIniKeyLines = pTracker->mInitialFrame.mvKeylinesUn;
        }
        else if(pTracker->mLastProcessedState==Tracking::OK)
        {
            for(int i=0;i<N;i++)
            {
                MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                {
                    if(!pTracker->mCurrentFrame.mvbOutlier[i])
                    {
                        if(pMP->Observations()>0)
                            mvbMap[i]=true;
                        else
                            mvbVO[i]=true;
                    }
                }
            }

            for(int i=0; i<NL; i++)
            {
                MapLine* pML = pTracker->mCurrentFrame.mvpMapLines[i];
                if(pML)
                {
                    if(!pTracker->mCurrentFrame.mvbLineOutlier[i])
                    {
                        if(pML->Observations()>0)
                            mvbLineMap[i] = true;
                        else
                            mvbLineVO[i] = true;
                    }
                }
            }
        }
        mState=static_cast<int>(pTracker->mLastProcessedState);
    }

} //namespace Planar_SLAM