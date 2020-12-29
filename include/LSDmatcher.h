//
// Created by lan on 17-12-26.
//

#ifndef ORB_SLAM2_LSDMATCHER_H
#define ORB_SLAM2_LSDMATCHER_H

#include "MapLine.h"
#include "KeyFrame.h"
#include "Frame.h"

namespace Planar_SLAM
{
    class LSDmatcher
    {
    public:
        static const int TH_HIGH, TH_LOW;

        LSDmatcher(float nnratio=0.6, bool checkOri=true);

        int SearchByDescriptor(KeyFrame* pKF, Frame &currentF, std::vector<MapLine*> &vpMapLineMatches);
        int SearchByDescriptor(KeyFrame* pKF, KeyFrame *pKF2, std::vector<MapLine*> &vpMapLineMatches);
        int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono);
        int SearchByProjection(Frame &F, const std::vector<MapLine*> &vpMapLines, const float th=3);
        int SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapLine*> &vpLines, std::vector<MapLine*> &vpMatched, int th);
        int SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapLine *> &vpMatches12, const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th);
        int SerachForInitialize(Frame &InitialFrame, Frame &CurrentFrame, std::vector<std::pair<int,int>> &LineMatches);
        int SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<std::pair<size_t, size_t>> &vMatchedPairs);

        // Project MapLines into KeyFrame and search for duplicated MapLines
        int Fuse(KeyFrame* pKF, const vector<MapLine *> &vpMapLines, const float th=3.0);
//        int Fuse(KeyFrame* pKF, const vector<MapLine *> &vpMapLines);

        int Fuse(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapLine*> &vpLines, float th, vector<MapLine *> &vpReplaceLine);

        static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    protected:
        float RadiusByViewingCos(const float &viewCos);
        float mfNNratio;
        bool mbCheckOrientation;
    };
}


#endif //ORB_SLAM2_LSDMATCHER_H
