#include "PlaneMatcher.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace Planar_SLAM
{
    PlaneMatcher::PlaneMatcher(float dTh, float aTh, float verTh, float parTh):dTh(dTh), aTh(aTh), verTh(verTh), parTh(parTh) {}
    int PlaneMatcher::SearchMapByCoefficients(Frame &pF, const std::vector<MapPlane *> &vpMapPlanes) {
        pF.mbNewPlane = false;

        int nmatches = 0;

        for (int i = 0; i < pF.mnPlaneNum; ++i) {

            cv::Mat pM = pF.ComputePlaneWorldCoeff(i);

            float ldTh = dTh;
            float lverTh = verTh;
            float lparTh = parTh;

            bool found = false;
            for (auto vpMapPlane : vpMapPlanes) {
                if (vpMapPlane->isBad())
                    continue;

                cv::Mat pW = vpMapPlane->GetWorldPos();

                float angle = pM.at<float>(0, 0) * pW.at<float>(0, 0) +
                              pM.at<float>(1, 0) * pW.at<float>(1, 0) +
                              pM.at<float>(2, 0) * pW.at<float>(2, 0);

                // associate plane
                if ((angle > aTh || angle < -aTh))
                {
                    double dis = PointDistanceFromPlane(pM, vpMapPlane->mvPlanePoints);
                    if(dis < ldTh) {
                        ldTh = dis;
                        pF.mvpMapPlanes[i] = static_cast<MapPlane*>(nullptr);
                        pF.mvpMapPlanes[i] = vpMapPlane;
                        found = true;
                        continue;
                    }
                }

                // vertical planes
                if (angle < lverTh && angle > -lverTh) {
                    lverTh = abs(angle);
                    pF.mvpVerticalPlanes[i] = static_cast<MapPlane*>(nullptr);
                    pF.mvpVerticalPlanes[i] = vpMapPlane;
                    continue;
                }

                //parallel planes
                if (angle > lparTh || angle < -lparTh) {
                    lparTh = abs(angle);
                    pF.mvpParallelPlanes[i] = static_cast<MapPlane*>(nullptr);
                    pF.mvpParallelPlanes[i] = vpMapPlane;
                }
            }

            if (found) {
                nmatches++;
            }
        }

        return nmatches;
    }
    double PlaneMatcher::PointDistanceFromPlane(const cv::Mat &plane, PointCloud::Ptr pointCloud) {
        double res = 100;
        for(auto p : pointCloud->points){
            double dis = abs(plane.at<float>(0, 0) * p.x +
                             plane.at<float>(1, 0) * p.y +
                             plane.at<float>(2, 0) * p.z +
                             plane.at<float>(3, 0));
            if(dis < res)
                res = dis;
        }
        return res;
    }
}
