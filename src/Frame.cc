#include "Frame.h"
#include "Converter.h"
#include <thread>


using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace Planar_SLAM {

    long unsigned int Frame::nNextId = 0;
    bool Frame::mbInitialComputations = true;
    float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
    float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
    float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

    Frame::Frame() {}

    //Copy Constructor
    Frame::Frame(const Frame &frame)
            : mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft),
              mpORBextractorRight(frame.mpORBextractorRight),
              mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
              mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
              mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
              mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
              mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
              mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
              mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
              mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
              mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
              mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2),
              mLdesc(frame.mLdesc), NL(frame.NL), mvKeylinesUn(frame.mvKeylinesUn),
              mvpMapLines(frame.mvpMapLines),  mvbLineOutlier(frame.mvbLineOutlier), mvKeyLineFunctions(frame.mvKeyLineFunctions),
              mvDepthLine(frame.mvDepthLine), mvLines3D(frame.mvLines3D), mv3DLineforMap(frame.mv3DLineforMap), dealWithLine(frame.dealWithLine),
              blurNumber(frame.blurNumber), vSurfaceNormal(frame.vSurfaceNormal),
              vVanishingDirection(frame.vVanishingDirection), mVF3DLines(frame.mVF3DLines), mvPlaneCoefficients(frame.mvPlaneCoefficients),
              mbNewPlane(frame.mbNewPlane), mvpMapPlanes(frame.mvpMapPlanes), mnPlaneNum(frame.mnPlaneNum), mvbPlaneOutlier(frame.mvbPlaneOutlier),
              mvpParallelPlanes(frame.mvpParallelPlanes), mvpVerticalPlanes(frame.mvpVerticalPlanes),
              vSurfaceNormalx(frame.vSurfaceNormalx), vSurfaceNormaly(frame.vSurfaceNormaly), vSurfaceNormalz(frame.vSurfaceNormalz),
              vSurfacePointx(frame.vSurfacePointx), vSurfacePointy(frame.vSurfacePointy), vSurfacePointz(frame.vSurfacePointz),
              vVanishingLinex(frame.vVanishingLinex),vVanishingLiney(frame.vVanishingLiney),vVanishingLinez(frame.vVanishingLinez),
              mvPlanePoints(frame.mvPlanePoints) {
        for (int i = 0; i < FRAME_GRID_COLS; i++)
            for (int j = 0; j < FRAME_GRID_ROWS; j++)
                mGrid[i][j] = frame.mGrid[i][j];

        if (!frame.mTcw.empty())
            SetPose(frame.mTcw);
    }

    // Extract points,lines and planes by using a multi-thread manner
    Frame::Frame(const cv::Mat &imRGB, const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp,
                 ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf,
                 const float &thDepth, const float &depthMapFactor)
            : mpORBvocabulary(voc), mpORBextractorLeft(extractor),
              mpORBextractorRight(static_cast<ORBextractor *>(NULL)),
              mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth) {
        // Frame ID
        mnId = nNextId++;

        // Scale Level Info
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        cv::Mat depth;
        if (depthMapFactor != 1 || imDepth.type() != CV_32F) {
            imDepth.convertTo(depth, CV_32F, depthMapFactor);
        }
        cv::Mat tmpK = (cv::Mat_<double>(3, 3) << fx, 0, cx,
                0, fy, cy,
                0, 0, 1);
        dealWithLine = true;

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        thread threadLines(&Planar_SLAM::Frame::ExtractLSD, this, imGray, depth, tmpK);
        thread threadPoints(&Planar_SLAM::Frame::ExtractORB, this, 0, imGray);
        thread threadPlanes(&Planar_SLAM::Frame::ComputePlanes, this, depth, imDepth, imRGB, K, depthMapFactor);
        threadPoints.join();
        threadLines.join();
        threadPlanes.join();
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double t12= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        //cout<<"time-frame-feature:" <<t12<<endl;
        N = mvKeys.size();

        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
        double t32= std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();
        //cout<<"time-frame-line:" <<t32<<endl;
        NL = mvKeylinesUn.size();

        if (mvKeys.empty())
            return;

        UndistortKeyPoints();

        ComputeStereoFromRGBD(depth);

        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
        mvpMapLines = vector<MapLine *>(NL, static_cast<MapLine *>(NULL));
        mvbOutlier = vector<bool>(N, false);
        mvbLineOutlier = vector<bool>(NL, false);

        // This is done only for the first Frame (or after a change in the calibration)
        if (mbInitialComputations) {
            ComputeImageBounds(imGray);

            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            mbInitialComputations = false;
        }

        mb = mbf / fx;

        AssignFeaturesToGrid();

        std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
        double t43= std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
        //cout<<"time-frame-plane:" <<t43<<endl;

        mnPlaneNum = mvPlanePoints.size();
        mvpMapPlanes = vector<MapPlane *>(mnPlaneNum, static_cast<MapPlane *>(nullptr));
        mvpParallelPlanes = vector<MapPlane *>(mnPlaneNum, static_cast<MapPlane *>(nullptr));
        mvpVerticalPlanes = vector<MapPlane *>(mnPlaneNum, static_cast<MapPlane *>(nullptr));
        mvPlanePointMatches = vector<vector<MapPoint *>>(mnPlaneNum);
        mvPlaneLineMatches = vector<vector<MapLine *>>(mnPlaneNum);
        mvbPlaneOutlier = vector<bool>(mnPlaneNum, false);
        mvbVerPlaneOutlier = vector<bool>(mnPlaneNum, false);
        mvbParPlaneOutlier = vector<bool>(mnPlaneNum, false);
    }


    void Frame::AssignFeaturesToGrid() {
        int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
        for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
            for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
                mGrid[i][j].reserve(nReserve);

        for (int i = 0; i < N; i++) {
            const cv::KeyPoint &kp = mvKeysUn[i];

            int nGridPosX, nGridPosY;
            if (PosInGrid(kp, nGridPosX, nGridPosY))
                mGrid[nGridPosX][nGridPosY].push_back(i);
        }
    }

    void Frame::ExtractLSD(const cv::Mat &im, const cv::Mat &depth,cv::Mat K) {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        mpLineSegment->ExtractLineSegment(im, mvKeylinesUn, mLdesc, mvKeyLineFunctions);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double t43= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        isLineGood(im, depth, K);
        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
        double t23= std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();

    }

    void Frame::ExtractORB(int flag, const cv::Mat &im) {
        if (flag == 0)
            (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors);
        else
            (*mpORBextractorRight)(im, cv::Mat(), mvKeysRight, mDescriptorsRight);
    }

    // Optimize Lines
    void Frame::isLineGood(const cv::Mat &imGray, const cv::Mat &imDepth, cv::Mat K) {
        mvDepthLine = vector<float>(mvKeylinesUn.size(), -1.0f);
        mvLines3D = vector<Vector6d>(mvKeylinesUn.size(), Eigen::Matrix<double, 6, 1>::Zero());

        for (int i = 0; i < mvKeylinesUn.size(); ++i) { // each line
            double len = cv::norm(mvKeylinesUn[i].getStartPoint() - mvKeylinesUn[i].getEndPoint());
            vector<cv::Point3d> pts3d;
            // iterate through a line
            double numSmp = (double) min((int) len, 50); //number of line points sampled

            pts3d.reserve(numSmp);

            for (int j = 0; j <= numSmp; ++j) {
                // use nearest neighbor to querry depth value
                // assuming position (0,0) is the top-left corner of image, then the
                // top-left pixel's center would be (0.5,0.5)
                cv::Point2d pt = mvKeylinesUn[i].getStartPoint() * (1 - j / numSmp) +
                                 mvKeylinesUn[i].getEndPoint() * (j / numSmp);

                if (pt.x < 0 || pt.y < 0 || pt.x >= imDepth.cols || pt.y >= imDepth.rows) continue;
                int row, col; // nearest pixel for pt
                if ((floor(pt.x) == pt.x) && (floor(pt.y) == pt.y)) { // boundary issue
                    col = max(int(pt.x - 1), 0);
                    row = max(int(pt.y - 1), 0);
                } else {
                    col = int(pt.x);
                    row = int(pt.y);
                }

                float d = -1;
                if (imDepth.at<float>(row, col) <= 0.01) { // no depth info
                    continue;
                } else {
                    d = imDepth.at<float>(row, col);
                }
                cv::Point3d p;

                p.z = d;
                p.x = (col - cx) * p.z * invfx;
                p.y = (row - cy) * p.z * invfy;

                pts3d.push_back(p);

            }

            if (pts3d.size() < 10.0)
                continue;

            RandomLine3d tmpLine;
            vector<RandomPoint3d> rndpts3d;
            rndpts3d.reserve(pts3d.size());
            // compute uncertainty of 3d points
            for (int j = 0; j < pts3d.size(); ++j) {
                rndpts3d.push_back(compPt3dCov(pts3d[j], K, 1));
            }
            // using ransac to extract a 3d line from 3d pts
            tmpLine = extract3dline_mahdist(rndpts3d);

            if (tmpLine.pts.size() / len > 0.4 && cv::norm(tmpLine.A - tmpLine.B) > 0.02) {
                //this line is reliable
                mvDepthLine[i] = std::min(imDepth.at<float>(mvKeylinesUn[i].endPointY, mvKeylinesUn[i].endPointX),
                                          imDepth.at<float>(mvKeylinesUn[i].startPointY, mvKeylinesUn[i].startPointX));

                FrameLine tempLine;
                tempLine.haveDepth = true;
                tempLine.rndpts3d = tmpLine.pts;
                tempLine.direction = tmpLine.director;
                tempLine.direct1 = tmpLine.direct1;
                tempLine.direct2 = tmpLine.direct2;
                tempLine.p = Point2d(mvKeylinesUn[i].endPointX, mvKeylinesUn[i].endPointY);
                tempLine.q = Point2d(mvKeylinesUn[i].startPointX, mvKeylinesUn[i].startPointY);
                mVF3DLines.push_back(tempLine);

                mvLines3D[i] << tmpLine.A.x, tmpLine.A.y, tmpLine.A.z, tmpLine.B.x, tmpLine.B.y, tmpLine.B.z;
            }

        }

    }

    void Frame::lineDescriptorMAD(vector<vector<DMatch>> line_matches, double &nn_mad, double &nn12_mad) const {
        vector<vector<DMatch>> matches_nn, matches_12;
        matches_nn = line_matches;
        matches_12 = line_matches;

        // estimate the NN's distance standard deviation
        double nn_dist_median;
        sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
        nn_dist_median = matches_nn[int(matches_nn.size() / 2)][0].distance;

        for (unsigned int i = 0; i < matches_nn.size(); i++)
            matches_nn[i][0].distance = fabsf(matches_nn[i][0].distance - nn_dist_median);
        sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
        nn_mad = 1.4826 * matches_nn[int(matches_nn.size() / 2)][0].distance;

        // estimate the NN's 12 distance standard deviation
        double nn12_dist_median;
        sort(matches_12.begin(), matches_12.end(), conpare_descriptor_by_NN12_dist());
        nn12_dist_median =
                matches_12[int(matches_12.size() / 2)][1].distance - matches_12[int(matches_12.size() / 2)][0].distance;
        for (unsigned int j = 0; j < matches_12.size(); j++)
            matches_12[j][0].distance = fabsf(matches_12[j][1].distance - matches_12[j][0].distance - nn12_dist_median);
        sort(matches_12.begin(), matches_12.end(), compare_descriptor_by_NN_dist());
        nn12_mad = 1.4826 * matches_12[int(matches_12.size() / 2)][0].distance;
    }


    void Frame::SetPose(cv::Mat Tcw) {
        mTcw = Tcw.clone();
        UpdatePoseMatrices();
    }

    void Frame::UpdatePoseMatrices() {
        mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
        mRwc = mRcw.t();
        mtcw = mTcw.rowRange(0, 3).col(3);
        mOw = -mRcw.t() * mtcw;

        mTwc = cv::Mat::eye(4, 4, mTcw.type());
        mRwc.copyTo(mTwc.rowRange(0, 3).colRange(0, 3));
        mOw.copyTo(mTwc.rowRange(0, 3).col(3));
    }

    bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit) {
        pMP->mbTrackInView = false;

        // 3D in absolute coordinates
        cv::Mat P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const cv::Mat Pc = mRcw * P + mtcw;
        const float &PcX = Pc.at<float>(0);
        const float &PcY = Pc.at<float>(1);
        const float &PcZ = Pc.at<float>(2);

        // Check positive depth
        if (PcZ < 0.0f)
            return false;

        // Project in image and check it is not outside
        const float invz = 1.0f / PcZ;
        const float u = fx * PcX * invz + cx;
        const float v = fy * PcY * invz + cy;

        if (u < mnMinX || u > mnMaxX)
            return false;
        if (v < mnMinY || v > mnMaxY)
            return false;

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const cv::Mat PO = P - mOw;
        const float dist = cv::norm(PO);

        if (dist < minDistance || dist > maxDistance)
            return false;

        // Check viewing angle
        cv::Mat Pn = pMP->GetNormal();

        const float viewCos = PO.dot(Pn) / dist;

        if (viewCos < viewingCosLimit)
            return false;

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist, this);

        // Data used by the tracking
        pMP->mbTrackInView = true;
        pMP->mTrackProjX = u;
        pMP->mTrackProjXR = u - mbf * invz;
        pMP->mTrackProjY = v;
        pMP->mnTrackScaleLevel = nPredictedLevel;
        pMP->mTrackViewCos = viewCos;

        return true;
    }

    bool Frame::isInFrustum(MapLine *pML, float viewingCosLimit) {
        pML->mbTrackInView = false;

        Vector6d P = pML->GetWorldPos();

        cv::Mat SP = (Mat_<float>(3, 1) << P(0), P(1), P(2));
        cv::Mat EP = (Mat_<float>(3, 1) << P(3), P(4), P(5));

        const cv::Mat SPc = mRcw * SP + mtcw;
        const float &SPcX = SPc.at<float>(0);
        const float &SPcY = SPc.at<float>(1);
        const float &SPcZ = SPc.at<float>(2);

        const cv::Mat EPc = mRcw * EP + mtcw;
        const float &EPcX = EPc.at<float>(0);
        const float &EPcY = EPc.at<float>(1);
        const float &EPcZ = EPc.at<float>(2);

        if (SPcZ < 0.0f || EPcZ < 0.0f)
            return false;

        const float invz1 = 1.0f / SPcZ;
        const float u1 = fx * SPcX * invz1 + cx;
        const float v1 = fy * SPcY * invz1 + cy;

        if (u1 < mnMinX || u1 > mnMaxX)
            return false;
        if (v1 < mnMinY || v1 > mnMaxY)
            return false;

        const float invz2 = 1.0f / EPcZ;
        const float u2 = fx * EPcX * invz2 + cx;
        const float v2 = fy * EPcY * invz2 + cy;

        if (u2 < mnMinX || u2 > mnMaxX)
            return false;
        if (v2 < mnMinY || v2 > mnMaxY)
            return false;


        const float maxDistance = pML->GetMaxDistanceInvariance();
        const float minDistance = pML->GetMinDistanceInvariance();

        const cv::Mat OM = 0.5 * (SP + EP) - mOw;
        const float dist = cv::norm(OM);

        if (dist < minDistance || dist > maxDistance)
            return false;


        Vector3d Pn = pML->GetNormal();
        cv::Mat pn = (Mat_<float>(3, 1) << Pn(0), Pn(1), Pn(2));
        const float viewCos = OM.dot(pn) / dist;

        if (viewCos < viewingCosLimit)
            return false;

        const int nPredictedLevel = pML->PredictScale(dist, mfLogScaleFactor);

        pML->mbTrackInView = true;
        pML->mTrackProjX1 = u1;
        pML->mTrackProjY1 = v1;
        pML->mTrackProjX2 = u2;
        pML->mTrackProjY2 = v2;
        pML->mnTrackScaleLevel = nPredictedLevel;
        pML->mTrackViewCos = viewCos;

        return true;
    }


    vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel,
                                            const int maxLevel) const {
        vector<size_t> vIndices;
        vIndices.reserve(N);

        const int nMinCellX = max(0, (int) floor((x - mnMinX - r) * mfGridElementWidthInv));
        if (nMinCellX >= FRAME_GRID_COLS)
            return vIndices;

        const int nMaxCellX = min((int) FRAME_GRID_COLS - 1, (int) ceil((x - mnMinX + r) * mfGridElementWidthInv));
        if (nMaxCellX < 0)
            return vIndices;

        const int nMinCellY = max(0, (int) floor((y - mnMinY - r) * mfGridElementHeightInv));
        if (nMinCellY >= FRAME_GRID_ROWS)
            return vIndices;

        const int nMaxCellY = min((int) FRAME_GRID_ROWS - 1, (int) ceil((y - mnMinY + r) * mfGridElementHeightInv));
        if (nMaxCellY < 0)
            return vIndices;

        const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

        for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
            for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
                const vector<size_t> vCell = mGrid[ix][iy];
                if (vCell.empty())
                    continue;

                for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
                    const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                    if (bCheckLevels) {
                        if (kpUn.octave < minLevel)
                            continue;
                        if (maxLevel >= 0)
                            if (kpUn.octave > maxLevel)
                                continue;
                    }

                    const float distx = kpUn.pt.x - x;
                    const float disty = kpUn.pt.y - y;

                    if (fabs(distx) < r && fabs(disty) < r)
                        vIndices.push_back(vCell[j]);
                }
            }
        }

        return vIndices;
    }

    vector<size_t>  Frame::GetLinesInArea(const float &x1, const float &y1, const float &x2, const float &y2, const float &r,
                          const int minLevel, const int maxLevel) const {
        vector<size_t> vIndices;

        vector<KeyLine> vkl = this->mvKeylinesUn;

        const bool bCheckLevels = (minLevel > 0) || (maxLevel > 0);

        for (size_t i = 0; i < vkl.size(); i++) {
            KeyLine keyline = vkl[i];

            // 1.对比中点距离
            float distance = (0.5 * (x1 + x2) - keyline.pt.x) * (0.5 * (x1 + x2) - keyline.pt.x) +
                             (0.5 * (y1 + y2) - keyline.pt.y) * (0.5 * (y1 + y2) - keyline.pt.y);
            if (distance > r * r)
                continue;

            float slope = (y1 - y2) / (x1 - x2) - keyline.angle;
            if (slope > r * 0.01)
                continue;

            if (bCheckLevels) {
                if (keyline.octave < minLevel)
                    continue;
                if (maxLevel >= 0 && keyline.octave > maxLevel)
                    continue;
            }

            vIndices.push_back(i);
        }

        return vIndices;
    }


    bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY) {
        posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
        posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

        //Keypoint's coordinates are undistorted, which could cause to go out of the image
        if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
            return false;

        return true;
    }


    void Frame::ComputeBoW() {
        if (mBowVec.empty()) {
            vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
            mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
        }
    }

    void Frame::UndistortKeyPoints() {
        if (mDistCoef.at<float>(0) == 0.0) {
            mvKeysUn = mvKeys;
            return;
        }

        // Fill matrix with points
        cv::Mat mat(N, 2, CV_32F);
        for (int i = 0; i < N; i++) {
            mat.at<float>(i, 0) = mvKeys[i].pt.x;
            mat.at<float>(i, 1) = mvKeys[i].pt.y;
        }

        // Undistort points
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        // Fill undistorted keypoint vector
        mvKeysUn.resize(N);
        for (int i = 0; i < N; i++) {
            cv::KeyPoint kp = mvKeys[i];
            kp.pt.x = mat.at<float>(i, 0);
            kp.pt.y = mat.at<float>(i, 1);
            mvKeysUn[i] = kp;
        }
    }

    void Frame::ComputeImageBounds(const cv::Mat &imLeft) {
        if (mDistCoef.at<float>(0) != 0.0) {
            cv::Mat mat(4, 2, CV_32F);
            mat.at<float>(0, 0) = 0.0;
            mat.at<float>(0, 1) = 0.0;
            mat.at<float>(1, 0) = imLeft.cols;
            mat.at<float>(1, 1) = 0.0;
            mat.at<float>(2, 0) = 0.0;
            mat.at<float>(2, 1) = imLeft.rows;
            mat.at<float>(3, 0) = imLeft.cols;
            mat.at<float>(3, 1) = imLeft.rows;

            // Undistort corners
            mat = mat.reshape(2);
            cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
            mat = mat.reshape(1);

            mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
            mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
            mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
            mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));

        } else {
            mnMinX = 0.0f;
            mnMaxX = imLeft.cols;
            mnMinY = 0.0f;
            mnMaxY = imLeft.rows;
        }
    }

    void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth) {
        mvuRight = vector<float>(N, -1);
        mvDepth = vector<float>(N, -1);

        for (int i = 0; i < N; i++) {
            const cv::KeyPoint &kp = mvKeys[i];
            const cv::KeyPoint &kpU = mvKeysUn[i];

            const float &v = kp.pt.y;
            const float &u = kp.pt.x;

            const float d = imDepth.at<float>(v, u);

            if (d > 0) {
                mvDepth[i] = d;
                mvuRight[i] = kpU.pt.x - mbf / d;
            }
        }
    }

    cv::Mat Frame::UnprojectStereo(const int &i) {
        const float z = mvDepth[i];
        if (z > 0) {
            const float u = mvKeysUn[i].pt.x;
            const float v = mvKeysUn[i].pt.y;
            const float x = (u - cx) * z * invfx;
            const float y = (v - cy) * z * invfy;
            cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);
            return mRwc * x3Dc + mOw;
        } else
            return cv::Mat();
    }

    Vector6d Frame::obtain3DLine(const int &i) {
        Vector6d Lines3D = mvLines3D[i];
        cv::Mat Ac = (Mat_<float>(3, 1) << Lines3D(0), Lines3D(1), Lines3D(2));
        cv::Mat A = mRwc * Ac + mOw;
        cv::Mat Bc = (Mat_<float>(3, 1) << Lines3D(3), Lines3D(4), Lines3D(5));
        cv::Mat B = mRwc * Bc + mOw;
        Lines3D << A.at<float>(0, 0), A.at<float>(1, 0), A.at<float>(2, 0),
                B.at<float>(0, 0), B.at<float>(1,0), B.at<float>(2, 0);
        return Lines3D;
    }

    void Frame::ComputePlanes(const cv::Mat &imDepth, const cv::Mat &Depth, const cv::Mat &imRGB, cv::Mat K, float depthMapFactor) {
        planeDetector.readColorImage(imRGB);
        planeDetector.readDepthImage(Depth, K, depthMapFactor);
        planeDetector.runPlaneDetection(imDepth.rows, imDepth.cols);

        for (int i = 0; i < planeDetector.plane_num_; i++) {
            auto &indices = planeDetector.plane_vertices_[i];
            PointCloud::Ptr inputCloud(new PointCloud());
            for (int j : indices) {
                PointT p;
                p.x = (float) planeDetector.cloud.vertices[j][0];
                p.y = (float) planeDetector.cloud.vertices[j][1];
                p.z = (float) planeDetector.cloud.vertices[j][2];

                inputCloud->points.push_back(p);
            }

            auto extractedPlane = planeDetector.plane_filter.extractedPlanes[i];
            double nx = extractedPlane->normal[0];
            double ny = extractedPlane->normal[1];
            double nz = extractedPlane->normal[2];
            double cx = extractedPlane->center[0];
            double cy = extractedPlane->center[1];
            double cz = extractedPlane->center[2];

            float d = (float) -(nx * cx + ny * cy + nz * cz);

            pcl::VoxelGrid<PointT> voxel;
            voxel.setLeafSize(0.1, 0.1, 0.1);

            PointCloud::Ptr coarseCloud(new PointCloud());
            voxel.setInputCloud(inputCloud);
            voxel.filter(*coarseCloud);

            cv::Mat coef = (cv::Mat_<float>(4, 1) << nx, ny, nz, d);

            bool valid = MaxPointDistanceFromPlane(coef, coarseCloud);

            if (!valid) {
                continue;
            }

            mvPlanePoints.push_back(*coarseCloud);

            mvPlaneCoefficients.push_back(coef);
        }

        std::vector<SurfaceNormal> surfaceNormals;

        PointCloud::Ptr inputCloud( new PointCloud() );
        for (int m=0; m<imDepth.rows; m+=3)
        {
            for (int n=0; n<imDepth.cols; n+=3)
            {
                float d = imDepth.ptr<float>(m)[n];
                PointT p;
                p.z = d;
                //cout << "depth:" << d<<endl;
                p.x = ( n - cx) * p.z / fx;
                p.y = ( m - cy) * p.z / fy;

                inputCloud->points.push_back(p);
            }
        }
        inputCloud->height = ceil(imDepth.rows/3.0);
        inputCloud->width = ceil(imDepth.cols/3.0);

        //compute normals
        pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
        ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
        ne.setMaxDepthChangeFactor(0.05f);
        ne.setNormalSmoothingSize(10.0f);
        ne.setInputCloud(inputCloud);
        //计算特征值

        if (inputCloud->size()== 0)
        {
            PCL_ERROR ("Could not estimate a planar model for the given initial plane.\n");
            return;
        }
        ne.compute(*cloud_normals);

        for ( int m=0; m<inputCloud->height; m+=1 ) {
            if(m%2==0) continue;
            for (int n = 0; n < inputCloud->width; n+=1) {
                pcl::Normal normal = cloud_normals->at(n, m);
                SurfaceNormal surfaceNormal;
                if(n%2==0) continue;
                surfaceNormal.normal.x = normal.normal_x;
                surfaceNormal.normal.y = normal.normal_y;
                surfaceNormal.normal.z = normal.normal_z;

                pcl::PointXYZRGB point = inputCloud->at(n, m);
                surfaceNormal.cameraPosition.x = point.x;
                surfaceNormal.cameraPosition.y = point.y;
                surfaceNormal.cameraPosition.z = point.z;
                surfaceNormal.FramePosition.x = n*3;
                surfaceNormal.FramePosition.y = m*3;

                surfaceNormals.push_back(surfaceNormal);
            }
        }

        vSurfaceNormal = surfaceNormals;

    }

    bool Frame::MaxPointDistanceFromPlane(cv::Mat &plane, PointCloud::Ptr pointCloud) {
        auto disTh = Config::Get<double>("Plane.DistanceThreshold");
        bool erased = false;
//        double max = -1;
        double threshold = 0.04;
        int i = 0;
        auto &points = pointCloud->points;
//        std::cout << "points before: " << points.size() << std::endl;
        for (auto &p : points) {
            double absDis = abs(plane.at<float>(0) * p.x +
                                plane.at<float>(1) * p.y +
                                plane.at<float>(2) * p.z +
                                plane.at<float>(3));

            if (absDis > disTh)
                return false;
            i++;
        }

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        // Create the segmentation object
        pcl::SACSegmentation<PointT> seg;
        // Optional
        seg.setOptimizeCoefficients(true);
        // Mandatory
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(disTh);

        seg.setInputCloud(pointCloud);
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size () == 0)
        {
            PCL_ERROR ("Could not estimate a planar model for the given initial plane.\n");
            return false;
        }

        float oldVal = plane.at<float>(3);
        float newVal = coefficients->values[3];

        cv::Mat oldPlane = plane.clone();


        plane.at<float>(0) = coefficients->values[0];
        plane.at<float>(1) = coefficients->values[1];
        plane.at<float>(2) = coefficients->values[2];
        plane.at<float>(3) = coefficients->values[3];

        if ((newVal < 0 && oldVal > 0) || (newVal > 0 && oldVal < 0)) {
            plane = -plane;
//                double dotProduct = plane.dot(oldPlane) / sqrt(plane.dot(plane) * oldPlane.dot(oldPlane));
//                std::cout << "Flipped plane: " << plane.t() << std::endl;
//                std::cout << "Flip plane: " << dotProduct << std::endl;
        }
//        }

        return true;
    }

    cv::Mat Frame::ComputePlaneWorldCoeff(const int &idx) {
        cv::Mat temp;
        cv::transpose(mTcw, temp);
        cv::Mat b = -mOw.t();
        return temp * mvPlaneCoefficients[idx];
    }
} //namespace Planar_SLAM


