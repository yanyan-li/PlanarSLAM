//
// Created by fishmarch on 19-5-29.
//

#ifndef ORB_SLAM2_EDGEPLANE_H
#define ORB_SLAM2_EDGEPLANE_H

#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/core/hyper_graph_action.h"
#include "Thirdparty/g2o/g2o/core/eigen_types.h"
#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"
#include "Thirdparty/g2o/g2o/stuff/misc.h"
#include "g2oAddition/Plane3D.h"
#include "g2oAddition/VertexPlane.h"
#include "Converter.h"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace g2o {
    class EdgePlane : public BaseBinaryEdge<3, Plane3D, VertexPlane, VertexSE3Expmap> {
        typedef pcl::PointXYZRGB PointT;
        typedef pcl::PointCloud<PointT> PointCloud;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgePlane() {}

        void computeError() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[1]);
            const VertexPlane *planeVertex = static_cast<const VertexPlane *>(_vertices[0]);

            const Plane3D &plane = planeVertex->estimate();
            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n * plane;

//            PointCloud::Ptr localPoints(new PointCloud());
//            pcl::transformPointCloud(*planePoints, *localPoints, w2n.matrix());
//
//            _error = localPlane.ominus(_measurement, localPoints);

            _error = localPlane.ominus(_measurement);
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

        bool isDepthPositive() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[1]);
            const VertexPlane *planeVertex = static_cast<const VertexPlane *>(_vertices[0]);

            const Plane3D &plane = planeVertex->estimate();
            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n * plane;

            return localPlane.distance() > 0;
        }

        virtual bool read(std::istream &is) {
            Vector4D v;
            is >> v(0) >> v(1) >> v(2) >> v(3);
            setMeasurement(Plane3D(v));
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }

        virtual bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }

        virtual void linearizeOplus() {
            BaseBinaryEdge::linearizeOplus();

//        const VertexSE3Expmap* poseVertex = static_cast<const VertexSE3Expmap*>(_vertices[1]);
//        const VertexPlane* planeVertex = static_cast<const VertexPlane*>(_vertices[0]);
//
//        const Plane3D& plane = planeVertex->estimate();
//        // measurement function: remap the plane in global coordinates
//        Isometry3D w2n = poseVertex->estimate();
//        Plane3D localPlane = w2n*plane;
//
//        Vector4D vector = localPlane.coeffs();
//        double n_cx = vector(0);
//        double n_cy = vector(1);
//        double n_cz = vector(2);
//        double denominator1 = std::pow(n_cx, 2) + std::pow(n_cy, 2);
//        double denominator2 = std::sqrt(1 - std::pow(n_cz, 2));
//
//        _jacobianOplusXi(0, 0) = (n_cx * n_cz) / denominator1;
//        _jacobianOplusXi(0, 1) = (n_cy * n_cz) / denominator1;
//        _jacobianOplusXi(0, 2) = -1;
//        _jacobianOplusXi(0, 3) = 0;
//        _jacobianOplusXi(0, 4) = 0;
//        _jacobianOplusXi(0, 5) = 0;
//
//        _jacobianOplusXi(1, 0) = -n_cy / denominator2;
//        _jacobianOplusXi(1, 1) = -n_cx / denominator2;
//        _jacobianOplusXi(1, 2) = 0;
//        _jacobianOplusXi(0, 3) = 0;
//        _jacobianOplusXi(0, 4) = 0;
//        _jacobianOplusXi(0, 5) = 0;
//
//        _jacobianOplusXi(2, 0) = 0;
//        _jacobianOplusXi(2, 1) = 0;
//        _jacobianOplusXi(2, 2) = 0;
//        _jacobianOplusXi(2, 3) = n_cx;
//        _jacobianOplusXi(2, 4) = n_cy;
//        _jacobianOplusXi(2, 5) = n_cz;
        }
        PointCloud::Ptr planePoints;
    };

    class EdgePlaneOnlyPose : public BaseUnaryEdge<3, Plane3D, VertexSE3Expmap> {
        typedef pcl::PointXYZRGB PointT;
        typedef pcl::PointCloud<PointT> PointCloud;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgePlaneOnlyPose() {}

        void computeError() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[0]);
            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n * Xw;

//            PointCloud::Ptr localPoints(new PointCloud());
//            pcl::transformPointCloud(*planePoints, *localPoints, w2n.matrix());
//
//            _error = localPlane.ominus(_measurement, localPoints);
            _error = localPlane.ominus(_measurement);
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

        bool isDepthPositive() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[0]);
            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n * Xw;

            return localPlane.distance() > 0;
        }

        virtual bool read(std::istream &is) {
            Vector4D v;
            is >> v(0) >> v(1) >> v(2) >> v(3);
            setMeasurement(Plane3D(v));
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }

        virtual bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }

        virtual void linearizeOplus() {
            BaseUnaryEdge::linearizeOplus();

//        const VertexSE3Expmap* poseVertex = static_cast<const VertexSE3Expmap*>(_vertices[1]);
//        const VertexPlane* planeVertex = static_cast<const VertexPlane*>(_vertices[0]);
//
//        const Plane3D& plane = planeVertex->estimate();
//        // measurement function: remap the plane in global coordinates
//        Isometry3D w2n = poseVertex->estimate();
//        Plane3D localPlane = w2n*plane;
//
//        Vector4D vector = localPlane.coeffs();
//        double n_cx = vector(0);
//        double n_cy = vector(1);
//        double n_cz = vector(2);
//        double denominator1 = std::pow(n_cx, 2) + std::pow(n_cy, 2);
//        double denominator2 = std::sqrt(1 - std::pow(n_cz, 2));
//
//        _jacobianOplusXi(0, 0) = (n_cx * n_cz) / denominator1;
//        _jacobianOplusXi(0, 1) = (n_cy * n_cz) / denominator1;
//        _jacobianOplusXi(0, 2) = -1;
//        _jacobianOplusXi(0, 3) = 0;
//        _jacobianOplusXi(0, 4) = 0;
//        _jacobianOplusXi(0, 5) = 0;
//
//        _jacobianOplusXi(1, 0) = -n_cy / denominator2;
//        _jacobianOplusXi(1, 1) = -n_cx / denominator2;
//        _jacobianOplusXi(1, 2) = 0;
//        _jacobianOplusXi(0, 3) = 0;
//        _jacobianOplusXi(0, 4) = 0;
//        _jacobianOplusXi(0, 5) = 0;
//
//        _jacobianOplusXi(2, 0) = 0;
//        _jacobianOplusXi(2, 1) = 0;
//        _jacobianOplusXi(2, 2) = 0;
//        _jacobianOplusXi(2, 3) = n_cx;
//        _jacobianOplusXi(2, 4) = n_cy;
//        _jacobianOplusXi(2, 5) = n_cz;
        }

        Plane3D Xw;
        PointCloud::Ptr planePoints;
    };

    class EdgePlaneOnlyTranslation : public BaseUnaryEdge<3, Plane3D, VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgePlaneOnlyTranslation() {}

        void computeError() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[0]);

            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n + Xc;

            _error = localPlane.ominus(_measurement);
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

        bool isDepthPositive() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[0]);

            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n + Xc;

            return localPlane.distance() > 0;
        }

        virtual bool read(std::istream &is) {
            Vector4D v;
            is >> v(0) >> v(1) >> v(2) >> v(3);
            setMeasurement(Plane3D(v));
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }

        virtual bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }

        virtual void linearizeOplus() {
            BaseUnaryEdge::linearizeOplus();

//        const VertexSE3Expmap* poseVertex = static_cast<const VertexSE3Expmap*>(_vertices[0]);
//
//        // measurement function: remap the plane in global coordinates
//        Isometry3D w2n = poseVertex->estimate();
//        Plane3D localPlane = w2n+Xc;
//
//        Vector4D vector = localPlane.coeffs();
//        double n_cx = vector(0);
//        double n_cy = vector(1);
//        double n_cz = vector(2);
//        double denominator1 = std::pow(n_cx, 2) + std::pow(n_cy, 2);
//        double denominator2 = std::sqrt(1 - std::pow(n_cz, 2));

            _jacobianOplusXi(0, 0) = 0;
            _jacobianOplusXi(0, 1) = 0;
            _jacobianOplusXi(0, 2) = 0;
//        _jacobianOplusXi(0, 3) = 0;
//        _jacobianOplusXi(0, 4) = 0;
//        _jacobianOplusXi(0, 5) = 0;

            _jacobianOplusXi(1, 0) = 0;
            _jacobianOplusXi(1, 1) = 0;
            _jacobianOplusXi(1, 2) = 0;
//        _jacobianOplusXi(1, 3) = 0;
//        _jacobianOplusXi(1, 4) = 0;
//        _jacobianOplusXi(1, 5) = 0;

            _jacobianOplusXi(2, 0) = 0;
            _jacobianOplusXi(2, 1) = 0;
            _jacobianOplusXi(2, 2) = 0;
//        _jacobianOplusXi(2, 3) = n_cx;
//        _jacobianOplusXi(2, 4) = n_cy;
//        _jacobianOplusXi(2, 5) = n_cz;
        }

        Plane3D Xc;
    };

    class EdgePlaneSim3Project : public g2o::BaseBinaryEdge<3, Plane3D, VertexPlane, g2o::VertexSim3Expmap> {
        typedef pcl::PointXYZRGB PointT;
        typedef pcl::PointCloud<PointT> PointCloud;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgePlaneSim3Project() {}

        void computeError() {
            const g2o::VertexSim3Expmap *v1 = static_cast<const g2o::VertexSim3Expmap *>(_vertices[1]);
            const g2o::VertexPlane *v2 = static_cast<const g2o::VertexPlane *>(_vertices[0]);

            const Plane3D &plane = v2->estimate();
            g2o::Sim3 sim3 = v1->estimate();

            Vector4D coeffs = plane._coeffs;
            Vector4D localCoeffs;
            Matrix3D R = sim3.rotation().matrix();
            Vector3D t = sim3.translation();
            localCoeffs.head<3>() = sim3.scale() * (R * coeffs.head<3>());
            localCoeffs(3) = coeffs(3) - t.dot(localCoeffs.head<3>());
            if (localCoeffs(3) < 0.0)
                localCoeffs = -localCoeffs;
            Plane3D localPlane = Plane3D(localCoeffs);

//            Matrix4d pose;
//
//            pose << R(0,0), R(0,1), R(0,2), t(0,3),
//                    R(1,0), R(1,1), R(1,2), t(1,3),
//                    R(2,0), R(2,1), R(2,2), t(2,3),
//                    0, 0, 0, 1;
//
//            PointCloud::Ptr localPoints(new PointCloud());
//            pcl::transformPointCloud(*planePoints, *localPoints, pose);
//
//            _error = localPlane.ominus(_measurement, localPoints);

            _error = localPlane.ominus(_measurement);
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

        virtual bool read(std::istream &is) {
            Vector4D v;
            is >> v(0) >> v(1) >> v(2) >> v(3);
            setMeasurement(Plane3D(v));
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }

        virtual bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }

        PointCloud::Ptr planePoints;
    };

    class EdgePlaneInverseSim3Project : public g2o::BaseBinaryEdge<3, Plane3D, VertexPlane, g2o::VertexSim3Expmap> {
        typedef pcl::PointXYZRGB PointT;
        typedef pcl::PointCloud<PointT> PointCloud;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgePlaneInverseSim3Project() {}

        void computeError() {
            const g2o::VertexSim3Expmap *v1 = static_cast<const g2o::VertexSim3Expmap *>(_vertices[1]);
            const g2o::VertexPlane *v2 = static_cast<const g2o::VertexPlane *>(_vertices[0]);

            const Plane3D &plane = v2->estimate();
            g2o::Sim3 sim3 = v1->estimate().inverse();

            Vector4D coeffs = plane._coeffs;
            Vector4D localCoeffs;
            Matrix3D R = sim3.rotation().matrix();
            Vector3D t = sim3.translation();
            localCoeffs.head<3>() = sim3.scale() * (R * coeffs.head<3>());
            localCoeffs(3) = coeffs(3) - t.dot(localCoeffs.head<3>());
            if (localCoeffs(3) < 0.0)
                localCoeffs = -localCoeffs;
            Plane3D localPlane = Plane3D(localCoeffs);

//            Matrix4d pose;
//
//            pose << R(0,0), R(0,1), R(0,2), t(0,3),
//                    R(1,0), R(1,1), R(1,2), t(1,3),
//                    R(2,0), R(2,1), R(2,2), t(2,3),
//                    0, 0, 0, 1;
//
//            PointCloud::Ptr localPoints(new PointCloud());
//            pcl::transformPointCloud(*planePoints, *localPoints, pose);
//
//            _error = localPlane.ominus(_measurement, localPoints);

            _error = localPlane.ominus(_measurement);
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

        virtual bool read(std::istream &is) {
            Vector4D v;
            is >> v(0) >> v(1) >> v(2) >> v(3);
            setMeasurement(Plane3D(v));
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }

        virtual bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }

        PointCloud::Ptr planePoints;
    };

    class EdgePlaneSim3GeoProject : public g2o::BaseUnaryEdge<3, Plane3D, g2o::VertexSim3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgePlaneSim3GeoProject() {}

        void computeError() {
            const g2o::VertexSim3Expmap *v1 = static_cast<const g2o::VertexSim3Expmap *>(_vertices[0]);

            _error(0) = (_measurement.normal().dot(v1->estimate().map(point)) + _measurement.distance()) / _measurement.normal().norm();
            _error(1) = 0;
            _error(2) = 0;
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

        virtual bool read(std::istream &is) {
            Vector4D v;
            is >> v(0) >> v(1) >> v(2) >> v(3);
            setMeasurement(Plane3D(v));
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }

        virtual bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }

        virtual void linearizeOplus() {
            BaseUnaryEdge::linearizeOplus();

//            _jacobianOplusXi(0, 0) = 0;
//            _jacobianOplusXi(0, 1) = 0;
//            _jacobianOplusXi(0, 2) = 0;
//            _jacobianOplusXi(0, 3) = 0;
//            _jacobianOplusXi(0, 4) = 0;
//            _jacobianOplusXi(0, 5) = 0;
//
            _jacobianOplusXi(1, 0) = 0;
            _jacobianOplusXi(1, 1) = 0;
            _jacobianOplusXi(1, 2) = 0;
            _jacobianOplusXi(1, 3) = 0;
            _jacobianOplusXi(1, 4) = 0;
            _jacobianOplusXi(1, 5) = 0;

            _jacobianOplusXi(2, 0) = 0;
            _jacobianOplusXi(2, 1) = 0;
            _jacobianOplusXi(2, 2) = 0;
            _jacobianOplusXi(2, 3) = 0;
            _jacobianOplusXi(2, 4) = 0;
            _jacobianOplusXi(2, 5) = 0;
        }

        Vector3d point;
    };
}

#endif //ORB_SLAM2_EDGEPLANE_H
