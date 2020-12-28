//
// Created by fishmarch on 19-6-8.
//

#ifndef ORB_SLAM2_EDGEVERTICALPLANE_H
#define ORB_SLAM2_EDGEVERTICALPLANE_H

#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/core/hyper_graph_action.h"
#include "Thirdparty/g2o/g2o/core/eigen_types.h"
#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/stuff/misc.h"
#include "g2oAddition/Plane3D.h"
#include "g2oAddition/VertexPlane.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace g2o {
    class EdgeVerticalPlane : public BaseBinaryEdge<2, Plane3D, VertexPlane, VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeVerticalPlane() {}

        void computeError() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[1]);
            const VertexPlane *planeVertex = static_cast<const VertexPlane *>(_vertices[0]);

            const Plane3D &plane = planeVertex->estimate();
            // measurement function: remap the plane in global coordinates
            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n * plane;

            _error = localPlane.ominus_ver(_measurement);
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

        bool isDepthPositive() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[1]);
            const VertexPlane *planeVertex = static_cast<const VertexPlane *>(_vertices[0]);

            const Plane3D &plane = planeVertex->estimate();
            // measurement function: remap the plane in global coordinates
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
        }
    };

    class EdgeVerticalPlaneOnlyPose : public BaseUnaryEdge<2, Plane3D, VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeVerticalPlaneOnlyPose() {}

        void computeError() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[0]);
            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n * Xw;

            _error = localPlane.ominus_ver(_measurement);
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
        }

        Plane3D Xw;
    };

    class EdgeVerticalPlaneOnlyTranslation : public BaseUnaryEdge<2, Plane3D, VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeVerticalPlaneOnlyTranslation() {}

        void computeError() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[0]);

            // measurement function: remap the plane in global coordinates
            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n + Xc;

            _error = localPlane.ominus_ver(_measurement);
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

        bool isDepthPositive() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[1]);
            const VertexPlane *planeVertex = static_cast<const VertexPlane *>(_vertices[0]);

            const Plane3D &plane = planeVertex->estimate();
            // measurement function: remap the plane in global coordinates
            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n + plane;

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
        }

        Plane3D Xc;
    };

    class EdgeVerticalPlaneSim3Project : public g2o::BaseBinaryEdge<2, Plane3D, VertexPlane, g2o::VertexSim3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeVerticalPlaneSim3Project() {}

        void computeError() {
            const g2o::VertexSim3Expmap *v1 = static_cast<const g2o::VertexSim3Expmap *>(_vertices[1]);
            const g2o::VertexPlane *v2 = static_cast<const g2o::VertexPlane *>(_vertices[0]);

            const Plane3D &plane = v2->estimate();
            g2o::Sim3 sim3 = v1->estimate();

            Vector4D coeffs = plane._coeffs;
            Vector4D localCoeffs;
            Matrix3D R = sim3.rotation().matrix();
            localCoeffs.head<3>() = sim3.scale() * (R * coeffs.head<3>());
            localCoeffs(3) = coeffs(3) - sim3.translation().dot(localCoeffs.head<3>());
            if (localCoeffs(3) < 0.0)
                localCoeffs = -localCoeffs;
            Plane3D localPlane = Plane3D(localCoeffs);

            _error = localPlane.ominus_ver(_measurement);
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
    };

    class EdgeVerticalPlaneInverseSim3Project
            : public g2o::BaseBinaryEdge<2, Plane3D, VertexPlane, g2o::VertexSim3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeVerticalPlaneInverseSim3Project() {}

        void computeError() {
            const g2o::VertexSim3Expmap *v1 = static_cast<const g2o::VertexSim3Expmap *>(_vertices[1]);
            const g2o::VertexPlane *v2 = static_cast<const g2o::VertexPlane *>(_vertices[0]);

            const Plane3D &plane = v2->estimate();
            g2o::Sim3 sim3 = v1->estimate().inverse();

            Vector4D coeffs = plane._coeffs;
            Vector4D localCoeffs;
            Matrix3D R = sim3.rotation().matrix();
            localCoeffs.head<3>() = sim3.scale() * (R * coeffs.head<3>());
            localCoeffs(3) = coeffs(3) - sim3.translation().dot(localCoeffs.head<3>());
            if (localCoeffs(3) < 0.0)
                localCoeffs = -localCoeffs;
            Plane3D localPlane = Plane3D(localCoeffs);

            _error = localPlane.ominus_ver(_measurement);
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
    };
}


#endif //ORB_SLAM2_EDGEVERTICALPLANE_H
