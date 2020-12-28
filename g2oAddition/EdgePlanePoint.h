//
// Created by raza on 16.02.20.
//

#ifndef ORB_SLAM2_EDGEPLANEPOINT_H
#define ORB_SLAM2_EDGEPLANEPOINT_H

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "g2oAddition/Plane3D.h"

namespace g2o {
    class EdgePlanePoint : public BaseUnaryEdge<3, Plane3D, g2o::VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgePlanePoint() {}

        void computeError() {
            const g2o::VertexSE3Expmap *v1 = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
            Plane3D obs = _measurement;
            Eigen::Vector3d proj = v1->estimate().map(Xw);

            Eigen::Vector4d plane = obs.toVector();

            _error(0) = plane(0) * proj(0) + plane(1) * proj(1) + plane(2) * proj(2) + plane(3);
            _error(1) = 0;
            _error(2) = 0;
        }

        bool read(std::istream &is) {
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

        bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }

        Eigen::Vector3d Xw;
    };

    class EdgePlanePointTranslationOnly : public BaseUnaryEdge<3, Plane3D, g2o::VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgePlanePointTranslationOnly() {}

        void computeError() {
            const g2o::VertexSE3Expmap *v = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
            Plane3D obs = _measurement;
            Eigen::Vector3d proj = v->estimate().mapTrans(Xc);

            Eigen::Vector4d plane = obs.toVector();

            _error(0) = plane(0) * proj(0) + plane(1) * proj(1) + plane(2) * proj(2) + plane(3);
            _error(1) = 0;
            _error(2) = 0;
        }

        bool read(std::istream &is) {
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

        bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }

        virtual void linearizeOplus() {
            g2o::VertexSE3Expmap *v = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
            Eigen::Vector3d xyz_trans = v->estimate().mapTrans(Xc);

            Vector3D normal = _measurement.normal();

            _jacobianOplusXi(0, 0) = 0;
            _jacobianOplusXi(0, 1) = 0;
            _jacobianOplusXi(0, 2) = 0;
            _jacobianOplusXi(0, 3) = normal(0);
            _jacobianOplusXi(0, 4) = normal(1);
            _jacobianOplusXi(0, 5) = normal(2);

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

        Eigen::Vector3d Xc;
    };
}


#endif //ORB_SLAM2_EDGEPLANEPOINT_H
