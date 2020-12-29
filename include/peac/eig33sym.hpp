//
// Copyright 2014 Mitsubishi Electric Research Laboratories All
// Rights Reserved.
//
// Permission to use, copy and modify this software and its
// documentation without fee for educational, research and non-profit
// purposes, is hereby granted, provided that the above copyright
// notice, this paragraph, and the following three paragraphs appear
// in all copies.
//
// To request permission to incorporate this software into commercial
// products contact: Director; Mitsubishi Electric Research
// Laboratories (MERL); 201 Broadway; Cambridge, MA 02139.
//
// IN NO EVENT SHALL MERL BE LIABLE TO ANY PARTY FOR DIRECT,
// INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
// LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
// DOCUMENTATION, EVEN IF MERL HAS BEEN ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGES.
//
// MERL SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN
// "AS IS" BASIS, AND MERL HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE,
// SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
//
#pragma once
//
// Note:
// If you use dsyevh3 library [http://www.mpi-hd.mpg.de/personalhomes/globes/3x3/],
// this function can be accelerated by uncommenting the following line
//#define USE_DSYEVH3
//
#ifdef USE_DSYEVH3
	#include "dsyevh3/dsyevh3.h"
#else
	#include <Eigen/Core>
	#include <Eigen/Dense>
#endif

#include <cmath>
#include <limits>

namespace LA {
	//s[0]<=s[1]<=s[2], V[:][i] correspond to s[i]
	inline static bool eig33sym(double K[3][3], double s[3], double V[3][3])
	{
#ifdef USE_DSYEVH3
		double tmpV[3][3];
		if(dsyevh3(K, tmpV, s)!=0) return false;

		int order[]={0,1,2};
		for(int i=0; i<3; ++i) {
			for(int j=i+1; j<3; ++j) {
				if(s[i]>s[j]) {
					double tmp=s[i];
					s[i]=s[j];
					s[j]=tmp;
					int tmpor=order[i];
					order[i]=order[j];
					order[j]=tmpor;
				}
			}
		}
		V[0][0]=tmpV[0][order[0]]; V[0][1]=tmpV[0][order[1]]; V[0][2]=tmpV[0][order[2]];
		V[1][0]=tmpV[1][order[0]]; V[1][1]=tmpV[1][order[1]]; V[1][2]=tmpV[1][order[2]];
		V[2][0]=tmpV[2][order[0]]; V[2][1]=tmpV[2][order[1]]; V[2][2]=tmpV[2][order[2]];
#else
		//below we did not specify row major since it does not matter, K==K'
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(
			Eigen::Map<Eigen::Matrix3d>(K[0], 3, 3) );
		Eigen::Map<Eigen::Vector3d>(s,3,1)=es.eigenvalues();
		//below we need to specify row major since V!=V'
		Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor>>(V[0],3,3)=es.eigenvectors();
#endif
		return true;
	}
}//end of namespace LA