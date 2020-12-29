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

#include<string>
#include "opencv2/opencv.hpp"

namespace ahc {
namespace utils {

/**
 *  \brief Generate pseudo-colors 
 *  
 *  \param [in] ncolors number of colors to be generated (will be reset to 10 if ncolors<=0)
 *  \return a vector of cv::Vec3b
 */
inline std::vector<cv::Vec3b> pseudocolor(int ncolors) {
	srand((unsigned int)time(0));
	if(ncolors<=0) ncolors=10;
	std::vector<cv::Vec3b> ret(ncolors);
	for(int i=0; i<ncolors; ++i) {
		cv::Vec3b& color=ret[i];
		color[0]=rand()%255;
		color[1]=rand()%255;
		color[2]=rand()%255;
	}
	return ret;
}

/**
\brief helper class for measuring time elapse
*/
struct Timer {
	int scale;
	double startTick;

	/**
	constructor
	
	@param scale time scale, 1 means second, 1000 means milli-second,
	             1/60.0 means minutes, etc.
	*/
	Timer(int scale=1) {
		this->scale = scale;
		startTick=0;
	}

	/**
	start record time, similar to matlab function "tic";
	
	@return the start tick
	*/
	inline double tic() {
		return startTick = (double)cv::getTickCount();
	}

	/**
	return duration from last tic, in (second * scale), similar to matlab function "toc"
	
	@return duration from last tic,  in (second * scale)
	*/
	inline double toc() {
		return ((double)cv::getTickCount()-startTick)/cv::getTickFrequency() * scale;
	}
	inline double toc(std::string tag) {
		double time=toc();
		std::cout<<tag<<" "<<time<<std::endl;
		return time;
	}

	/**
	equivalent to { toc(); tic(); }
	
	@return duration from last tic,  in (second * scale)
	*/
	inline double toctic() {
		double ret = ((double)cv::getTickCount()-startTick)/cv::getTickFrequency() * scale;
		tic();
		return ret;
	}
	inline double toctic(std::string tag) {
		double time=toctic();
		std::cout<<tag<<" "<<time<<std::endl;
		return time;
	}
};

} //utils
} //ach