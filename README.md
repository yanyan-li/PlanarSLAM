# PlanarSLAM

This repo will be released at the beginning of 2021 for celebrating the new year.  We hope the world will get better in the new year. 

----

![teaser](/home/yanyan/Documents/ThinkFree/PlanarSLAM/Examples/teaser.png)

# 1. Prerequisites

We have tested the library in **ubuntu 16.04 and ubuntu 18.04**, but it should be easy to compile in other platforms. A powerful computer (e.g. i7) will ensure real-time performance and provide more stable and accurate results.

## C++11 or C++0x Compiler
We use the new thread and chrono functionalities of C++11.

## Pangolin
We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **Tested with OpenCV 3.4.1**

## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

## DBoW2 and g2o (Included in Thirdparty folder)
We use modified versions of the [DBoW2](https://github.com/dorian3d/DBoW2) library to perform place recognition and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

## PCL

We use [PCL](http://www.pointclouds.org/) to reconstruct and visualize mesh. Download and install instructions can be found at: https://github.com/ros-perception/perception_pcl. **Tested with PCL 1.7.0**.

1. https://github.com/PointCloudLibrary/pcl/releases

2. compile and install

   ```
   cd pcl 
   mkdir release 
   cd release
   
   cmake  -DCMAKE_INSTALL_PREFIX=/usr/local \ -DBUILD_GPU=ON -DBUILD_apps=ON -DBUILD_examples=ON \ -DCMAKE_INSTALL_PREFIX=/usr/local -DPCL_DIR=/usr/local/share/pcl  .. 
   
   make -j12
   sudo make install
   ```



# 2. Test the system

*compile the system*

```
./build.sh
```

*command for testing TUM-RGBD sequences* 

```
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml .PATH_TO_SEQUENCE_FOLDER .PATH_TO_SEQUENCE_FOLDER/ASSOCIATIONS_FILE

```

