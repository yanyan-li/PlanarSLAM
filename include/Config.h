//
// Created by fishmarch on 19-5-24.
//

#ifndef ORB_SLAM2_CONFIG_H
#define ORB_SLAM2_CONFIG_H
#include <opencv2/core/core.hpp>
#include <memory>
#include <iostream>

using namespace std;
namespace Planar_SLAM {
    class Config{
    public:
        static void SetParameterFile( const string& filename );

        template <typename T>
        static T Get(const string& key){
            return T(Config::mConfig->mFile[key]);
        }
        ~Config();
    private:
        Config(){}
        static std::shared_ptr<Config> mConfig;
        cv::FileStorage mFile;
    };
}
#endif //ORB_SLAM2_CONFIG_H
