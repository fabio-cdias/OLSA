#ifndef SHAREDDATA_H
#define SHAREDDATA_H

#include <vector>
#include <mutex>
#include <string>
#include <opencv2/opencv.hpp>
#include "inference.h"

class SharedData
{
public:
    void setData(const std::vector<Detection>& newData);
    std::vector<Detection> getData(); 

private:
    std::vector<Detection> SharedDetectionData;
    std::mutex mtx;
};

#endif // SHAREDDATA_H