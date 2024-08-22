#include <vector>
#include <mutex>
#include <string>
#include <opencv2/opencv.hpp>
#include "Shared.h"

using namespace std;

void SharedData::setData(const vector<Detection>& newData)
{
    lock_guard<mutex> lock(mtx);
    SharedDetectionData = newData;
}


vector<Detection> SharedData::getData()
{
    lock_guard<mutex> lock(mtx);
    return SharedDetectionData;
}