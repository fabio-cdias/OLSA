#ifndef YOLO_H
#define YOLO_H


#include "Shared.h"
#include "Camera.h"
#include "inference.h"
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>

class Yolo{

public:
    Yolo(SharedData& sharedData,
        const std::string& url,
        const std::string& username,
        const std::string& password,
        const bool& show);

    Yolo(SharedData& sharedData,
        const int& deviceID,
        const int& apiID,
        const bool& show);
        
    ~Yolo();

private:
    SharedData& sharedData;
    ICamera* camera;
    std::thread yoloThread;
    bool running;
    bool show;

    void DetectObjects();
    void YoloStats(std::vector<Detection>& detections, cv::Mat& frame);
            
};

#endif // YOLO_H








