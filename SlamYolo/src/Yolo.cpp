#include "Yolo.h"

Yolo::Yolo(SharedData& sharedData, const std::string& url, const std::string& username, const std::string& password, const bool& show)
    : sharedData(sharedData),
      running(true),
      camera(nullptr),
      show(show)
{
    camera = new IpCamera(url, username, password);
    yoloThread = std::thread(&Yolo::DetectObjects, this);
}


Yolo::Yolo(SharedData& sharedData, const int& deviceID, const int& apiID, const bool& show)
    : sharedData(sharedData),
      running(true),
      camera(nullptr),
      show(show)
{
    camera = new USBCamera(deviceID, apiID);
    yoloThread = std::thread(&Yolo::DetectObjects, this);
}


Yolo::~Yolo() {
    running = false;
    if (yoloThread.joinable()) {
        yoloThread.join();
    }
    delete camera;
}


void Yolo::YoloStats(std::vector<Detection>& detections, cv::Mat& frame)
{
    for(const auto& detection : detections)
    {
        cv::Scalar color = detection.color;
        cv::Rect box = detection.box;
        cv::rectangle(frame, box, color, 2);
        std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
        cv::rectangle(frame, textBox, color, cv::FILLED);
        cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 20), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 255, 255), 2);
    }

    cv::imshow("Frame",frame);
}


void Yolo::DetectObjects()
{
    std::string projectBasePath = "/home/fabio/Projects/ComputerVision/ORB_SLAM3_YOLO/SlamYolo";
    bool runOnGPU = false;
    Inference inf(projectBasePath + "/yolov5nu.onnx", cv::Size(640, 480), "classes.txt", runOnGPU);
    
    while (running)
    {

        cv::Mat frame = camera->fetchImage();
        std::vector<Detection> detections = inf.runInference(frame);
        sharedData.setData(detections);
        
        if (show)
        {
            YoloStats(detections, frame);
        }
        
        if (cv::waitKey(1) == 27) break;  // Exit 'esc'

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

