#include <iostream>
#include <vector>
#include <getopt.h>

#include <opencv2/opencv.hpp>

#include "inference.h"

using namespace std;
using namespace cv;

cv::Mat fetchImageFromCamera(const string& url, const string& username, const string& password) {
    cv::Mat img;
    try {
        // Construct the URL with authentication if provided
        string full_url = "http://" + username + ":" + password + "@" + url;
        
        // Fetch the image from the IP camera
        cv::VideoCapture cap(full_url);
        if (!cap.isOpened()) {
            cerr << "Failed to open camera stream: " << full_url << endl;
            return img;
        }
        
        cap >> img;
        if (img.empty()) {
            cerr << "Failed to capture camera image from: " << full_url << endl;
        }
    } catch (const cv::Exception& e) {
        cerr << "Exception occurred while fetching camera image: " << e.what() << endl;
    }
    return img;
}

int main(int argc, char **argv)
{
    std::string projectBasePath = "/home/fabio/Projects/ComputerVision/YoloCpp/ultralytics"; // Set your ultralytics base path

    bool runOnGPU = false;

    //
    // Pass in either:
    //
    // "yolov8s.onnx" or "yolov5s.onnx"
    //
    // To run Inference with yolov8/yolov5 (ONNX)
    //

    // Note that in this example the classes are hard-coded and 'classes.txt' is a place holder.
    std::cout << "HERE" << std::endl;
    Inference inf(projectBasePath + "/yolov8n.onnx", cv::Size(640, 480), "classes.txt", runOnGPU);

    // std::vector<std::string> imageNames;
    // imageNames.push_back(projectBasePath + "/ultralytics/assets/bus.jpg");
    // imageNames.push_back(projectBasePath + "/ultralytics/assets/zidane.jpg");
    // for (int i = 0; i < imageNames.size(); ++i)
    // {
    while(true){
        cv::Mat frame = fetchImageFromCamera("192.168.226.180:8080/shot.jpg","chibs","penguintux");
        // cv::Mat frame = cv::imread(imageNames[i]);

        // Inference starts here...
        std::vector<Detection> output = inf.runInference(frame);
        int detections = output.size();
        std::cout << "Number of detections:" << detections << std::endl;

        for (int i = 0; i < detections; ++i)
        {
            Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

            // Detection box
            cv::rectangle(frame, box, color, 2);

            // Detection box text
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            cv::rectangle(frame, textBox, color, cv::FILLED);
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
            std::cout << "Name:"<< detection.className << "\nConfidence: " << detection.confidence << "\n Box:"<< detection.box << std::endl;
        }
        // Inference ends here...

        // This is only for preview purposes
        float scale = 0.8;
        cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
        cv::imshow("Inference", frame);

        if (cv::waitKey(1) != -1)
        {
            break;
        }
    }
}
