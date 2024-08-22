#include "Camera.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

using namespace std;

IpCamera::IpCamera(const std::string& url, const std::string& username, const std::string& password)
: url(url), username(username), password(password) {}


cv::Mat IpCamera::fetchImage()
{
    cv::Mat img;
    string full_url = "http://" + username + ":" + password + "@" + url;

    // string url = "http://" + username + ":" + password + "@" + url;
    // cap.open(full_url);
    // if (!cap.isOpened()) {
    //     cerr << "Failed to open camera stream: " << url << endl;
    // }
    cv::VideoCapture cap(full_url);
    cap >> img;
    return img;
    // if (!cap.isOpened()) {
    //     cerr << "Camera stream is not opened." << endl;
    //     return img;
    // }

    // try {
    //     cap >> img;
    //     if (img.empty()) {
    //         cerr << "Failed to capture camera image." << endl;
    //     }
    // } catch (const cv::Exception& e) {
    //     cerr << "Exception occurred while fetching camera image: " << e.what() << endl;
    // }
    // return img;
}


USBCamera::USBCamera(const int& deviceID, const int& apiID) : deviceID(0), apiID(cv::CAP_ANY){}
cv::Mat USBCamera::fetchImage()
{
    cv::Mat img;
    cv::VideoCapture cap(deviceID,apiID);
    cap >> img;
    return img;
}
