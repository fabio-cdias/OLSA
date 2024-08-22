#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

class ICamera
{
public:
    virtual ~ICamera() = default;
    virtual cv::Mat fetchImage() = 0;      
};


class IpCamera : public ICamera
{
public:
    IpCamera(const std::string& url,
             const std::string& username, 
             const std::string& password);

    cv::Mat fetchImage() override;

private:
    std::string url, username, password;
};


class USBCamera : public ICamera
{
public:
    USBCamera(const int& deviceID, const int& apiID);
    cv::Mat fetchImage() override;

private:
    int deviceID, apiID;
};


#endif // CAMERA_H