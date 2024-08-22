

class Slam{
    private:

    public:
    
}
// ORB SLAM 3 Initiation
ORB_SLAM3::System SLAM(strVocFile, strSettingsFile, ORB_SLAM3::System::MONOCULAR, false);    
// Track image using ORB-SLAM 
double tframe = cv::getTickCount() / cv::getTickFrequency();
// Camera Pose
Sophus::SE3f cameraPose = SLAM.TrackMonocular(frame, tframe);
// Transf Camera World
Eigen::Matrix4f Tcw = cameraPose.matrix();
// Transf World Camera
Eigen::Matrix4f Twc = Tcw.inverse();

// World 3D points: Map Points current frame
std::vector<ORB_SLAM3::MapPoint*> mapPoints = SLAM.GetTrackedMapPoints();
// All map points adjusted by SLAM
std::vector<ORB_SLAM3::MapPoint*> allMapPoints = SLAM.GetAllMapPoints();
// 2D keyPoints current frame
std::vector<cv::KeyPoint> keyPoints = SLAM.GetTrackedKeyPointsUn();
// Keypoints descriptors
cv::Mat descriptors = SLAM.GetDescriptors();
// SLAM state - initializing / mapping and localization / lost / merging / loop closing ...
int state = SLAM.GetTrackingState();