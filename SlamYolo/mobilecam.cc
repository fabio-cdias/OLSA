#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "System.h"
#include "Atlas.h"
#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <pangolin/gl/glfont.h>
#include <pangolin/gl/gltext.h>
#include <mutex>
#include "inference.h"
#include <algorithm>
#include <omp.h>
#include <AL/al.h>
#include <AL/alc.h>
#include <AL/alext.h>
#include <sndfile.h>
#include <boost/filesystem.hpp>
#include <filesystem>


// Function to fetch image from IP camera
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

struct Object {
    int id;
    int lastSeen;
    int detection;

    std::string name;
    float confidence;
    cv::Mat descriptors;
    std::vector<Eigen::Vector3f> mapPoints;
    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
};

pangolin::GlFont* glFont = nullptr;

float GetDistancePt(const Eigen::Vector3f& pointOne, const Eigen::Vector3f& pointTwo){
    return (pointOne - pointTwo).norm();
}

void DrawText(const std::string text, const float x, const float y, const float z){
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    pangolin::GlText glText = glFont->Text(text);
    glText.Draw(x, y, z);
    glDisable(GL_BLEND);
}

void DrawCamera(const Eigen::Matrix4f& Twc, int nObjects) {
    const float w = 0.05;
    const float h = w * 0.75;
    const float z = w * 0.6;
    glPushMatrix();
    // Apply the camera pose
    glMultMatrixf(Twc.data());
    glLineWidth(4.0f);
    glColor3f(0.0f, 1.0f, 1.0f);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);
    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);
    glEnd();
    glPopMatrix();
    
    Eigen::Vector3f camPosition = Twc.block<3, 1>(0, 3);
    std::string text = std::to_string(nObjects)+" Objects";
    DrawText(text,camPosition.x(), camPosition.y()-0.08, camPosition.z());
    glEnd();
}

void DrawAllPoints(const std::vector<ORB_SLAM3::MapPoint*>& allMapPoints) {
    glPointSize(1);
    glBegin(GL_POINTS);
    glColor3f(1.0, 1.0, 1.0); 
    for (const auto& mapPoint : allMapPoints) {
        if (mapPoint) {
            Eigen::Vector3f pos = mapPoint->GetWorldPos();
            glVertex3f(pos.x(), pos.y(), pos.z());
        }
    }
    glEnd();
}

void DrawObjectPoints(const std::vector<Object>& objects,const Eigen::Matrix4f& Twc){
    for (const auto& obj : objects) {
        if (obj.descriptors.empty()){
            continue;
        } else if (obj.name == "laptop"){
            glColor3f(0.647058824, 1.0, 0.101960784); //GreenYellow laptop
        } else if (obj.name == "cup"){
            glColor3f(0.6, 0.752941176, 890196078); // Light Blue cup
        } else if (obj.name == "chair"){
            glColor3f(0.835294118, 0.647058824, 0.403921569); //Wood Brown chair
        } else if (obj.name == "mouse"){
            glColor3f(1.0,0.388235294,0.278431373);   // Tomato mouse
        } else if (obj.name == "bottle"){
            glColor3f(0.121568627, 0.3176470593,1.0); //Medium blue bottle
        } else if (obj.name == "apple"){
            glColor3f(1.0, 0.0, 0.0);  // Red apple 
        } else if (obj.name == "scissors"){
            glColor3f(1.0, 0.854901961, 0.078431373); // Gold Yellow scissors
        } else{
            glColor3f(0.564705882, 0.207843137, 0.890196078); //Violet bed
        }
   

        // Render mean point
        glPointSize(30);
        glBegin(GL_POINTS);
        glVertex3f(obj.mean.x(), obj.mean.y(), obj.mean.z());
        glEnd();

        // Render lines for object
        glLineWidth(2);
        glBegin(GL_LINES);
        float nameSize = obj.name.size();
        
        glVertex3f(obj.mean.x(), obj.mean.y(), obj.mean.z());
        glVertex3f(obj.mean.x(), obj.mean.y() - 0.1f, obj.mean.z());
        glEnd();

        // Render object map points
        glPointSize(5);
        glBegin(GL_POINTS);
        for (auto& pt : obj.mapPoints) {
            glVertex3f(pt.x(), pt.y(), pt.z());
        }
        glEnd();

        Eigen::Vector3f camPosition = Twc.block<3, 1>(0, 3);

        // float distance = GetDistancePt(obj.mean,camPosition);
        // std::string textName = obj.name+"-"+std::to_string(obj.id);
        // DrawText(textName, obj.mean.x()-0.01, obj.mean.y() - 0.08f, obj.mean.z());

        float distance = GetDistancePt(obj.mean,camPosition);
        std::ostringstream ss;
        ss << obj.name << " - " << std::to_string(obj.id) << "| Dist: " << std::fixed << std::setprecision(1) << distance;
        std::string text = ss.str();
        DrawText(text, obj.mean.x(), obj.mean.y() - 0.1f, obj.mean.z());

        glLineWidth(1);
        glColor3f(0.50,0.50,0.50);
        glBegin(GL_LINES);
        glVertex3f(camPosition.x(), camPosition.y(), camPosition.z());
        glVertex3f(obj.mean.x(), obj.mean.y(), obj.mean.z());
        glEnd();

        

    }
}
void checkAlError(const char* msg) {
    ALenum error = alGetError();
    if (error != AL_NO_ERROR) {
        std::cerr << "OpenAL error (" << msg << "): " << alGetString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}
ALuint loadAudioFile(const char* filename) {
    // Open the audio file
    ALenum err, format;
    SF_INFO sfInfo;
    SNDFILE* sndFile = sf_open(filename, SFM_READ, &sfInfo);
    if (!sndFile) {
        std::cerr << "Failed to open audio file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read the audio data
    std::vector<short> samples(sfInfo.frames * sfInfo.channels);
    sf_read_short(sndFile, samples.data(), samples.size());
    sf_close(sndFile);

    // Determine the OpenAL format

    format = AL_NONE;
    if(sfInfo.channels == 1)
        format = AL_FORMAT_MONO16;
    else if(sfInfo.channels == 2)
        format = AL_FORMAT_STEREO16;
    else if(sfInfo.channels == 3)
    {
        if(sf_command(sndFile, SFC_WAVEX_GET_AMBISONIC, NULL, 0) == SF_AMBISONIC_B_FORMAT)
            format = AL_FORMAT_BFORMAT2D_16;
    }
    else if(sfInfo.channels == 4)
    {
        if(sf_command(sndFile, SFC_WAVEX_GET_AMBISONIC, NULL, 0) == SF_AMBISONIC_B_FORMAT)
            format = AL_FORMAT_BFORMAT3D_16;
    }
    if(!format)
    {
        std::cerr << "Unsupported channel count: " << sfInfo.channels << std::endl;
        exit(EXIT_FAILURE);
    }

    // Create an OpenAL buffer and copy the audio data to it
    ALuint buffer;
    alGenBuffers(1, &buffer);
    checkAlError("alGenBuffers");

    alBufferData(buffer, format, samples.data(), samples.size() * sizeof(short), sfInfo.samplerate);
    checkAlError("alBufferData");

    return buffer;
}

std::vector<ALuint> storeAudioBuffer(const std::string& path) {
    std::vector<ALuint> buffers;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().string();
            std::cout << filename << std::endl;
            ALuint buffer = loadAudioFile(filename.c_str());
            buffers.push_back(buffer);
        }
    }
    return buffers;
}

// void playAt(const ALuint& buffer,const Sophus::SE3f&  cameraPose, Eigen::Vector3f tObj) {
//     // Initialize OpenAL context
//     ALCdevice* device = alcOpenDevice(nullptr);
//     ALCcontext* context = alcCreateContext(device, nullptr);
//     alcMakeContextCurrent(context);
    
//     ALuint source;
//     alGenSources(1, &source);

//     alSourcei(source, AL_BUFFER, buffer);

//     ALfloat sourcePos[] = { tObj.x(), tObj.y(), tObj.z() }; //Source pos
//     alSourcefv(source, AL_POSITION, sourcePos);


//     Eigen::Vector3f vForward = cameraPose.so3().matrix().col(2);
//     Eigen::Vector3f vUp      = cameraPose.so3().matrix().col(1);
//     Eigen::Vector3f tCam     = cameraPose.translation();
//     // vForward.x() = cos(rCam.y()) * cos(rCam.z());
//     // vForward.y() = sin(rCam.x()) * sin(rCam.z());
//     // vForward.z() = sin(rCam.z()) * cos(rCam.x());
//     ALfloat listenerPos[] = {tCam.x(),tCam.y(), tCam.z()}; // Listener at origin
//     ALfloat listenerOri[] = {vForward.x(), vForward.y(), vForward.z(), vUp.x(), vUp.y(), vUp.z() }; // Forward and up vectors

//     alSourcePlay(source);

//     // Wait for audio to finish 
//     ALint sourceState;
//     do {
//         alGetSourcei(source, AL_SOURCE_STATE, &sourceState);
//     } while (sourceState == AL_PLAYING);

//     // Clean up resources
//     alDeleteSources(1, &source);
//     alDeleteBuffers(1, &buffer);
//     alcDestroyContext(context);
//     alcCloseDevice(device);
// }

std::vector<Detection> detections;


omp_lock_t sharedYoloLock;

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cerr << "Usage: ./mono_live_mcam path_to_vocabulary path_to_settings camera_url username password" << std::endl;
        return 1;
    }
    omp_init_lock(&sharedYoloLock);

    // Setup to use smartphone camera ip-ish (usb-tethering)
    std::string strVocFile = argv[1];
    std::string strSettingsFile = argv[2];
    std::string camera_url = argv[3];
    std::string username = argv[4];
    std::string password = argv[5];

    glEnable(GL_DEPTH_TEST);
    

    // Setup Pangolin
    pangolin::CreateWindowAndBind("Map Viewer", 1024, 768);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0.0, -0.5, -0.5, 0, 0, 0, 0.0, -0.01, 0.0) //eye pos
    );
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::CreatePanel("menu").SetBounds(0.0,0.03,0.0,pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",false,true);
    pangolin::Var<bool> menuCamView("menu.Camera View",false,false);

    bool bFollow = true;
    bool bCameraView = true;

    pangolin::OpenGlMatrix Ow; // Oriented with g in the z axis
    Ow.SetIdentity();
    

    // YOLO Inference
    std::string projectBasePath = "/home/fabio/Projects/ComputerVision/ORB_SLAM3_YOLO/SlamYolo";
    bool runOnGPU = false;
    Inference inf(projectBasePath + "/yolov5nu.onnx", cv::Size(640, 480), "classes.txt", runOnGPU);

    // ORB SLAM 3 Initiation
    ORB_SLAM3::System SLAM(strVocFile, strSettingsFile, ORB_SLAM3::System::MONOCULAR, false);


    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    int countID = 0;
    int frameID = 0;


    glFont = new pangolin::GlFont("/home/fabio/Projects/ComputerVision/ORB_SLAM3_YOLO/SlamYolo/Arial.ttf", 30,5000,5000);


    ALCdevice* device = alcOpenDevice(nullptr);
    ALCcontext* context = alcCreateContext(device, nullptr);
    alcMakeContextCurrent(context);
    ALuint source;
    ALCint hrtf;
    alcGetIntegerv(device, ALC_HRTF_SOFT, 1, &hrtf);
    if (hrtf == ALC_TRUE) {
        std::cout << "HRTF is enabled." << std::endl;
    } else {
        std::cout << "HRTF is not enabled." << std::endl;
}
    std::vector<ALuint> buffers = storeAudioBuffer("/home/fabio/Projects/ComputerVision/ORB_SLAM3_YOLO/SlamYolo/Audio");
    
    
    std::vector<Object> objects;
    int nObjects;

    while (!pangolin::ShouldQuit()) {
        frameID++;
        nObjects = objects.size();

        // Live image from IP camera
        cv::Mat frame = fetchImageFromCamera(camera_url, username, password);
        if (frame.empty()) {
            std::cerr << "Error: Captured empty frame" << std::endl;
            break;
        }

    

        d_cam.Activate(s_cam);

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // Run YOLO detection
                std::vector<Detection> localDetections = inf.runInference(frame);
                
                omp_set_lock(&sharedYoloLock);

                detections = std::move(localDetections);

                omp_unset_lock(&sharedYoloLock);

            }
        }
        

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

       
        
        // std::vector<std::string> classFilter = {"notebook","copo","cadeira","cama","garrafa","mouse","tesoura","maçã"};
        std::vector<std::string> classFilter = {"laptop","cup","chair","bed","bottle","mouse","scissors","apple"};


        // Clear screen
        glClearColor(0.1,0.1,0.1,1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

         if(menuFollowCamera && bFollow)
        {
            if(bCameraView)
                s_cam.Follow(Twc);
            else
                s_cam.Follow(Ow);
        }
        else if(menuFollowCamera && !bFollow)
        {
            if(bCameraView)
            {
                s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,500,500,512,389,0.1,1000));
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0.0,-0.05,-0.5, 0,0,0,0.0,-1.0, 0.0));
                s_cam.Follow(Twc);
            }
            else
            {
                s_cam.SetProjectionMatrix(pangolin::ProjectionMatrix(1024,768,3000,3000,512,389,0.1,1000));
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,0.01,10, 0,0,0,0.0,0.0, 1.0));
                s_cam.Follow(Ow);
            }
            bFollow = true;
        }
        else if(!menuFollowCamera && bFollow)
        {
            bFollow = false;
        }
 

        // Clear objects if lost
        if (state == 0){
            objects.clear();
        }


        for (const auto& detection : detections){
            // Filters by class name and level of confidence
            auto contains = std::find(classFilter.begin(),classFilter.end(),detection.className);

            if (detection.confidence > 0.5  &&
                contains != classFilter.end()   &&
                !keyPoints.empty()              &&
                !mapPoints.empty())
            {
                int count = 0;
                cv::Rect box = detection.box;
                float boxRatio = box.width/box.height;
                cv::Mat currentDescriptors;
                std::vector<Eigen::Vector3f> currentMapPoints;
                Eigen::Vector3f currentMean = Eigen::Vector3f::Zero();
                for (int i = 0; i < keyPoints.size(); i++){

                    if (box.contains(keyPoints[i].pt) && mapPoints[i]){
                        Eigen::Vector3f point = mapPoints[i]->GetWorldPos();
                        currentMean += point;
                        currentDescriptors.push_back(descriptors.row(i));
                        currentMapPoints.push_back(point);
                        count++;
                    }
                }
                
                if (count > 0) { 
                    currentMean /= count;
                } else {
                    continue;
                }
                
                bool objectMatched = false;
                bool meanMatched = true;

                for (auto& obj : objects) {
                    
                    float meanDistance = GetDistancePt(obj.mean,currentMean);

                    // Checks the mean distance between objects and presumed new object
                    if (meanDistance < 0.1){
                        meanMatched = false;
                    }
                    
                    // Checks features descriptor matching
                    if (!obj.descriptors.empty() &&
                        !currentDescriptors.empty() &&
                        state == 2 &&
                        obj.name == detection.className
                         )
                    {

                        std::vector<cv::DMatch> matches;
                        std::vector<cv::DMatch> good_matches;
                        matcher.match(obj.descriptors, currentDescriptors, matches);

                        for (int j = 0; j < matches.size(); j++) {
                            if (matches[j].distance < 50){

                                good_matches.push_back(matches[j]);
                            }
                        }


                        // OBJECT UPDATE
                        if (good_matches.size() > 10) {
                            // if (meanDistance > 5){
                            // }
                            
                            obj.descriptors = currentDescriptors;
                            obj.detection ++;
                            obj.lastSeen = frameID;
                            obj.mapPoints = currentMapPoints;
                            obj.mean = currentMean; 
                            objectMatched = true;
                            break; 
                        }
                    }  
                }
               
                // OBJECT CREATION
                // int creationThreshold = std::max(20, static_cast<int>(box.area() / 1000.0));
                if (!objectMatched && meanMatched && count > 30) {
                    Object obj;
                    obj.id = countID++;
                    obj.lastSeen = frameID;
                    obj.detection++;
                    obj.name = detection.className;
                    obj.confidence = detection.confidence;
                    obj.descriptors = currentDescriptors;
                    obj.mapPoints = currentMapPoints;
                    obj.mean = currentMean;
                    objects.push_back(obj);
                    
                    // if (obj.name =="cup"){
                    //     playAt(buffers[0],cameraPose,obj.mean);
                    // }
                    // if (obj.name == "cup"){
                        
                    //     alSourcei(source, AL_BUFFER, buffers[0]);
                    //     ALfloat sourcePos[] = { obj.mean.x(), obj.mean.y(), obj.mean.z() }; //Source pos
                    //     alSourcefv(source, AL_POSITION, sourcePos);

                    //     Eigen::Vector3f vForward = cameraPose.so3().matrix().col(2);
                    //     Eigen::Vector3f vUp      = cameraPose.so3().matrix().col(1);
                    //     Eigen::Vector3f tCam     = cameraPose.translation();
                    //     ALfloat listenerPos[] = {tCam.x(),tCam.y(), tCam.z()}; // Listener at origin
                    //     ALfloat listenerOri[] = {vForward.x(), vForward.y(), vForward.z(), vUp.x(), vUp.y(), vUp.z() }; // Forward and up vectors
                    //     alSourcePlay(source);

                    //     // Wait for audio to finish 
                    //     ALint sourceState;
                    //     do {
                    //         alGetSourcei(source, AL_SOURCE_STATE, &sourceState);
                    //     } while (sourceState == AL_PLAYING);

                    //     // Clean up resources
                    //     alDeleteSources(1, &source);
                    //     if (!buffers.empty()){
                    //         alDeleteBuffers(buffers.size(),buffers.data());
                    //     }

                    // }

                }
                

                // Bounding box drawing
                cv::Scalar color = detection.color;
                cv::rectangle(frame, box, color, 2);
                std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
                cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
                cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
                cv::rectangle(frame, textBox, color, cv::FILLED);
                cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 20), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 255, 255), 2);
            }

            
        }
        // Removes stale objects based on last detection frame
        // *Each object detection will increase the frame counter*
        // std::cout << "-------------------------------------\n";
        // std::cout << "OBJECTS N: " << objects.size() << std::endl;
        for (auto it = objects.begin(); it != objects.end(); ) {
            int frameLimit = 20 + it->detection * 20;
            // std::cout << it->id << "-" << it->name << " | Buffer: "<< frameID - it->lastSeen << " : " << frameLimit << std::endl;
            if ( frameID - it->lastSeen > frameLimit) {
                it = objects.erase(it);
            } else {
                ++it;
            }
        }

        //Patches erroneous object creation
        //Removes objects with distance similarity        
        for (auto it = objects.begin(); it != objects.end(); ) {
            bool duplicateFound = false;
            for (auto jt = it + 1; jt != objects.end(); ++jt) {
                float meanDistance = GetDistancePt(it->mean, jt->mean);
                if (meanDistance < 0.1) {
                    // Compare duplicate IDs
                    if (it->id  >  jt->id) {
                        std::cout << "Removing" << it->name << " with ID " << it->id << std::endl;
                        it = objects.erase(it);
                    } else {
                        std::cout << "Removing " << jt->name << " with ID " << jt->id << std::endl;
                        jt = objects.erase(jt);
                    }
                    duplicateFound = true;
                    break;
                }
            }
            if (!duplicateFound) {
                ++it;
            }
        }

        if (cv::waitKey(1)== 115){

            for(auto& obj : objects){
                if (obj.name == "cup"){
                    alGenSources(1, &source);
                    // alSourcei(source, AL_SOURCE_RELATIVE, AL_TRUE); // Source is relative to listener

                    ALfloat maxDistance = 5.0f; 
                    alSourcef(source, AL_MAX_DISTANCE, maxDistance);
                    ALfloat referenceDistance = 1.0f; 
                    alSourcef(source, AL_REFERENCE_DISTANCE, referenceDistance);
                    ALfloat rolloffFactor = 1.0f; 
                    alSourcef(source, AL_ROLLOFF_FACTOR, rolloffFactor);
                            
                    alSourcei(source, AL_BUFFER, buffers[0]);
                    ALfloat sourcePos[] = { obj.mean.x(), obj.mean.y(), obj.mean.z() }; //Source pos
                    alSourcefv(source, AL_POSITION, sourcePos);

                    Eigen::Vector3f vForward = cameraPose.so3().matrix().col(2);
                    Eigen::Vector3f vUp      = cameraPose.so3().matrix().col(1);
                    Eigen::Vector3f tCam     = cameraPose.translation();

                    // vForward.normalize();
                    // vUp.normalize();
                    // tCam.normalize();
                    
                    ALfloat listenerPos[] = {tCam.x(),tCam.y(), tCam.z()}; // Listener at origin
                    ALfloat listenerOri[] = {vForward.x(), vForward.y(), vForward.z(), vUp.x(), vUp.y(), vUp.z() }; // Forward and up vectors

                    alListenerfv(AL_POSITION, listenerPos);
                    alListenerfv(AL_ORIENTATION, listenerOri);

                    alSourcePlay(source);

                    // Wait for audio to finish 
                    ALint sourceState;
                    do {
                        alGetSourcei(source, AL_SOURCE_STATE, &sourceState);
                    } while (sourceState == AL_PLAYING);

                    alDeleteSources(1, &source);

                }
            }

        }


        DrawAllPoints(allMapPoints);
        DrawObjectPoints(objects,Twc);
        DrawCamera(Twc,nObjects);


        cv::imshow("Frame", frame);
        if (cv::waitKey(1) == 27) break;  // Exit 'esc'

        pangolin::FinishFrame();
    }

    SLAM.Shutdown();
    omp_destroy_lock(&sharedYoloLock);

    alcDestroyContext(context);
    alcCloseDevice(device);
    return 0;
}
