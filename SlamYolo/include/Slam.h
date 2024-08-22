#include "Shared.h"
#include "Camera.h"
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>
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
#include "inference.h"
#include <algorithm>
#include <boost/filesystem.hpp>
#include <filesystem>


class Slam
{
public:

    Slam();
    ~Slam();

private:

};