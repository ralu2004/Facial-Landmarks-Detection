// main.cpp
// Entry point. Presents a menu to pick which approach to run.

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include "FacialLandmarks.h"

using namespace std;

wchar_t* projectPath;

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

    string path = "Images/Angelina_Jolie_0006.jpg";

    int choice = 0;
    cout << "Facial Landmarks Detection\n";
    cout << "1. HSV skin + darkness/redness features\n";
    cout << "2. Interactive region growing\n";
    cout << "0. Exit\n";
    cout << "Choice: ";
    cin >> choice;

    switch (choice) {
        case 1: runApproach1(path); break;
        case 2: runApproach2(path); break;
        case 0: break;
        default: cout << "Unknown choice.\n"; break;
    }

    return 0;
}
