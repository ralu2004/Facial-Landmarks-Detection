#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include "FacialLandmarks.h"
#include "Evaluation.h"

using namespace std;

wchar_t* projectPath;

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

    char fname[MAX_PATH];
    openFileDlg(fname);
    string path(fname);

    int choice = 0;
    cout << "Facial Landmarks Detection\n";
    cout << "1. Approach 1 - HSV skin detection\n";
    cout << "2. Approach 2 - Region growing\n";
    cout << "3. Approach 3 - Viola Jones Algorithm\n";
    cout << "4. Evaluate Approach 1 on MTFL\n";
    cout << "5. Evaluate Approach 2 on MTFL\n";
    cout << "6. Evaluate A2 upper bound (GT seed)\n";
    cout << "7. Evaluate Approach 3 on MTFL\n";
    cout << "0. Exit\n";
    cout << "Choice: ";
    cin >> choice;

    if (choice == 1 || choice == 2|| choice == 3) {
        switch (choice) {
        case 1: runApproach1(path); break;
        case 2: runApproach2(path); break;
        case 3: runApproach3(path); break;
        }
    }
    else if (choice >= 4) {
        string annotationFile, imageRoot;
        int    maxImages;

        cin.ignore();  

        cout << "Annotation file path (training.txt): ";
        getline(cin, annotationFile);

        cout << "Image root folder: ";
        getline(cin, imageRoot);

        cout << "Max images to evaluate: ";
        cin >> maxImages;
        cin.ignore();

        LandmarkParams p;
        EvalResult result;

        if (choice == 4)
            result = runEvaluation(annotationFile, imageRoot, maxImages, detectSkinHSV, p);
        else if (choice == 5)
            result = runEvaluation(annotationFile, imageRoot, maxImages, detectSkinAutoSeed, p);
        else if (choice == 6)
            result = runEvaluationGTSeed(annotationFile, imageRoot, maxImages, p);
        else if (choice == 7) {
            string cascadePath = "data/haarcascade_frontalface_default.xml";
            LandmarkParams vjParams;
            vjParams.eyeDarknessOffset = 40;  // same as params3::lp
            result = runEvaluationVJ(annotationFile, imageRoot, cascadePath, maxImages, vjParams);
        }

        cout << "\n=== Results ===\n";
        cout << "Total images:    " << result.total << "\n";
        cout << "Detected:        " << result.detected << "\n";
        cout << "Mean NME:        " << result.meanNME << "\n";
        cout << "Failure rate:    " << result.failureRate << "%\n";
    }
    
    return 0;
}