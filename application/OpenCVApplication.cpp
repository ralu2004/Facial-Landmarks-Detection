// OpenCVApplication.cpp
// Entry point. Presents a menu to pick which approach to run or evaluate.

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
    cout << "1. Approach 1 - HSV + YCbCr skin detection\n";
    cout << "2. Approach 2 - Region growing\n";
    cout << "3. Approach 3 - Viola Jones face + classical landmarks\n";
    cout << "4. Approach 4 - Viola Jones full pipeline\n";
    cout << "5. Evaluate Approach 1 on MTFL\n";
    cout << "6. Evaluate Approach 2 on MTFL\n";
    cout << "7. Evaluate A2 upper bound (GT seed)\n";
    cout << "8. Evaluate Approach 3 on MTFL\n";
    cout << "9. Evaluate Approach 4 on MTFL\n";
    cout << "0. Exit\n";
    cout << "Choice: ";
    cin >> choice;

    if (choice >= 1 && choice <= 4) {
        switch (choice) {
        case 1: runApproach1(path); break;
        case 2: runApproach2(path); break;
        case 3: runApproach3(path); break;
        case 4: runApproach4(path); break;
        }
    }
    else if (choice >= 5) {
        string annotationFile, imageRoot, csvPath;
        int maxImages;

        cin.ignore();

        cout << "Annotation file path (training.txt): ";
        getline(cin, annotationFile);

        cout << "Image root folder: ";
        getline(cin, imageRoot);

        cout << "Max images to evaluate: ";
        cin >> maxImages;
        cin.ignore();

        cout << "Output CSV path (leave empty to skip): ";
        getline(cin, csvPath);

        EvalResult result;

        if (choice == 5) {
            // A1: HSV + YCbCr skin + Otsu eye threshold
            LandmarkParams a1Params;
            a1Params.useOtsu = true;
            result = runEvaluation(annotationFile, imageRoot, maxImages, detectSkinHSV, a1Params, csvPath);
        }
        else if (choice == 6) {
            // A2: region growing, custom mouth band, no Otsu
            LandmarkParams a2Params;
            a2Params.mouthBandBottom = 0.95f;
            a2Params.mouthBandTop = 0.80f;
            a2Params.useOtsu = false;
            result = runEvaluation(annotationFile, imageRoot, maxImages, detectSkinAutoSeed, a2Params, csvPath);
        }
        else if (choice == 7) {
            // A2 upper bound: GT seed, same params as A2
            LandmarkParams a2Params;
            a2Params.mouthBandBottom = 0.95f;
            a2Params.mouthBandTop = 0.80f;
            a2Params.useOtsu = false;
            result = runEvaluationGTSeed(annotationFile, imageRoot, maxImages, a2Params, csvPath);
        }
        else if (choice == 8) {
            // A3: VJ face + classical landmarks, stricter darkness offset, no Otsu
            LandmarkParams a3Params;
            a3Params.eyeDarknessOffset = 40;
            a3Params.useOtsu = false;
            result = runEvaluationVJ(annotationFile, imageRoot,
                "data/haarcascade_frontalface_default.xml",
                maxImages, a3Params, csvPath);
        }
        else if (choice == 9) {
            // A4: full VJ pipeline 
            result = runEvaluationVJFull(
                annotationFile, imageRoot,
                "data/haarcascade_frontalface_default.xml",
                "data/haarcascade_eye.xml",
                "data/haarcascade_mcs_mouth.xml",
                maxImages, csvPath);
        }

        cout << "\n=== Results ===\n";
        cout << "Total images:    " << result.total << "\n";
        cout << "Eyes detected:   " << result.detected << " (" << 100.0 * result.detected / result.total << "%)\n";
        cout << "Mouth detected:  " << result.mouthDetected << " (" << 100.0 * result.mouthDetected / result.total << "%)\n";
        cout << "Eye NME:         " << result.eyeNME << "\n";
        cout << "Mouth NME:       " << result.mouthNME << "\n";
        cout << "Combined NME:    " << result.meanNME << "\n";
        cout << "Failure rate:    " << result.failureRate << "%\n";

        if (!csvPath.empty())
            cout << "Results saved to: " << csvPath << "\n";
    }

    return 0;
}