#pragma once
// Evaluation.h
// Evaluation framework for facial landmark detection.
// Include this in Evaluator.cpp and in main.cpp.

#include "FacialLandmarks.h"
#include <functional>

// per-image ground truth from MTFL annotation file
struct GTLandmarks {
    string imagePath;
    Point  leftEye;
    Point  rightEye;
    Point  nose;
    Point  mouthLeft;
    Point  mouthRight;

    // image metadata from annotation file
    int pose;    // 1=frontal, 2=left profile, 3=right profile, 4=upward, 5=downward
    int smile;   // 1=smiling, 2=not smiling
    int glasses; // 1=with glasses, 2=without glasses
    int gender;  // 1=male, 2=female

    // mouth center — average of left and right mouth corners
    // this is what we compare against the detected single mouth point
    Point mouthCenter() const {
        return Point(
            (mouthLeft.x + mouthRight.x) / 2,
            (mouthLeft.y + mouthRight.y) / 2
        );
    }

    // inter-ocular distance — used to normalize NME
    double interocularDistance() const {
        double dx = rightEye.x - leftEye.x;
        double dy = rightEye.y - leftEye.y;
        return sqrt(dx * dx + dy * dy);
    }
};

// aggregated results for one approach on N images
struct EvalResult {
    double eyeNME = 0;           // mean NME for eyes only (left + right)
    double mouthNME = 0;         // mean NME for mouth only (when detected)
    double meanNME = 0;          // combined NME: eyes + mouth when available

    double failureRate = 0;      // % of detected images where combined NME > 0.1

    int    detected = 0;         // images with successful eye detection
    int    mouthDetected = 0;    // images with successful mouth detection
    int    total = 0;            // total annotation lines attempted
};

EvalResult runEvaluation(const string& annotationFile, const string& imageRoot, int maxImages, function<Mat_<uchar>(Mat_<Vec3b>)> skinDetector, const LandmarkParams& p, const string& csvPath = "");

EvalResult runEvaluationGTSeed(const string& annotationFile, const string& imageRoot, int maxImages, const LandmarkParams& p, const string& csvPath = "");

EvalResult runEvaluationVJ(const string& annotationFile, const string& imageRoot, const string& cascadePath, int maxImages, const LandmarkParams& p, const string& csvPath = "");

EvalResult runEvaluationVJFull(const string& annotationFile, const string& imageRoot, const string& faceCascadePath, const string& eyeCascadePath, const string& mouthCascadePath, int maxImages, const string& csvPath = "");