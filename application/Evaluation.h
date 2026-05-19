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

    // mouth center (average of corners)
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
    double meanNME = 0;         // mean normalized error — lower is better
    double failureRate = 0;     // % images where NME > 0.1
    int    detected = 0;        // images with successful detection
    int    total = 0;           // total images attempted
};

EvalResult runEvaluation(
    const string& annotationFile,
    const string& imageRoot,
    int maxImages,
    function<Mat_<uchar>(Mat_<Vec3b>)> skinDetector,    // function that produces a skin mask from an image: HSV threshold for A1, auto-seed regionGrowing for A2
    const LandmarkParams& p
);

EvalResult runEvaluationGTSeed(
    const string& annotationFile,
    const string& imageRoot,
    int maxImages,
    const LandmarkParams& p
);

EvalResult runEvaluationVJ(
    const string& annotationFile,
    const string& imageRoot,
    const string& cascadePath,
    int maxImages,
    const LandmarkParams& p
);