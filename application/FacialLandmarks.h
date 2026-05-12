#pragma once
// FacialLandmarks.h
// Shared types, structs, and helper function declarations.
// Include this in every approach .cpp and in main.cpp.

#include "stdafx.h"
#include "common.h"
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <queue>
#include <algorithm>
#include <climits>

using namespace std;

// ============================================================================
// Shared types
// ============================================================================

// face bounding box + midlines + mask
struct FaceGeometry {
    Rect        bbox;
    int         midRow;
    int         midCol;
    Mat_<uchar> mask;       // closed mask — for bbox + face region
    Mat_<uchar> skinOnly;   // opened-only — still has feature holes (eyes, mouth)
    bool        valid = false;
};

// detected landmark points + flags (detection successful or not)
struct Landmarks {
    Point leftEye  = { -1, -1 };
    Point rightEye = { -1, -1 };
    Point mouth    = { -1, -1 };
    bool  eyesOk   = false;
    bool  mouthOk  = false;
};

// per-blob stats from a labeled image
struct Component {
    int    label = 0;
    int    area  = 0;
    double cy    = 0;
    double cx    = 0;
    int    minR = 0, maxR = 0;
    int    minC = 0, maxC = 0;
};

// ============================================================================
// Shared helper declarations (defined in FacialLandmarksUtils.cpp)
// ============================================================================

bool                isInside(Mat img, int i, int j);
Mat_<uchar>         convertToGray(Mat_<Vec3b> img);
vector<Mat_<uchar>> computeHSV(Mat_<Vec3b> img);
Mat_<uchar>         dilation(Mat_<uchar> src, Mat_<uchar> strel);
Mat_<uchar>         erosion(Mat_<uchar> src, Mat_<uchar> strel);
Mat_<int>           twoPassLabeling(Mat_<uchar> img);
Mat_<uchar>         circStrel(int k);
Mat_<uchar>         opening(Mat_<uchar> src, int ksize);
Mat_<uchar>         closing(Mat_<uchar> src, int ksize);
vector<Component>   componentStats(Mat_<int> labels, Mat_<uchar> mask);
bool                isInsideFace(const Mat_<uchar>& faceMask, int i, int j, int bboxLeft, int bboxRight);
void                drawCross(Mat& img, Point p, Scalar color, int sz = 10);
Mat_<Vec3b>         drawLandmarks(Mat_<Vec3b> img, FaceGeometry face, Landmarks lm);

// ============================================================================
// Approach entry points (one per approach .cpp)
// ============================================================================

void runApproach1(const string& path);   // HSV skin + darkness/redness features
// void runApproach2(const string& path);
