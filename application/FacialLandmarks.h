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
#include <opencv2/objdetect.hpp>

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
    Point leftEye = { -1, -1 };
    Point rightEye = { -1, -1 };
    Point mouth = { -1, -1 };
    bool  eyesOk = false;
    bool  mouthOk = false;
};

// per-blob stats from a labeled image
struct Component {
    int    label = 0;
    int    area = 0;
    double cy = 0;
    double cx = 0;
    int    minR = 0, maxR = 0;
    int    minC = 0, maxC = 0;
};

struct LandmarkParams {
    // morphology
    int   strelKsize = 5;               // opening kernel size
    int   closingKsize = 3;             // closing kernel size — smaller than opening to fill tiny gaps without filling eyes/mouth

    // eye search band, as fractions of face height from top of bbox
    float eyeBandTop = 0.20f;           // skip forehead/hair
    float eyeBandBottom = 0.55f;        // stop above the nose tip
    int   eyeDarknessOffset = 15;       // how much darker than face mean (empirical)
    // only used when useOtsu = false

    // whether to use Otsu's method for eye darkness threshold
    bool  useOtsu = false;

    // mouth search band
    float mouthBandTop = 0.68f;
    float mouthBandBottom = 0.90f;

    // component area filter, as fractions of face bbox area
    float minCompAreaFrac = 0.0003f;
    float maxCompAreaFrac = 0.05f;

    // eye-pair geometry, as fractions of face width/height
    float minEyeSepFrac = 0.20f;
    float maxEyeSepFrac = 0.65f;
    float maxEyeDyFrac = 0.10f;

    // mouth geometry
    float maxMouthAreaFrac = 0.25f;     // redness blobs can be large
    float mouthAspectMaxRatio = 1.3f;   // max height/width — mouths are wider than tall
    float mouthMinWidthFrac = 0.15f;    // min width as fraction of face width
    float mouthMaxWidthFrac = 0.90f;    // max width as fraction of face width
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
Mat_<Vec3b>         drawLandmarks(Mat_<Vec3b> img, const FaceGeometry& face, const Landmarks& lm);
FaceGeometry        extractFace(Mat_<uchar> skinMask, const LandmarkParams& p);
Mat_<uchar>         darkFeatureMask(Mat_<Vec3b> img, const FaceGeometry& face, float bandTopFrac, float bandBottomFrac, int darknessOffset, bool useOtsu);
Mat_<uchar>         mouthFeatureMask(Mat_<Vec3b> img, const FaceGeometry& face, float bandTopFrac, float bandBottomFrac);
bool                selectEyePair(const vector<Component>& comps, const FaceGeometry& face, Point& leftEye, Point& rightEye, const LandmarkParams& p);
bool                selectMouth(const vector<Component>& comps, const FaceGeometry& face, Point& mouth, const LandmarkParams& p);
Landmarks           detectLandmarks(Mat_<Vec3b> img, const FaceGeometry& face, const LandmarkParams& p);
FaceGeometry        detectFaceVJ(const Mat_<Vec3b>& img, CascadeClassifier& cascade);
Landmarks           detectLandmarksVJ(const Mat_<Vec3b>& img, const FaceGeometry& face, CascadeClassifier& eyeCascade, CascadeClassifier& mouthCascade);

// ============================================================================
// Approach entry points (one per approach .cpp)
// ============================================================================

void runApproach1(const string& path);
void runApproach2(const string& path);
void runApproach3(const string& path);
void runApproach4(const string& path);

// ============================================================================
// Evaluation wrappers
// ============================================================================

Mat_<uchar> detectSkinHSV(Mat_<Vec3b> img);
Mat_<uchar> detectSkinAutoSeed(Mat_<Vec3b> img);
Mat_<uchar> regionGrowingPublic(Mat_<Vec3b> img, Point seed);