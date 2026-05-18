// Approach1_HSV.cpp
// Facial landmarks via HSV skin detection + darkness-based eye detection
// + redness-based mouth detection.

#include "stdafx.h"
#include "FacialLandmarks.h"

namespace params {
    const int SKIN_H_MIN = 0, SKIN_H_MAX = 25;
    const int SKIN_S_MIN = 40, SKIN_S_MAX = 255;
    const int SKIN_V_MIN = 60, SKIN_V_MAX = 255;
}

static bool isSkin(uchar H, uchar S, uchar V) {
    return H >= params::SKIN_H_MIN && H <= params::SKIN_H_MAX
        && S >= params::SKIN_S_MIN && S <= params::SKIN_S_MAX
        && V >= params::SKIN_V_MIN && V <= params::SKIN_V_MAX;
}

static Mat_<uchar> detectSkin(Mat_<Vec3b> img) {
    vector<Mat_<uchar>> hsv = computeHSV(img);
    Mat_<uchar> mask(img.size(), (uchar)0);
    for (int i = 0; i < img.rows; ++i)
        for (int j = 0; j < img.cols; ++j) {
            uchar H = hsv[0](i, j);
            uchar S = hsv[1](i, j);
            uchar V = hsv[2](i, j);
            mask(i, j) = isSkin(H, S, V) ? 255 : 0;
        }
    return mask;
}

// wrapper for evaluation
Mat_<uchar> detectSkinHSV(Mat_<Vec3b> img) {
    return detectSkin(img);
}

void runApproach1(const string& path) {
    Mat_<Vec3b> img = imread(path, IMREAD_COLOR);
    if (img.empty()) { cout << "Image not loaded: " << path << "\n"; return; }

    Mat_<uchar>  skin = detectSkin(img);

    LandmarkParams p;  // uses defaults from the struct
    FaceGeometry face = extractFace(skin, p.strelKsize);
    if (!face.valid) { cout << "No face found.\n"; return; }

    Landmarks   lm = detectLandmarks(img, face, p);
    Mat_<Vec3b> out = drawLandmarks(img, face, lm);

    imshow("A1 Input", img);
    imshow("A1 Skin mask", skin);
    imshow("A1 Face mask", face.mask);
    imshow("A1 Landmarks", out);
    waitKey(0);
}
