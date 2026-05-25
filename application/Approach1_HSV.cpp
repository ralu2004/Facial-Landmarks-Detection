// Approach1_HSV.cpp
// Facial landmarks via HSV + YCbCr skin detection + darkness-based eye
// detection (Otsu threshold) + redness-based mouth detection.

#include "stdafx.h"
#include "FacialLandmarks.h"

namespace params {
    // HSV thresholds — empirically tuned
    // informed by Shaik et al. (2015) and Hassan & Saud (2023)
    const int SKIN_H_MIN = 0, SKIN_H_MAX = 25;
    const int SKIN_S_MIN = 40, SKIN_S_MAX = 255;
    const int SKIN_V_MIN = 60, SKIN_V_MAX = 255;

    // YCbCr thresholds — Rahmat et al. (2016)
    const int SKIN_CR_MIN = 133, SKIN_CR_MAX = 173;
    const int SKIN_CB_MIN = 77, SKIN_CB_MAX = 127;
}

static bool isSkin(uchar H, uchar S, uchar V, uchar Cb, uchar Cr) {
    bool hsvOk = H >= params::SKIN_H_MIN && H <= params::SKIN_H_MAX
        && S >= params::SKIN_S_MIN && S <= params::SKIN_S_MAX
        && V >= params::SKIN_V_MIN && V <= params::SKIN_V_MAX;

    bool ycbcrOk = Cr >= params::SKIN_CR_MIN && Cr <= params::SKIN_CR_MAX
        && Cb >= params::SKIN_CB_MIN && Cb <= params::SKIN_CB_MAX;

    return hsvOk && ycbcrOk;
}

static Mat_<uchar> detectSkin(Mat_<Vec3b> img) {
    vector<Mat_<uchar>> hsv = computeHSV(img);
    Mat_<uchar> mask(img.size(), (uchar)0);

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            uchar B = img(i, j)[0];
            uchar G = img(i, j)[1];
            uchar R = img(i, j)[2];

            uchar Cb = (uchar)(-0.169f * R - 0.331f * G + 0.500f * B + 128);
            uchar Cr = (uchar)(0.500f * R - 0.419f * G - 0.081f * B + 128);

            uchar H = hsv[0](i, j);
            uchar S = hsv[1](i, j);
            uchar V = hsv[2](i, j);

            mask(i, j) = isSkin(H, S, V, Cb, Cr) ? 255 : 0;
        }
    }
    return mask;
}

// public wrapper for evaluation
Mat_<uchar> detectSkinHSV(Mat_<Vec3b> img) {
    return detectSkin(img);
}

void runApproach1(const string& path) {
    Mat_<Vec3b> img = imread(path, IMREAD_COLOR);
    if (img.empty()) { 
        cout << "Image not loaded: " << path << "\n"; 
        return; 
    }

    Mat_<uchar> skin = detectSkin(img);

    // use Otsu's method for eye threshold
    LandmarkParams p;
    p.useOtsu = true;

    FaceGeometry face = extractFace(skin, p);
    if (!face.valid) { 
        cout << "No face found.\n"; 
        return; 
    }

    Landmarks lm = detectLandmarks(img, face, p);
    Mat_<Vec3b> out = drawLandmarks(img, face, lm);

    imshow("A1 Input", img);
    imshow("A1 Skin mask", skin);
    imshow("A1 Face mask", face.mask);
    imshow("A1 Landmarks", out);
    waitKey(0);
}