// Approach3_ViolaJones.cpp
// Facial landmarks via Viola-Jones face detection + classical landmark pipeline.
// Face bbox from Haar cascade; eyes and mouth from darkness/redness thresholds.

#include "stdafx.h"
#include "FacialLandmarks.h"
#include <opencv2/objdetect.hpp>

namespace params3 {
    const string CASCADE_PATH = "../data/haarcascade_frontalface_default.xml";

    // detectMultiScale parameters
    const double SCALE_FACTOR = 1.1;  // image pyramid scale step
    const int    MIN_NEIGHBORS = 4;   // detections needed to confirm a face
    const int    MIN_FACE_SIZE = 60;  // minimum face width/height in pixels

    // landmark params 
    // eyeDarknessOffset raised to 40 because skinOnly is
    // all-zero (no skin filter), so the darkness threshold must be stricter
    // to avoid picking up non-eye dark regions
    // useOtsu = false: mean-offset works better than Otsu on VJ's tight bbox
    const LandmarkParams lp = []() {
        LandmarkParams p;
        p.eyeDarknessOffset = 40;
        p.useOtsu = false;
        return p;
        }();
}

// detect face using Viola-Jones Haar cascade
// returns the largest detected face as a FaceGeometry, or invalid if none found
FaceGeometry detectFaceVJ(const Mat_<Vec3b>& img, CascadeClassifier& cascade) {
    FaceGeometry fg;

    Mat_<uchar> gray = convertToGray(img);
    vector<Rect> faces;
    cascade.detectMultiScale(
        gray, faces,
        params3::SCALE_FACTOR, params3::MIN_NEIGHBORS,
        0, Size(params3::MIN_FACE_SIZE, params3::MIN_FACE_SIZE));

    if (faces.empty()) return fg;

    // pick the largest detected face
    Rect best = faces[0];
    for (const Rect& r : faces)
        if (r.area() > best.area()) best = r;

    fg.bbox = best;
    fg.midRow = best.y + best.height / 2;
    fg.midCol = best.x + best.width / 2;
    fg.valid = true;

    // mask = full bbox rectangle
    // skinOnly = all zeros, no skin classification inside VJ bbox
    // darkFeatureMask will examine all pixels inside the bbox
    fg.mask = Mat_<uchar>(img.size(), (uchar)0);
    fg.skinOnly = Mat_<uchar>(img.size(), (uchar)0);
    for (int i = best.y; i < best.y + best.height; ++i)
        for (int j = best.x; j < best.x + best.width; ++j)
            fg.mask(i, j) = 255;

    return fg;
}

void runApproach3(const string& path) {
    Mat_<Vec3b> img = imread(path, IMREAD_COLOR);
    if (img.empty()) { 
        cout << "Image not loaded: " << path << "\n"; 
        return; 
    }

    CascadeClassifier cascade;
    if (!cascade.load(params3::CASCADE_PATH)) {
        cout << "Could not load cascade: " << params3::CASCADE_PATH << "\n";
        return;
    }

    FaceGeometry face = detectFaceVJ(img, cascade);
    if (!face.valid) { 
        cout << "No face found.\n"; 
        return; 
    }

    Landmarks lm = detectLandmarks(img, face, params3::lp);
    Mat_<Vec3b> out = drawLandmarks(img, face, lm);

    imshow("A3 Input", img);
    imshow("A3 Landmarks", out);
    waitKey(0);
}