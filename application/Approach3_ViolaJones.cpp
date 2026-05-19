#include "stdafx.h"
#include "FacialLandmarks.h"
#include <opencv2/objdetect.hpp>
#include <fstream>

namespace params3 {
    // Viola-Jones cascade file — relative to project working directory
    const string CASCADE_PATH = "../data/haarcascade_frontalface_default.xml";

    // detectMultiScale parameters
    const double SCALE_FACTOR = 1.1;  // image pyramid scale step
    const int    MIN_NEIGHBORS = 4;    // detections needed to confirm a face
    const int    MIN_FACE_SIZE = 60;   // minimum face width/height in pixels

    // landmark params 
    const LandmarkParams lp = []() {
        LandmarkParams p;
        p.eyeDarknessOffset = 40;
        return p;
        }();
}

// detect face using Viola-Jones Haar cascade
// returns the largest detected face as a FaceGeometry, or invalid if none found
static FaceGeometry detectFaceVJ(const Mat_<Vec3b>& img, CascadeClassifier& cascade) {
    FaceGeometry fg;

    Mat_<uchar> gray = convertToGray(img);

    vector<Rect> faces;
    cascade.detectMultiScale(gray, faces, params3::SCALE_FACTOR, params3::MIN_NEIGHBORS, 0, Size(params3::MIN_FACE_SIZE, params3::MIN_FACE_SIZE));

    if (faces.empty()) return fg;

    // pick the largest detected face
    Rect best = faces[0];
    for (const Rect& r : faces)
        if (r.area() > best.area()) best = r;

    // build FaceGeometry from the detected bbox
    // no skin mask — VJ gives us the bbox directly
    fg.bbox = best;
    fg.midRow = best.y + best.height / 2;
    fg.midCol = best.x + best.width / 2;
    fg.valid = true;

    // build mask and skinOnly covering the full bbox rectangle
    // (no skin classification — treat entire bbox as face region)
    fg.mask = Mat_<uchar>(img.size(), (uchar)0);
    fg.skinOnly = Mat_<uchar>(img.size(), (uchar)0);
    /*vector<Mat_<uchar>> hsv = computeHSV(img);
    for (int i = best.y; i < best.y + best.height; ++i) {
        for (int j = best.x; j < best.x + best.width; ++j) {
            uchar H = hsv[0](i, j);
            uchar S = hsv[1](i, j);
            uchar V = hsv[2](i, j);
            // same thresholds as Approach 1
            if (H >= 0 && H <= 25 && S >= 40 && S <= 255 && V >= 60 && V <= 255)
                fg.skinOnly(i, j) = 255;
        }
    }*/
    for (int i = best.y; i < best.y + best.height; ++i)
        for (int j = best.x; j < best.x + best.width; ++j)
            fg.mask(i, j) = 255;
    // skinOnly is all zeros — no pixel is "skin" so darkFeatureMask
    // will look at everything inside the bbox

    return fg;
}

void runApproach3(const string& path) {
    Mat_<Vec3b> img = imread(path, IMREAD_COLOR);
    if (img.empty()) { cout << "Image not loaded: " << path << "\n"; return; }

    CascadeClassifier cascade;
    cout << "Loading cascade from: " << params3::CASCADE_PATH << "\n";
    cout << "File exists: " << (ifstream(params3::CASCADE_PATH).good() ? "yes" : "no") << "\n";
    if (!cascade.load(params3::CASCADE_PATH)) {
        cout << "Could not load cascade: " << params3::CASCADE_PATH << "\n";
        return;
    }

    FaceGeometry face = detectFaceVJ(img, cascade);
    if (!face.valid) { cout << "No face found.\n"; return; }

    Landmarks   lm = detectLandmarks(img, face, params3::lp);
    Mat_<Vec3b> out = drawLandmarks(img, face, lm);

    imshow("A3 Input", img);
    imshow("A3 Landmarks", out);
    waitKey(0);
}