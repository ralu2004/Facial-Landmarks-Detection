// Approach4_ViolaJonesFull.cpp
// Full Viola-Jones pipeline: face + eye + mouth Haar cascades.
// No classical landmark detection — all three features found by trained cascades.

#include "stdafx.h"
#include "FacialLandmarks.h"
#include <opencv2/objdetect.hpp>

namespace params4 {
    const string FACE_CASCADE = "../data/haarcascade_frontalface_default.xml";
    const string EYE_CASCADE = "../data/haarcascade_eye.xml";
    const string MOUTH_CASCADE = "../data/haarcascade_mcs_mouth.xml";

    const double SCALE_FACTOR = 1.1;
    const int    MIN_NEIGHBORS_FACE = 4;
    const int    MIN_NEIGHBORS_EYE = 2;    // permissive — restricted to upper face ROI
    const int    MIN_NEIGHBORS_MOUTH = 11; // strict — mouth cascade has many false positives
    const int    MIN_FACE_SIZE = 60;
    const int    MIN_EYE_SIZE = 20;
    const int    MIN_MOUTH_SIZE = 20;
}

// detect eyes and mouth using Haar cascades within the face ROI
// eyes searched in upper half, mouth in lower half — reduces false positives
Landmarks detectLandmarksVJ(const Mat_<Vec3b>& img,
    const FaceGeometry& face,
    CascadeClassifier& eyeCascade,
    CascadeClassifier& mouthCascade)
{
    Landmarks lm;
    Mat_<uchar> gray = convertToGray(img);

    // eyes
    int eyeHeight = face.bbox.height / 2;
    Rect eyeRoi(face.bbox.x, face.bbox.y, face.bbox.width, eyeHeight);
    Mat_<uchar> eyeGray = gray(eyeRoi);

    vector<Rect> eyes;
    eyeCascade.detectMultiScale(eyeGray, eyes,
        params4::SCALE_FACTOR, params4::MIN_NEIGHBORS_EYE, 0,
        Size(params4::MIN_EYE_SIZE, params4::MIN_EYE_SIZE));

    if (eyes.size() >= 2) {
        // take two largest
        sort(eyes.begin(), eyes.end(), 
            [](const Rect& a, const Rect& b) { 
                return a.area() > b.area(); 
            });

        Point e1(eyeRoi.x + eyes[0].x + eyes[0].width / 2, eyeRoi.y + eyes[0].y + eyes[0].height / 2);
        Point e2(eyeRoi.x + eyes[1].x + eyes[1].width / 2, eyeRoi.y + eyes[1].y + eyes[1].height / 2);

        // convention: left eye = smaller x
        if (e1.x < e2.x) { lm.leftEye = e1; lm.rightEye = e2; }
        else { lm.leftEye = e2; lm.rightEye = e1; }
        lm.eyesOk = true;
    }

    // mputh
    int mouthTop = face.bbox.height / 2;
    Rect mouthRoi(face.bbox.x, face.bbox.y + mouthTop, face.bbox.width, face.bbox.height - mouthTop);
    Mat_<uchar> mouthGray = gray(mouthRoi);

    vector<Rect> mouths;
    mouthCascade.detectMultiScale(mouthGray, mouths,
        params4::SCALE_FACTOR, params4::MIN_NEIGHBORS_MOUTH, 0,
        Size(params4::MIN_MOUTH_SIZE, params4::MIN_MOUTH_SIZE));

    if (!mouths.empty()) {
        // largest detection
        Rect best = *max_element(mouths.begin(), mouths.end(),
            [](const Rect& a, const Rect& b) { return a.area() < b.area(); });
        lm.mouth = Point(mouthRoi.x + best.x + best.width / 2,
            mouthRoi.y + best.y + best.height / 2);
        lm.mouthOk = true;
    }

    return lm;
}

void runApproach4(const string& path) {
    Mat_<Vec3b> img = imread(path, IMREAD_COLOR);
    if (img.empty()) { 
        cout << "Image not loaded: " << path << "\n"; 
        return; 
    }

    CascadeClassifier faceCascade, eyeCascade, mouthCascade;
    if (!faceCascade.load(params4::FACE_CASCADE)) {
        cout << "Could not load face cascade\n"; 
        return;
    }
    if (!eyeCascade.load(params4::EYE_CASCADE)) {
        cout << "Could not load eye cascade\n"; 
        return;
    }
    if (!mouthCascade.load(params4::MOUTH_CASCADE)) {
        cout << "Could not load mouth cascade\n"; 
        return;
    }

    FaceGeometry face = detectFaceVJ(img, faceCascade);
    if (!face.valid) { 
        cout << "No face found.\n"; 
        return; 
    }

    Landmarks lm = detectLandmarksVJ(img, face, eyeCascade, mouthCascade);
    Mat_<Vec3b> out = drawLandmarks(img, face, lm);

    imshow("A4 Input", img);
    imshow("A4 Landmarks", out);
    waitKey(0);
}