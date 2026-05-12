// Approach1_HSV.cpp
// Facial landmarks via HSV skin detection + darkness-based eye detection
// + redness-based mouth detection.

#include "stdafx.h"
#include "FacialLandmarks.h"

// tuning constants
namespace params {
    // skin (HSV thresholds)
    const int SKIN_H_MIN = 0,  SKIN_H_MAX = 25;
    const int SKIN_S_MIN = 40, SKIN_S_MAX = 255;
    const int SKIN_V_MIN = 60, SKIN_V_MAX = 255;

    // structuring element size (odd)
    const int STREL_KSIZE = 5;

    // eye search band, as fractions of face height from top of bbox
    const float EYE_BAND_TOP    = 0.20f; // skip forehead/hair
    const float EYE_BAND_BOTTOM = 0.55f; // stop above the nose tip

    // mouth search band
    const float MOUTH_BAND_TOP    = 0.68f;
    const float MOUTH_BAND_BOTTOM = 0.90f;

    // how much darker than the face mean a pixel must be (larger = stricter)
    const int EYE_DARKNESS_OFFSET = 15;

    // accepted component area, as fractions of face area
    const float MIN_COMP_AREA_FRAC = 0.0003f;
    const float MAX_COMP_AREA_FRAC = 0.05f;

    // eye-pair geometry, as fractions of face width/height
    const float MIN_EYE_SEPARATION_FRAC = 0.20f;
    const float MAX_EYE_SEPARATION_FRAC = 0.65f;
    const float MAX_EYE_DY_FRAC         = 0.10f;
}

// HSV thresholds for skin
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

// face = largest connected component of the cleaned skin mask
static FaceGeometry extractFace(Mat_<uchar> skinMask) {
    FaceGeometry fg;

    // opening removes specks (hands, neck patches),
    // closing fills small holes so the face is one connected component
    Mat_<uchar> opened  = opening(skinMask, params::STREL_KSIZE);
    Mat_<uchar> cleaned = closing(opened, 3);

    Mat_<int> labels = twoPassLabeling(cleaned);

    // count pixels per label
    map<int, int> area;
    for (int i = 0; i < labels.rows; ++i)
        for (int j = 0; j < labels.cols; ++j)
            if (labels(i, j) > 0)
                area[labels(i, j)]++;

    if (area.empty()) return fg;

    // pick the label with the most pixels
    int faceLabel = 0, faceArea = 0;
    for (auto& entry : area)
        if (entry.second > faceArea) { faceArea = entry.second; faceLabel = entry.first; }

    // build face mask + bounding box in one pass
    int minR = INT_MAX, maxR = 0, minC = INT_MAX, maxC = 0;
    Mat_<uchar> face(skinMask.size(), (uchar)0);
    Mat_<uchar> faceSkinOnly(skinMask.size(), (uchar)0);
    for (int i = 0; i < labels.rows; ++i) {
        for (int j = 0; j < labels.cols; ++j) {
            if (labels(i, j) == faceLabel) {
                face(i, j) = 255;
                // skinOnly uses the opened-only mask so eye/mouth holes are still 0
                if (opened(i, j) == 255) faceSkinOnly(i, j) = 255;
                if (i < minR) minR = i;
                if (i > maxR) maxR = i;
                if (j < minC) minC = j;
                if (j > maxC) maxC = j;
            }
        }
    }

    fg.bbox     = Rect(minC, minR, maxC - minC + 1, maxR - minR + 1);
    fg.midRow   = (minR + maxR) / 2;
    fg.midCol   = (minC + maxC) / 2;
    fg.mask     = face;
    fg.skinOnly = faceSkinOnly;
    fg.valid    = true;
    return fg;
}

// dark-feature mask: pixels significantly darker than the face mean,
// inside the expected vertical band, not on skin
static Mat_<uchar> darkFeatureMask(Mat_<Vec3b> img, FaceGeometry face, float bandTopFrac, float bandBottomFrac, int darknessOffset) {
    Mat_<uchar> gray = convertToGray(img);

    // mean intensity inside the face mask only
    long sum = 0, n = 0;
    for (int i = 0; i < gray.rows; ++i)
        for (int j = 0; j < gray.cols; ++j)
            if (face.mask(i, j) == 255) { sum += gray(i, j); n++; }

    int faceMean  = (n > 0) ? int(sum / n) : 128;
    int threshold = faceMean - darknessOffset;

    int rTop      = face.bbox.y + int(bandTopFrac    * face.bbox.height);
    int rBottom   = face.bbox.y + int(bandBottomFrac * face.bbox.height);
    int bboxLeft  = face.bbox.x;
    int bboxRight = face.bbox.x + face.bbox.width;

    Mat_<uchar> mask(gray.size(), (uchar)0);
    for (int i = rTop; i <= rBottom && i < gray.rows; ++i) {
        if (i < 0) continue;
        for (int j = bboxLeft; j < bboxRight; ++j) {
            if (j < 0 || j >= gray.cols) continue;
            if (face.skinOnly(i, j) == 255) continue; // skin is not a feature
            if (!isInsideFace(face.mask, i, j, bboxLeft, bboxRight)) continue;
            if (gray(i, j) < threshold) mask(i, j) = 255;
        }
    }

    // small opening to drop single-pixel noise but not enough to merge
    // eyebrows with eyes
    return opening(mask, 3);
}

// redness-based mouth mask: mouths have characteristic redness (R > G, R > B)
// regardless of darkness, which works on lips that aren't significantly darker
// than skin in grayscale (Hsu et al. 2002 MouthMap approach)
static Mat_<uchar> mouthFeatureMask(Mat_<Vec3b> img, FaceGeometry face) {
    int rTop      = face.bbox.y + int(params::MOUTH_BAND_TOP    * face.bbox.height);
    int rBottom   = face.bbox.y + int(params::MOUTH_BAND_BOTTOM * face.bbox.height);
    int bboxLeft  = face.bbox.x;
    int bboxRight = face.bbox.x + face.bbox.width;

    Mat_<uchar> mask(img.size(), (uchar)0);
    for (int i = rTop; i <= rBottom && i < img.rows; ++i) {
        if (i < 0) continue;
        for (int j = bboxLeft; j < bboxRight; ++j) {
            if (j < 0 || j >= img.cols) continue;
            if (!isInsideFace(face.mask, i, j, bboxLeft, bboxRight)) continue;

            int R = img(i, j)[2];
            int G = img(i, j)[1];
            int B = img(i, j)[0];

            // mouth is reddish: R substantially greater than G, and R > B
            if (R - G > 30 && R > B) mask(i, j) = 255;
        }
    }

    return opening(mask, 3);
}

// score every pair of candidates on opposite sides of the face midline:
//   - small |dy|        (eyes are level with each other)
//   - symmetric         (equidistant from the midline)
//   - sensible spacing  (fraction of face width)
//   - similar area      (left and right eye are about the same size)
// pick the lowest-scoring pair
static bool selectEyePair(vector<Component> comps, FaceGeometry face, Point& leftEye, Point& rightEye) {
    int   faceArea = face.bbox.area();
    float minArea  = params::MIN_COMP_AREA_FRAC * faceArea;
    float maxArea  = params::MAX_COMP_AREA_FRAC * faceArea;
    float maxDy    = params::MAX_EYE_DY_FRAC    * face.bbox.height;
    float minSep   = params::MIN_EYE_SEPARATION_FRAC * face.bbox.width;
    float maxSep   = params::MAX_EYE_SEPARATION_FRAC * face.bbox.width;

    // keep only sensibly-sized components
    vector<Component> valid;
    for (Component c : comps)
        if (c.area >= minArea && c.area <= maxArea)
            valid.push_back(c);

    if (valid.size() < 2) return false;

    double bestScore = 1e18;
    int    bestI = -1, bestJ = -1;

    for (size_t i = 0; i < valid.size(); ++i) {
        for (size_t j = i + 1; j < valid.size(); ++j) {
            Component a = valid[i];
            Component b = valid[j];

            // must straddle the face vertical midline
            bool aLeft = a.cx < face.midCol;
            bool bLeft = b.cx < face.midCol;
            if (aLeft == bLeft) continue;

            double dy  = fabs(a.cy - b.cy);
            double sep = fabs(a.cx - b.cx);
            if (dy > maxDy) continue;
            if (sep < minSep || sep > maxSep) continue;

            // distances from midline should be similar
            double dA   = fabs(a.cx - face.midCol);
            double dB   = fabs(b.cx - face.midCol);
            double asym = fabs(dA - dB);

            // areas should be similar
            double areaRatio = double(min(a.area, b.area)) / double(max(a.area, b.area));

            // prefer pairs lower in the band (eyebrows are above eyes)
            double avgY    = (a.cy + b.cy) / 2.0;
            double yReward = (avgY - face.bbox.y) / face.bbox.height; // 0..1

            // lower is better, weights are empirical
            double score = dy   * 1.0
                         + asym * 1.0
                         + (1.0 - areaRatio) * 30.0
                         - yReward * 5.0;

            if (score < bestScore) {
                bestScore = score;
                bestI = (int)i;
                bestJ = (int)j;
            }
        }
    }

    if (bestI < 0) {
        if (valid.empty()) return false;

        // single-eye fallback: mirror the best candidate across the midline
        const Component* bestSingle = nullptr;
        double bestSingleScore = -1;
        for (const Component& c : valid) {
            double yFrac = (c.cy - face.bbox.y) / face.bbox.height;
            if (yFrac > bestSingleScore) { bestSingleScore = yFrac; bestSingle = &c; }
        }
        if (!bestSingle) return false;

        int mirroredX = 2 * face.midCol - (int)bestSingle->cx;
        Point detected((int)bestSingle->cx, (int)bestSingle->cy);
        Point mirrored(mirroredX, (int)bestSingle->cy);

        if (detected.x < mirrored.x) { leftEye = detected; rightEye = mirrored; }
        else                         { leftEye = mirrored; rightEye = detected; }
        return true;
    }

    Point pa((int)valid[bestI].cx, (int)valid[bestI].cy);
    Point pb((int)valid[bestJ].cx, (int)valid[bestJ].cy);

    // convention: left eye = smaller x
    if (pa.x < pb.x) { leftEye = pa; rightEye = pb; }
    else             { leftEye = pb; rightEye = pa; }
    return true;
}

// mouth = largest reddish blob in the lower band, near the midline
static bool selectMouth(vector<Component> comps, FaceGeometry face, Point& mouth) {
    int   faceArea = face.bbox.area();
    float minArea  = 0.003f * faceArea;
    float maxArea  = 0.25f  * faceArea;

    Component best;
    bool   found     = false;
    double bestScore = -1;
    for (Component c : comps) {
        if (c.area < minArea || c.area > maxArea) continue;

        int width  = c.maxC - c.minC + 1;
        int height = c.maxR - c.minR + 1;
        if (height > width * 1.3) continue;
        if (width < 0.15f * face.bbox.width || width > 0.90f * face.bbox.width) continue;

        // prefer blobs near the horizontal midline
        double horizPenalty = fabs(c.cx - face.midCol) / face.bbox.width;
        double score = c.area * (1.0 - horizPenalty);
        if (score > bestScore) { bestScore = score; best = c; found = true; }
    }
    if (!found) return false;

    // use top of blob rather than centroid — centroid is pulled down by chin/neck
    mouth = Point((int)best.cx, best.minR);
    return true;
}

static Landmarks detectLandmarks(Mat_<Vec3b> img, FaceGeometry face) {
    Landmarks lm;

    // eyes
    {
        Mat_<uchar>       eyeMask   = darkFeatureMask(img, face, params::EYE_BAND_TOP, params::EYE_BAND_BOTTOM, params::EYE_DARKNESS_OFFSET);
        Mat_<int>         eyeLabels = twoPassLabeling(eyeMask);
        vector<Component> comps     = componentStats(eyeLabels, eyeMask);
        lm.eyesOk = selectEyePair(comps, face, lm.leftEye, lm.rightEye);
    }

    // mouth
    {
        Mat_<uchar>       mouthMask   = mouthFeatureMask(img, face);
        Mat_<int>         mouthLabels = twoPassLabeling(mouthMask);
        vector<Component> comps       = componentStats(mouthLabels, mouthMask);
        lm.mouthOk = selectMouth(comps, face, lm.mouth);
    }

    return lm;
}

void runApproach1(const string& path) {
    Mat_<Vec3b> img = imread(path, IMREAD_COLOR);
    if (img.empty()) { cout << "Image not loaded: " << path << "\n"; return; }

    Mat_<uchar>  skin = detectSkin(img);
    FaceGeometry face = extractFace(skin);
    if (!face.valid) { cout << "No face found.\n"; return; }

    Landmarks   lm  = detectLandmarks(img, face);
    Mat_<Vec3b> out = drawLandmarks(img, face, lm);

    imshow("A1 Input",     img);
    imshow("A1 Skin mask", skin);
    imshow("A1 Face mask", face.mask);
    imshow("A1 Landmarks", out);
    waitKey(0);
}
