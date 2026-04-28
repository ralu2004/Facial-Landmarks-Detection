// OpenCVApplication.cpp
// Facial Landmarks Detection
/*
* Pipeline:
* 1 — Detect skin
* 2 — Keep only face region (region of interest)
* 3 — Extract features (eyes, mouth)
* 4 — Estimate landmarks
* 5 — Show results
*
* Convention: 255 = object (white), 0 = background (black)
* eyes - darker regions, upper half of the face
* mouth - darker/redish, pinkish, lower half
*/

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <queue>
#include <algorithm>
#include <climits>

using namespace std;

wchar_t* projectPath;

// tuning constants
namespace params {
    // skin (HSV thresholds)
    const int SKIN_H_MIN = 0, SKIN_H_MAX = 25;
    const int SKIN_S_MIN = 40, SKIN_S_MAX = 255;
    const int SKIN_V_MIN = 60, SKIN_V_MAX = 255;

    // structuring element size (odd)
    const int STREL_KSIZE = 5;

    // eye search band, as fractions of face height from top of bbox
    const float EYE_BAND_TOP = 0.20f; // skip forehead/hair
    const float EYE_BAND_BOTTOM = 0.55f; // stop above the nose tip

    // mouth search band
    const float MOUTH_BAND_TOP = 0.65f;
    const float MOUTH_BAND_BOTTOM = 0.95f;

    // how much darker than the face mean a pixel must be (larger = stricter)
    const int EYE_DARKNESS_OFFSET = 35;

    // accepted component area, as fractions of face area
    const float MIN_COMP_AREA_FRAC = 0.0008f;
    const float MAX_COMP_AREA_FRAC = 0.05f;

    // eye-pair geometry, as fractions of face width/height
    const float MIN_EYE_SEPARATION_FRAC = 0.20f;
    const float MAX_EYE_SEPARATION_FRAC = 0.65f;
    const float MAX_EYE_DY_FRAC = 0.06f;
}

// face bounding box + midlines + mask, kept together so we don't pass them
// around as loose ints
struct FaceGeometry {
    Rect        bbox;
    int         midRow;
    int         midCol;
    Mat_<uchar> mask;
    bool        valid = false;
};

// detected landmark points + flags telling us if detection succeeded
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

bool isInside(Mat img, int i, int j) {
    if (i >= img.rows || i < 0) return false;
    if (j >= img.cols || j < 0) return false;
    return true;
}

Mat_<uchar> convertToGray(Mat_<Vec3b> img) {
    Mat_<uchar> gray(img.size());
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            gray(i, j) = (img(i, j)[2] + img(i, j)[1] + img(i, j)[0]) / 3;
        }
    }
    return gray;
}

vector<Mat_<uchar>> computeHSV(Mat_<Vec3b> img) {
    Mat_<uchar> value(img.size());
    Mat_<uchar> sat(img.size());
    Mat_<uchar> hue(img.size());
    float r, g, b, M, m, C, h, s, v;

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            // normalize BGR values to [0, 1]
            r = img(i, j)[2] / 255.0f;
            g = img(i, j)[1] / 255.0f;
            b = img(i, j)[0] / 255.0f;

            M = max(r, max(g, b)); // maximum of R, G, B
            m = min(r, min(g, b)); // minimum of R, G, B
            C = M - m;             // chroma

            // Value (Brightness)
            v = M;

            // Saturation
            s = (v != 0) ? (C / v) : 0;

            // Hue calculation
            if (C != 0) {
                if (M == r) h = 60 * (g - b) / C;
                else if (M == g) h = 120 + 60 * (b - r) / C;
                else if (M == b) h = 240 + 60 * (r - g) / C;
            }
            else {
                h = 0;
            }

            if (h < 0) h = h + 360;

            // scale values back to 8-bit range [0, 255]
            hue(i, j) = h / 2;     // H is halved to fit in [0, 180]
            sat(i, j) = s * 255.0;
            value(i, j) = v * 255.0;
        }
    }
    return { hue, sat, value };
}

Mat_<uchar> dilation(Mat_<uchar> src, Mat_<uchar> strel) {
    Mat_<uchar> dst(src.size());
    dst.setTo(0); // background

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            if (src(i, j) == 255) { // pixel belongs to an object
                for (int u = 0; u < strel.rows; ++u) {
                    for (int v = 0; v < strel.cols; ++v) {
                        if (strel(u, v) == 0) {
                            int i2 = i + u - strel.rows / 2;
                            int j2 = j + v - strel.cols / 2;
                            if (isInside(dst, i2, j2))
                                dst(i2, j2) = 255;
                        }
                    }
                }
            }
        }
    }
    return dst;
}

Mat_<uchar> erosion(Mat_<uchar> src, Mat_<uchar> strel) {
    Mat_<uchar> dst(src.size());
    dst.setTo(0);

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            if (src(i, j) == 255) {
                bool fits = true;
                for (int u = 0; u < strel.rows; ++u) {
                    for (int v = 0; v < strel.cols; ++v) {
                        if (strel(u, v) == 0) {
                            int i2 = i - strel.rows / 2 + u;
                            int j2 = j - strel.cols / 2 + v;
                            if (!isInside(src, i2, j2) || src(i2, j2) != 255) {
                                fits = false; // does not fit
                                break;
                            }
                        }
                    }
                    if (!fits) break;
                }
                if (fits) {
                    dst(i, j) = 255; // only set if the strel fits perfectly
                }
            }
        }
    }
    return dst;
}

Mat_<int> twoPassLabeling(Mat_<uchar> img) {
    int dx[8] = { -1, 1, 0, 0, -1, -1, 1, 1 };
    int dy[8] = { 0, 0,-1, 1, -1,  1,-1, 1 };
    int label = 0;

    Mat_<int> labels = Mat_<int>::zeros(img.rows, img.cols);
    unordered_map<int, set<int>> edges;

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {

            if (img(i, j) == 0) continue;

            vector<int> L;

            for (int d = 0; d < 4; d++) {
                int ni = i + dx[d];
                int nj = j + dy[d];

                if (isInside(img, ni, nj)) {
                    if (labels(ni, nj) > 0) {
                        L.push_back(labels(ni, nj));
                    }
                }
            }

            if (L.empty()) {
                label++;
                labels(i, j) = label;
            }
            else {
                int x = *min_element(L.begin(), L.end());
                labels(i, j) = x;

                for (auto y : L) {
                    if (y != x) {
                        edges[x].insert(y);
                        edges[y].insert(x);
                    }
                }
            }
        }
    }

    int newlabel = 0;
    vector<int> newlabels(label + 1, 0);

    for (int i = 1; i <= label; ++i) {
        if (newlabels[i] == 0) {
            newlabel++;
            queue<int> Q;
            Q.push(i);
            newlabels[i] = newlabel;

            while (!Q.empty()) {
                int x = Q.front();
                Q.pop();

                for (auto y : edges[x]) {
                    if (newlabels[y] == 0) {
                        newlabels[y] = newlabel;
                        Q.push(y);
                    }
                }
            }
        }
    }

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            if (labels(i, j) > 0) {
                labels(i, j) = newlabels[labels(i, j)];
            }
        }
    }

    return labels;
}

// circular structuring element of size k×k
// (convention: 0 = part of kernel, 255 = ignored)
Mat_<uchar> circStrel(int k) {
    Mat_<uchar> s(k, k, (uchar)255);
    int   c = k / 2;
    float r = k / 2.0f;
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j)
            if (sqrt((float)((i - c) * (i - c) + (j - c) * (j - c))) <= r)
                s(i, j) = 0;
    return s;
}

// open = erode then dilate, removes specks and thin links
Mat_<uchar> opening(Mat_<uchar> src, int ksize) {
    Mat_<uchar> s = circStrel(ksize);
    return dilation(erosion(src, s), s);
}

// close = dilate then erode, fills small holes
Mat_<uchar> closing(Mat_<uchar> src, int ksize) {
    Mat_<uchar> s = circStrel(ksize);
    return erosion(dilation(src, s), s);
}

// HSV thresholds for skin
bool isSkin(uchar H, uchar S, uchar V) {
    return H >= params::SKIN_H_MIN && H <= params::SKIN_H_MAX
        && S >= params::SKIN_S_MIN && S <= params::SKIN_S_MAX
        && V >= params::SKIN_V_MIN && V <= params::SKIN_V_MAX;
}

Mat_<uchar> detectSkin(Mat_<Vec3b> img) {
    vector<Mat_<uchar>> hsv = computeHSV(img);
    Mat_<uchar> mask(img.size(), (uchar)0);
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            uchar H = hsv[0](i, j);
            uchar S = hsv[1](i, j);
            uchar V = hsv[2](i, j);
            mask(i, j) = isSkin(H, S, V) ? 255 : 0;
        }
    }
    return mask;
}

// face = largest connected component of the cleaned skin mask
FaceGeometry extractFace(Mat_<uchar> skinMask) {
    FaceGeometry fg;

    // clean the mask 
    // opening removes specks (hands, neck patches),
    // closing fills small holes (eyes, glasses, mouth)
    Mat_<uchar> cleaned = opening(skinMask, params::STREL_KSIZE);
    cleaned = closing(cleaned, params::STREL_KSIZE);

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
    for (auto& entry : area) {
        if (entry.second > faceArea) {
            faceArea = entry.second;
            faceLabel = entry.first;
        }
    }

    // build face mask + bounding box in one pass
    int minR = INT_MAX, maxR = 0, minC = INT_MAX, maxC = 0;
    Mat_<uchar> face(skinMask.size(), (uchar)0);
    for (int i = 0; i < labels.rows; ++i) {
        for (int j = 0; j < labels.cols; ++j) {
            if (labels(i, j) == faceLabel) {
                face(i, j) = 255;
                if (i < minR) minR = i;
                if (i > maxR) maxR = i;
                if (j < minC) minC = j;
                if (j > maxC) maxC = j;
            }
        }
    }

    fg.bbox = Rect(minC, minR, maxC - minC + 1, maxR - minR + 1);
    fg.midRow = (minR + maxR) / 2;
    fg.midCol = (minC + maxC) / 2;
    fg.mask = face;
    fg.valid = true;
    return fg;
}

// dark-feature mask: pixels significantly darker than the face mean:
// eyes, eyebrows, nostrils, mouth interior
Mat_<uchar> darkFeatureMask(Mat_<Vec3b> img, FaceGeometry face, float bandTopFrac, float bandBottomFrac, int darknessOffset) {
    Mat_<uchar> gray = convertToGray(img);

    // mean intensity inside the face mask only 
    long sum = 0, n = 0;
    for (int i = 0; i < gray.rows; ++i)
        for (int j = 0; j < gray.cols; ++j)
            if (face.mask(i, j) == 255) {
                sum += gray(i, j);
                n++;
            }

    int faceMean = (n > 0) ? int(sum / n) : 128;
    int threshold = faceMean - darknessOffset;

    // search band rows
    int rTop = face.bbox.y + int(bandTopFrac * face.bbox.height);
    int rBottom = face.bbox.y + int(bandBottomFrac * face.bbox.height);

    Mat_<uchar> mask(gray.size(), (uchar)0);
    for (int i = rTop; i <= rBottom && i < gray.rows; ++i) {
        if (i < 0) continue;
        for (int j = face.bbox.x; j < face.bbox.x + face.bbox.width; ++j) {
            if (j < 0 || j >= gray.cols) continue;
            if (face.mask(i, j) == 255) continue; // skin is not a feature
            if (gray(i, j) < threshold) mask(i, j) = 255;
        }
    }

    // small opening to drop single-pixel noise but not enough to merge
    // eyebrows with eyes (that was the old bug)
    return opening(mask, 3);
}

// per-component stats: area + centroid + bbox
vector<Component> componentStats(Mat_<int> labels, Mat_<uchar> mask) {
    map<int, Component> m;
    for (int i = 0; i < labels.rows; ++i) {
        for (int j = 0; j < labels.cols; ++j) {
            int l = labels(i, j);
            if (l == 0 || mask(i, j) == 0) continue;
            Component& c = m[l];
            if (c.area == 0) {
                c.label = l;
                c.minR = c.maxR = i;
                c.minC = c.maxC = j;
            }
            c.area++;
            c.cy += i;
            c.cx += j;
            c.minR = min(c.minR, i);
            c.maxR = max(c.maxR, i);
            c.minC = min(c.minC, j);
            c.maxC = max(c.maxC, j);
        }
    }
    vector<Component> out;
    out.reserve(m.size());
    for (auto& entry : m) {
        Component c = entry.second;
        c.cy /= c.area;
        c.cx /= c.area;
        out.push_back(c);
    }
    return out;
}

// score every pair of candidates on opposite sides of the face midline:
//   - small |dy|        (eyes are level with each other)
//   - symmetric         (equidistant from the midline)
//   - sensible spacing  (fraction of face width)
//   - similar area      (left and right eye are about the same size)
// pick the lowest-scoring pair
bool selectEyePair(vector<Component> comps, FaceGeometry face, Point& leftEye, Point& rightEye) {
    int   faceArea = face.bbox.area();
    float minArea = params::MIN_COMP_AREA_FRAC * faceArea;
    float maxArea = params::MAX_COMP_AREA_FRAC * faceArea;
    float maxDy = params::MAX_EYE_DY_FRAC * face.bbox.height;
    float minSep = params::MIN_EYE_SEPARATION_FRAC * face.bbox.width;
    float maxSep = params::MAX_EYE_SEPARATION_FRAC * face.bbox.width;

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

            double dy = fabs(a.cy - b.cy);
            double sep = fabs(a.cx - b.cx);
            if (dy > maxDy) continue;
            if (sep < minSep || sep > maxSep) continue;

            // distances from midline should be similar
            double dA = fabs(a.cx - face.midCol);
            double dB = fabs(b.cx - face.midCol);
            double asym = fabs(dA - dB);

            // areas should be similar
            double areaRatio = double(min(a.area, b.area))
                / double(max(a.area, b.area));

            // lower is better, weights are empirical
            double score = dy * 1.0 + asym * 1.0 + (1.0 - areaRatio) * 30.0;

            if (score < bestScore) {
                bestScore = score;
                bestI = (int)i;
                bestJ = (int)j;
            }
        }
    }

    if (bestI < 0) return false;

    Point pa((int)valid[bestI].cx, (int)valid[bestI].cy);
    Point pb((int)valid[bestJ].cx, (int)valid[bestJ].cy);

    // convention: left eye = smaller x
    if (pa.x < pb.x) { leftEye = pa; rightEye = pb; }
    else { leftEye = pb; rightEye = pa; }
    return true;
}

// mouth = largest dark blob in the lower band, near the midline
bool selectMouth(vector<Component> comps, FaceGeometry face, Point& mouth) {
    int   faceArea = face.bbox.area();
    float minArea = params::MIN_COMP_AREA_FRAC * faceArea * 2; // mouth is bigger
    float maxArea = params::MAX_COMP_AREA_FRAC * faceArea;

    Component best;
    bool found = false;
    double bestScore = -1;
    for (Component c : comps) {
        if (c.area < minArea || c.area > maxArea) continue;
        // prefer blobs near the horizontal midline
        double horizPenalty = fabs(c.cx - face.midCol) / face.bbox.width;
        double score = c.area * (1.0 - horizPenalty);
        if (score > bestScore) {
            bestScore = score;
            best = c;
            found = true;
        }
    }
    if (!found) return false;
    mouth = Point((int)best.cx, (int)best.cy);
    return true;
}

Landmarks detectLandmarks(Mat_<Vec3b> img, FaceGeometry face) {
    Landmarks lm;

    // eyes
    {
        Mat_<uchar> eyeMask = darkFeatureMask(img, face, params::EYE_BAND_TOP, params::EYE_BAND_BOTTOM, params::EYE_DARKNESS_OFFSET);
        Mat_<int> eyeLabels = twoPassLabeling(eyeMask);
        vector<Component> comps = componentStats(eyeLabels, eyeMask);
        lm.eyesOk = selectEyePair(comps, face, lm.leftEye, lm.rightEye);
    }

    // mouth
    {
        Mat_<uchar> mouthMask = darkFeatureMask(img, face, params::MOUTH_BAND_TOP, params::MOUTH_BAND_BOTTOM, params::EYE_DARKNESS_OFFSET - 10);
        Mat_<int> mouthLabels = twoPassLabeling(mouthMask);
        vector<Component> comps = componentStats(mouthLabels, mouthMask);
        lm.mouthOk = selectMouth(comps, face, lm.mouth);
    }

    return lm;
}

void drawCross(Mat& img, Point p, Scalar color, int sz = 10) {
    if (p.x < 0) return;
    line(img, Point(p.x - sz, p.y), Point(p.x + sz, p.y), color, 2);
    line(img, Point(p.x, p.y - sz), Point(p.x, p.y + sz), color, 2);
}

Mat_<Vec3b> drawLandmarks(Mat_<Vec3b> img, FaceGeometry face, Landmarks lm) {
    Mat_<Vec3b> out = img.clone();
    rectangle(out, face.bbox, { 0, 255, 0 }, 2);
    if (lm.eyesOk) {
        drawCross(out, lm.leftEye, { 0, 0, 255 });
        drawCross(out, lm.rightEye, { 0, 0, 255 });
    }
    if (lm.mouthOk) drawCross(out, lm.mouth, { 255, 0, 0 });
    return out;
}

void runFacialLandmarks(string path) {
    Mat_<Vec3b> img = imread(path, IMREAD_COLOR);
    if (img.empty()) {
        cout << "Image not loaded: " << path << "\n";
        return;
    }

    Mat_<uchar>  skin = detectSkin(img);
    FaceGeometry face = extractFace(skin);
    if (!face.valid) {
        cout << "No face found.\n";
        return;
    }

    Landmarks   lm = detectLandmarks(img, face);
    Mat_<Vec3b> out = drawLandmarks(img, face, lm);

    imshow("Input", img);
    imshow("Skin mask", skin);
    imshow("Face mask", face.mask);
    imshow("Landmarks", out);
    waitKey(0);
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

    runFacialLandmarks("Images/Raluca.jpg");

    return 0;
}