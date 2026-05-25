// FacialLandmarksUtils.cpp
// Definitions of shared helpers used by all approaches.

#include "stdafx.h"
#include "FacialLandmarks.h"

bool isInside(Mat img, int i, int j) {
    if (i >= img.rows || i < 0) return false;
    if (j >= img.cols || j < 0) return false;
    return true;
}

// ITU-R BT.601 luminance weights
Mat_<uchar> convertToGray(Mat_<Vec3b> img) {
    Mat_<uchar> gray(img.size());
    for (int i = 0; i < img.rows; ++i)
        for (int j = 0; j < img.cols; ++j)
            gray(i, j) = 0.299 * img(i, j)[2] + 0.587 * img(i, j)[1] + 0.114 * img(i, j)[0];
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
                if (M == r)      h = 60 * (g - b) / C;
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

// two pass labeling algorithm
// uses 8-connectivity forward neighborhood:
// Np(i,j) = { (i, j-1), (i-1, j-1), (i-1, j), (i-1, j+1) }
Mat_<int> twoPassLabeling(Mat_<uchar> img) {
  
    int dRow[4] = { 0, -1, -1, -1 };    // row offsets
    int dCol[4] = { -1, -1,  0,  1 };   // col offsets
    int label = 0;

    Mat_<int> labels = Mat_<int>::zeros(img.rows, img.cols);
    unordered_map<int, set<int>> edges;

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {

            if (img(i, j) == 0) continue;

            vector<int> L;
            for (int d = 0; d < 4; d++) {
                int ni = i + dRow[d];
                int nj = j + dCol[d];
                if (isInside(img, ni, nj) && labels(ni, nj) > 0)
                    L.push_back(labels(ni, nj));
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
                int x = Q.front(); Q.pop();
                for (auto y : edges[x]) {
                    if (newlabels[y] == 0) {
                        newlabels[y] = newlabel;
                        Q.push(y);
                    }
                }
            }
        }
    }

    for (int i = 0; i < img.rows; ++i)
        for (int j = 0; j < img.cols; ++j)
            if (labels(i, j) > 0)
                labels(i, j) = newlabels[labels(i, j)];

    return labels;
}

// circular structuring element of size k×k
// 255 = part of kernel, 0 = ignored
Mat_<uchar> circStrel(int k) {
    Mat_<uchar> s(k, k, (uchar)0);   
    int   c = k / 2;
    float r = k / 2.0f;
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j)
            if (sqrt((float)((i - c) * (i - c) + (j - c) * (j - c))) <= r)
                s(i, j) = 255;       
    return s;
}

// 0 - background, 255 - object pixel
Mat_<uchar> dilation(Mat_<uchar> src, Mat_<uchar> strel) {
    Mat_<uchar> dst(src.size());
    dst.setTo(0);

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {

            if (src(i, j) == 255) {

                for (int u = 0; u < strel.rows; ++u) {
                    for (int v = 0; v < strel.cols; ++v) {

                        if (strel(u, v) == 255) {
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

                        if (strel(u, v) == 255) {
                            int i2 = i - strel.rows / 2 + u;
                            int j2 = j - strel.cols / 2 + v;
                            if (!isInside(src, i2, j2) || src(i2, j2) != 255) {
                                fits = false;
                                break;
                            }
                        }
                    }
                    if (!fits) break;
                }
                if (fits)
                    dst(i, j) = 255;    // only set if the strel fits perfectly
            }
        }
    }
    return dst;
}

Mat_<uchar> opening(Mat_<uchar> src, int ksize) {
    Mat_<uchar> s = circStrel(ksize);
    return dilation(erosion(src, s), s);
}

Mat_<uchar> closing(Mat_<uchar> src, int ksize) {
    Mat_<uchar> s = circStrel(ksize);
    return erosion(dilation(src, s), s);
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

// scanline test: pixel (i,j) is inside the face contour if there is a
// face-mask pixel both to its left and to its right on the same row
bool isInsideFace(const Mat_<uchar>& faceMask, int i, int j, int bboxLeft, int bboxRight) {
    bool faceLeft = false, faceRight = false;
    int leftLimit = max(0, bboxLeft);
    int rightLimit = min(faceMask.cols, bboxRight);

    for (int k = j - 1; k >= leftLimit; --k)
        if (faceMask(i, k) == 255) { 
            faceLeft = true; 
            break; 
        }

    for (int k = j + 1; k < rightLimit; ++k)
        if (faceMask(i, k) == 255) { 
            faceRight = true; 
            break; 
        }

    return faceLeft && faceRight;
}

void drawCross(Mat& img, Point p, Scalar color, int sz) {
    if (p.x < 0 || p.y < 0) return;
    line(img, Point(p.x - sz, p.y), Point(p.x + sz, p.y), color, 2);
    line(img, Point(p.x, p.y - sz), Point(p.x, p.y + sz), color, 2);
}

Mat_<Vec3b> drawLandmarks(Mat_<Vec3b> img, const FaceGeometry& face, const Landmarks& lm) {
    Mat_<Vec3b> out = img.clone();
    rectangle(out, face.bbox, { 0, 255, 0 }, 2);
    if (lm.eyesOk) {
        drawCross(out, lm.leftEye, { 0, 0, 255 });
        drawCross(out, lm.rightEye, { 0, 0, 255 });
    }
    if (lm.mouthOk) drawCross(out, lm.mouth, { 255, 0, 0 });
    return out;
}

// face = largest connected component of the cleaned skin mask
FaceGeometry extractFace(Mat_<uchar> skinMask, const LandmarkParams& p) {
    FaceGeometry fg;

    // opening removes specks (hands, neck patches)
    // closing fills small holes so the face is one connected component
    Mat_<uchar> opened = opening(skinMask, p.strelKsize);
    Mat_<uchar> cleaned = closing(opened, p.closingKsize);

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
        if (entry.second > faceArea) { 
            faceArea = entry.second; 
            faceLabel = entry.first; 
        }

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

    fg.bbox = Rect(minC, minR, maxC - minC + 1, maxR - minR + 1);
    fg.midRow = (minR + maxR) / 2;
    fg.midCol = (minC + maxC) / 2;
    fg.mask = face;
    fg.skinOnly = faceSkinOnly;
    fg.valid = true;
    return fg;
}

// Otsu's method — finds optimal threshold by maximizing between-class variance
// computed only on pixels inside the face mask
static int otsuThreshold(const Mat_<uchar>& gray, const Mat_<uchar>& faceMask) {
    int hist[256] = {};
    int total = 0;
    for (int i = 0; i < gray.rows; ++i)
        for (int j = 0; j < gray.cols; ++j)
            if (faceMask(i, j) == 255) {
                hist[gray(i, j)]++;
                total++;
            }

    if (total == 0) return 128;

    double sumAll = 0;
    for (int v = 0; v < 256; ++v) 
        sumAll += v * hist[v];

    double bestVar = -1;
    int    bestT = 0;
    double w0 = 0, sum0 = 0;

    for (int t = 0; t < 256; ++t) {
        w0 += hist[t];
        sum0 += t * hist[t];
        double w1 = total - w0;
        if (w0 == 0 || w1 == 0) continue;

        double mu0 = sum0 / w0;
        double mu1 = (sumAll - sum0) / w1;
        double var = (w0 / total) * (w1 / total) * (mu0 - mu1) * (mu0 - mu1);

        if (var > bestVar) { bestVar = var; bestT = t; }
    }
    return bestT;
}

// dark-feature mask: pixels significantly darker than the face mean,
// inside the expected vertical band, not on skin
// useOtsu = true  → threshold found automatically by Otsu's method (better for A1)
// useOtsu = false → threshold = faceMean - darknessOffset (better for A2/A3)
Mat_<uchar> darkFeatureMask(Mat_<Vec3b> img, const FaceGeometry& face, float bandTopFrac, float bandBottomFrac, int darknessOffset, bool useOtsu) {
    Mat_<uchar> gray = convertToGray(img);

    int threshold;
    if (useOtsu) {
        threshold = otsuThreshold(gray, face.mask);
    }
    else {
        // mean - offset: simple adaptive threshold relative to face brightness
        long sum = 0, n = 0;
        for (int i = 0; i < gray.rows; ++i)
            for (int j = 0; j < gray.cols; ++j)
                if (face.mask(i, j) == 255) { 
                    sum += gray(i, j); 
                    n++; 
                }
        int faceMean = (n > 0) ? int(sum / n) : 128;
        threshold = faceMean - darknessOffset;
    }

    int rTop = face.bbox.y + int(bandTopFrac * face.bbox.height);
    int rBottom = face.bbox.y + int(bandBottomFrac * face.bbox.height);
    int bboxLeft = face.bbox.x;
    int bboxRight = face.bbox.x + face.bbox.width;

    Mat_<uchar> mask(gray.size(), (uchar)0);
    for (int i = rTop; i <= rBottom && i < gray.rows; ++i) {
        if (i < 0) continue;
        for (int j = bboxLeft; j < bboxRight; ++j) {
            if (j < 0 || j >= gray.cols) 
                continue;
            if (face.skinOnly(i, j) == 255) 
                continue; 
            if (!isInsideFace(face.mask, i, j, bboxLeft, bboxRight)) 
                continue;
            if (gray(i, j) < threshold) 
                mask(i, j) = 255;
        }
    }

    // small opening to drop single-pixel noise but not enough to merge
    // eyebrows with eyes
    return opening(mask, 3);
}

// redness-based mouth mask: mouths have characteristic redness (R > G, R > B)
Mat_<uchar> mouthFeatureMask(Mat_<Vec3b> img, const FaceGeometry& face, float bandTopFrac, float bandBottomFrac) {
    int rTop = face.bbox.y + int(bandTopFrac * face.bbox.height);
    int rBottom = face.bbox.y + int(bandBottomFrac * face.bbox.height);
    int bboxLeft = face.bbox.x;
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
//   - symmetric         
//   - sensible spacing  
//   - similar area      (left and right eye are about the same size)
// pick the lowest-scoring pair — discrete approximation of Saber & Tekalp (1998)
bool selectEyePair(const vector<Component>& comps, const FaceGeometry& face, Point& leftEye, Point& rightEye, const LandmarkParams& p) {
    int   faceArea = face.bbox.area();
    float minArea = p.minCompAreaFrac * faceArea;
    float maxArea = p.maxCompAreaFrac * faceArea;
    float maxDy = p.maxEyeDyFrac * face.bbox.height;
    float minSep = p.minEyeSepFrac * face.bbox.width;
    float maxSep = p.maxEyeSepFrac * face.bbox.width;

    vector<Component> valid;
    for (const Component& c : comps)
        if (c.area >= minArea && c.area <= maxArea)
            valid.push_back(c);

    if (valid.size() < 2) return false;

    double bestScore = 1e18;
    int    bestI = -1, bestJ = -1;

    for (int i = 0; i < valid.size(); ++i) {
        for (int j = i + 1; j < valid.size(); ++j) {
            const Component& a = valid[i];
            const Component& b = valid[j];

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
            double areaRatio = double(min(a.area, b.area)) / double(max(a.area, b.area));

            // prefer pairs lower in the band (eyebrows are above eyes)
            double avgY = (a.cy + b.cy) / 2.0;
            double yReward = (avgY - face.bbox.y) / face.bbox.height; // 0..1

            // lower is better; weights are empirical
            double score = dy * 1.0
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
        else { leftEye = mirrored; rightEye = detected; }
        return true;
    }

    Point pa((int)valid[bestI].cx, (int)valid[bestI].cy);
    Point pb((int)valid[bestJ].cx, (int)valid[bestJ].cy);

    // convention: left eye = smaller x
    if (pa.x < pb.x) { leftEye = pa; rightEye = pb; }
    else { leftEye = pb; rightEye = pa; }
    return true;
}

// mouth = largest reddish blob in the lower band, near the midline
bool selectMouth(const vector<Component>& comps, const FaceGeometry& face, Point& mouth, const LandmarkParams& p) {
    int   faceArea = face.bbox.area();
    float minArea = p.minCompAreaFrac * faceArea;
    float maxArea = p.maxMouthAreaFrac * faceArea;

    Component best;
    bool   found = false;
    double bestScore = -1;

    for (const Component& c : comps) {
        if (c.area < minArea || c.area > maxArea) continue;

        int width = c.maxC - c.minC + 1;
        int height = c.maxR - c.minR + 1;
        if (height > width * p.mouthAspectMaxRatio) continue;
        if (width < p.mouthMinWidthFrac * face.bbox.width) continue;
        if (width > p.mouthMaxWidthFrac * face.bbox.width) continue;

        // prefer blobs near the horizontal midline
        double horizPenalty = fabs(c.cx - face.midCol) / face.bbox.width;
        double score = c.area * (1.0 - horizPenalty);
        if (score > bestScore) { bestScore = score; best = c; found = true; }
    }
    if (!found) return false;

    // use top of blob rather than centroid — centroid is pulled down by
    // chin/neck redness; minR lands closer to the upper lip
    mouth = Point((int)best.cx, best.minR);
    return true;
}

Landmarks detectLandmarks(Mat_<Vec3b> img, const FaceGeometry& face, const LandmarkParams& p) {
    Landmarks lm;

    // eyes
    {
        Mat_<uchar>       eyeMask = darkFeatureMask(img, face, p.eyeBandTop, p.eyeBandBottom, p.eyeDarknessOffset, p.useOtsu);
        Mat_<int>         eyeLabels = twoPassLabeling(eyeMask);
        vector<Component> comps = componentStats(eyeLabels, eyeMask);
        lm.eyesOk = selectEyePair(comps, face, lm.leftEye, lm.rightEye, p);
    }

    // mouth
    {
        Mat_<uchar>       mouthMask = mouthFeatureMask(img, face, p.mouthBandTop, p.mouthBandBottom);
        Mat_<int>         mouthLabels = twoPassLabeling(mouthMask);
        vector<Component> comps = componentStats(mouthLabels, mouthMask);
        lm.mouthOk = selectMouth(comps, face, lm.mouth, p);
    }

    return lm;
}