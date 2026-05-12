// FacialLandmarksUtils.cpp
// Definitions of shared helpers used by all approaches.

#include "stdafx.h"
#include "FacialLandmarks.h"

bool isInside(Mat img, int i, int j) {
    if (i >= img.rows || i < 0) return false;
    if (j >= img.cols || j < 0) return false;
    return true;
}

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
            hue(i, j)   = h / 2;     // H is halved to fit in [0, 180]
            sat(i, j)   = s * 255.0;
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
                if (fits)
                    dst(i, j) = 255; // only set if the strel fits perfectly
            }
        }
    }
    return dst;
}

Mat_<int> twoPassLabeling(Mat_<uchar> img) {
    // Np(i,j)={(i,j-1), (i-1,j-1), (i-1,j), (i-1,j+1)}.
    int dx[4] = { 0, -1, -1, -1 };
    int dy[4] = { -1, -1,  0,  1 };
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
                c.minR  = c.maxR = i;
                c.minC  = c.maxC = j;
            }
            c.area++;
            c.cy   += i;
            c.cx   += j;
            c.minR  = min(c.minR, i);
            c.maxR  = max(c.maxR, i);
            c.minC  = min(c.minC, j);
            c.maxC  = max(c.maxC, j);
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

bool isInsideFace(const Mat_<uchar>& faceMask, int i, int j, int bboxLeft, int bboxRight) {
    bool faceLeft = false, faceRight = false;
    int leftLimit  = max(0, bboxLeft);
    int rightLimit = min(faceMask.cols, bboxRight);
    for (int k = j - 1; k >= leftLimit; --k)
        if (faceMask(i, k) == 255) { faceLeft = true; break; }
    for (int k = j + 1; k < rightLimit; ++k)
        if (faceMask(i, k) == 255) { faceRight = true; break; }
    return faceLeft && faceRight;
}

void drawCross(Mat& img, Point p, Scalar color, int sz) {
    if (p.x < 0 || p.y < 0) return;
    line(img, Point(p.x - sz, p.y), Point(p.x + sz, p.y), color, 2);
    line(img, Point(p.x, p.y - sz), Point(p.x, p.y + sz), color, 2);
}

Mat_<Vec3b> drawLandmarks(Mat_<Vec3b> img, FaceGeometry face, Landmarks lm) {
    Mat_<Vec3b> out = img.clone();
    rectangle(out, face.bbox, { 0, 255, 0 }, 2);
    if (lm.eyesOk) {
        drawCross(out, lm.leftEye,  { 0, 0, 255 });
        drawCross(out, lm.rightEye, { 0, 0, 255 });
    }
    if (lm.mouthOk) drawCross(out, lm.mouth, { 255, 0, 0 });
    return out;
}
