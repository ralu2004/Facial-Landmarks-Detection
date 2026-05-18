#include "stdafx.h"
#include "FacialLandmarks.h"

struct ClickState {
    Point point = { -1, -1 };
    Point current = { -1, -1 };
    bool  clicked = false;
};

static void onMouse(int event, int x, int y, int, void* userdata) {
    ClickState* state = (ClickState*)userdata;
    state->current = Point(x, y);
    if (event == EVENT_LBUTTONDOWN) {
        state->point = Point(x, y);
        state->clicked = true;
    }
}

static Point getUserSeedPoint(const Mat_<Vec3b>& img) {
    ClickState state;
    namedWindow("A2 — click on skin");
    setMouseCallback("A2 — click on skin", onMouse, &state);

    while (!state.clicked) {
        Mat_<Vec3b> display = img.clone();

        // draw cross at current mouse position
        if (state.current.x >= 0) {
            int x = state.current.x;
            int y = state.current.y;
            int sz = 10;
            line(display, Point(x - sz, y), Point(x + sz, y), { 0, 255, 0 }, 1);
            line(display, Point(x, y - sz), Point(x, y + sz), { 0, 255, 0 }, 1);
        }

        imshow("A2 — click on skin", display);
        if (waitKey(10) == 27) break; // Esc to cancel
    }

    destroyWindow("A2 — click on skin");
    return state.point;
}

static Mat_<uchar> regionGrowing(const Mat_<Vec3b>& img, Point seed) {
    // compute HSV for the whole image
    auto hsv = computeHSV(img);
    Mat_<uchar>& H = hsv[0];
    Mat_<uchar>& S = hsv[1];
    Mat_<uchar>& V = hsv[2];

    // sample seed color
    int seedH = H(seed.y, seed.x);
    int seedS = S(seed.y, seed.x);
    int seedV = V(seed.y, seed.x);

    Mat_<uchar> mask(img.size(), (uchar)0);
    Mat_<uchar> visited(img.size(), (uchar)0);

    queue<Point> Q;
    Q.push(seed);
    visited(seed.y, seed.x) = 1;

    int dx[4] = { -1, 1,  0, 0 }; //col offset
    int dy[4] = { 0, 0, -1, 1 }; // row offset

    while (!Q.empty()) {
        Point p = Q.front(); Q.pop();
        mask(p.y, p.x) = 255;
        for (int i = 0; i < 4; ++i) {

            int ni = p.y + dy[i];   // row
            int nj = p.x + dx[i];   // col
            Point neigh = { nj, ni };

            if (!isInside(img, ni, nj)) continue;
            if (visited(ni, nj)) continue;

            int dH = abs(H(ni, nj) - seedH);
            int dS = abs(S(ni, nj) - seedS);
            int dV = abs(V(ni, nj) - seedV);
            if (dH > 90) dH = 180 - dH;

            if (dH < 10 && dS < 40 && dV < 50) {
                visited(ni, nj) = 1;
                Q.push(Point(nj, ni));
            }
        }
    }

    return mask;
}

// auto-seed version for evaluation — samples from upper-center of image
Mat_<uchar> detectSkinAutoSeed(Mat_<Vec3b> img) {
    Point seed(img.cols / 2, img.rows / 2); 
    return regionGrowing(img, seed);
}

Mat_<uchar> regionGrowingPublic(Mat_<Vec3b> img, Point seed) {
    return regionGrowing(img, seed);
}

void runApproach2(const string& path) {
    Mat_<Vec3b> img = imread(path, IMREAD_COLOR);
    if (img.empty()) { cout << "Image not loaded: " << path << "\n"; return; }

    Point seed = getUserSeedPoint(img);
    if (seed.x < 0) { cout << "No point selected.\n"; return; }

    Mat_<uchar> skin = regionGrowing(img, seed);

    LandmarkParams p;  // default params — tune later if needed
    p.mouthBandBottom = 0.95f;
    p.mouthBandTop = 0.8f;
    FaceGeometry face = extractFace(skin, p.strelKsize);
    if (!face.valid) { cout << "No face found.\n"; return; }

    Landmarks   lm = detectLandmarks(img, face, p);
    Mat_<Vec3b> out = drawLandmarks(img, face, lm);

    imshow("A2 Input", img);
    imshow("A2 Skin mask", skin);
    imshow("A2 Face mask", face.mask);
    imshow("A2 Landmarks", out);
    waitKey(0);
}