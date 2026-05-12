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

void runApproach2(const string& path) {
    Mat_<Vec3b> img = imread(path, IMREAD_COLOR);
    if (img.empty()) { cout << "Image not loaded: " << path << "\n"; return; }

    Point seed = getUserSeedPoint(img);
    if (seed.x < 0) { cout << "No point selected.\n"; return; }

    cout << "Seed point: (" << seed.x << ", " << seed.y << ")\n";
    // region growing comes next
    waitKey(0);
}