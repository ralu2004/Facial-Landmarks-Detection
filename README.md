# Facial Landmarks Detection

Classical (non-ML) facial landmark detection in C++ using OpenCV.
Detects eyes and mouth on face images using four approaches, evaluated
on the MTFL dataset with Normalized Mean Error (NME).

---

## Pipeline overview

```
Input image
    → Skin detection / face region extraction    [approach-specific]
    → Face bounding box + skinOnly mask          [extractFace]
    → Dark-feature mask (eyes)                   [darkFeatureMask]
    → Redness mask (mouth)                       [mouthFeatureMask]
    → Connected component analysis               [twoPassLabeling + componentStats]
    → Symmetry-scored eye pair selection         [selectEyePair]
    → Midline-anchored mouth selection           [selectMouth]
    → Landmark overlay                           [drawLandmarks]
```

The pipeline structure follows Hsu, Abdel-Mottaleb & Jain (2002) [1]:
skin detection → face candidate → feature map → landmark selection.
Approaches 1–3 share the landmark pipeline from `extractFace` onward.
Approach 4 replaces the landmark pipeline entirely with cascade detectors.

---

## Project structure

```
application/
    OpenCVApplication.cpp        — interactive menu, dispatches to approaches
    FacialLandmarks.h            — shared types (FaceGeometry, Landmarks,
                                   Component, LandmarkParams) + declarations
    FacialLandmarksUtils.cpp     — shared primitives and full landmark pipeline
    Approach1_HSV.cpp            — Approach 1: HSV + YCbCr skin detection
    Approach2_Interactive.cpp    — Approach 2: region growing from user click
    Approach3_ViolaJones.cpp     — Approach 3: VJ face + classical landmarks
    Approach4_ViolaJonesFull.cpp — Approach 4: full VJ pipeline
    Evaluation.h                 — GTLandmarks, EvalResult structs
    Evaluator.cpp                — MTFL parser, NME computation, eval loops
    data/
        haarcascade_frontalface_default.xml
        haarcascade_eye.xml
        haarcascade_mcs_mouth.xml
        haarcascade_eye_tree_eyeglasses.xml

analysis/
    analyze_results.py           — stratified analysis + chart generation
    extract_frontal.py           — filter MTFL annotations by pose==1
    plot_*.png                   — generated charts

results/
    results_A1.csv               — per-image results for each approach
    results_A2.csv
    results_A2_upper_bound.csv
    results_A3.csv
    results_A4.csv
    results_A*_frontal.csv       — same, filtered to frontal images only
    frontal_only.txt             — MTFL annotation subset: pose==1 (136 images)
```

---

## Approach 1 — HSV + YCbCr skin detection

### What it does
Classifies each pixel as skin if it passes **both** HSV and YCbCr
thresholds simultaneously:
```
HSV:   H ∈ [0, 25], S ∈ [40, 255], V ∈ [60, 255]
YCbCr: Cr ∈ [133, 173], Cb ∈ [77, 127]
```
The largest connected component of the cleaned skin mask is taken as
the face. Eye detection uses Otsu's adaptive threshold.

### Why HSV
RGB mixes luminance and chrominance, making skin look very different
under different lighting. HSV separates them: hue captures color
independently of brightness, saturation distinguishes vivid colors
from grey shadows. This is the standard justification in the skin
detection literature [2][3].

### Why YCbCr in addition
HSV alone produces false positives on warm-toned backgrounds. Adding
YCbCr thresholds — which separate luminance from chrominance differently
— significantly reduces these false positives. A pixel must satisfy
both color spaces to be classified as skin, following the multi-color-space
approach of Rahmat et al. (2016) [4].

### Why Otsu for eye detection
With the HSV+YCbCr mask, the face region has a more varied intensity
distribution. Otsu's method [6] automatically finds the optimal darkness
threshold by maximizing between-class variance on the face region's
grayscale histogram — better than a fixed `mean - offset` for this approach.

### Threshold values
HSV values **empirically tuned**, informed by ranges in [3][4].
YCbCr thresholds taken directly from Rahmat et al. (2016) [4].

---

## Approach 2 — Region growing

### What it does
The user clicks a pixel on the skin. A BFS expands outward from that
seed, accepting 4-connected neighbors whose HSV is within a tolerance
of the seed:
```
ΔH < 10   (hue — with wraparound at 180)
ΔS < 40   (saturation)
ΔV < 50   (value — loose, accounts for lighting variation across face)
```
The resulting mask is processed identically to Approach 1, but eye
detection uses `mean - offset` (not Otsu) since region growing already
produces a well-bounded face mask.

### Why region growing
Region growing is a classical segmentation technique from Gonzalez &
Woods [6], Chapter 10. Unlike fixed thresholds, it adapts to each
image — the seed pixel's color becomes the reference, handling any
skin tone or lighting automatically, as long as the seed is on skin.

### Tolerance values
**Empirically chosen.** V tolerance is loose (50) because the same
face can vary 60+ intensity units between forehead and cheek. H
tolerance (10) is tighter because hue is more stable across the face.

### GT seed experiment
Running A2 with the ground truth eye midpoint as seed (theoretical
upper bound) improved detection rate but left NME essentially unchanged,
confirming that **seed quality is not the primary bottleneck** — the
landmark localization is.

---

## Approach 3 — Viola-Jones face detection + classical landmarks

### What it does
Uses OpenCV's `CascadeClassifier` with `haarcascade_frontalface_default.xml`
to detect the face bounding box. Eyes and mouth are then found using
the same classical pipeline as A1/A2, with a stricter darkness offset
(`eyeDarknessOffset = 40`) since there is no skin mask to exclude skin
pixels. Eye detection uses `mean - offset` (not Otsu).

### Why Viola-Jones for face detection
The Viola-Jones algorithm (2001) [11] uses Haar-like features, integral
images for O(1) feature evaluation, AdaBoost feature selection, and a
cascade classifier to rapidly reject non-face windows. Trained on
thousands of face images, it is robust to lighting and skin tone
variation that defeats hand-crafted thresholds.

### Key difference from A1/A2
No skin detection step. The face bounding box comes directly from the
cascade, giving `skinOnly = all zeros` — `darkFeatureMask` examines
every pixel inside the bbox.

---

## Approach 4 — Full Viola-Jones pipeline

### What it does
Extends Approach 3 by replacing classical landmark detection with two
additional Haar cascades:
- `haarcascade_eye.xml` — detects eyes in the upper half of the face
  ROI (minNeighbors = 2)
- `haarcascade_mcs_mouth.xml` — detects mouth in the lower half
  (minNeighbors = 11, strict to reduce false positives)

The two largest eye detections are taken as left/right eye. The largest
mouth detection is the mouth point.

### Trade-off vs Approach 3
A4 achieves lower NME when it detects both eyes, because trained
cascades locate eye centers more precisely than darkness thresholding.
However, detection rate is lower because the eye cascade requires good
frontal alignment — conditions that VJ face detection alone does not
require. This is the classic precision-recall trade-off.

### Cascade source
The MCS mouth cascade was authored by Castrillón-Santana, University
of Las Palmas de Gran Canaria, trained on 7000 positive samples [12].

---

## Shared landmark pipeline

### Face extraction (`extractFace`)
- **Opening** (erode then dilate) removes speckles. Gonzalez & Woods [6], Ch. 9.
- **Closing** (dilate then erode) fills holes so the face is one
  connected component.

Order matters: opening first removes noise that closing would amplify.
A **circular structuring element** is used for isotropy.

`skinOnly` stores the opened-only mask, preserving eye/mouth holes
for feature detection. `mask` (post-closing) defines the face boundary.

### Eye detection (`darkFeatureMask` + `selectEyePair`)

**Feature mask:** inside the eye band (20–55% of face height), pixels
that are not classified as skin, inside the face contour (scanline
test), and darker than the threshold.

**Otsu's method** (A1): maximizes between-class variance on face pixels.
**Mean-offset** (A2/A3): `threshold = faceMean - offset`. Works better
when the face mask is already tight and the histogram is not strongly
bimodal.

**Eye pair selection** scores every candidate pair:
```
score = |Δy| × 1.0 + asymmetry × 1.0 + (1 - areaRatio) × 30.0 - yReward × 5.0
```
Concept from Saber & Tekalp (1998) [5]; weights are **empirical**.
Hard constraints: must straddle face midline; `|Δy| ≤ 10%` of face
height; separation ∈ `[20%, 65%]` of face width (Farkas [9]).

### Mouth detection (`mouthFeatureMask` + `selectMouth`)

**Feature mask:** mouth band (68–90% for A1/A3, 80–95% for A2), pixels
where `R - G > 30 AND R > B`. Simplified approximation of MouthMap [1].

**Selection:** largest redness blob near midline, wider than tall, width
between 15% and 90% of face width. Y-coordinate uses `blob.minR` — top
of blob rather than centroid, which is pulled below lips by chin/neck.

---

## Evaluation

### Metric — Normalized Mean Error (NME)
Eye NME and Mouth NME are computed separately:
```
Eye NME   = mean(||leftEye_det - leftEye_gt|| / IOD,
                 ||rightEye_det - rightEye_gt|| / IOD)
Mouth NME = ||mouth_det - mouthCenter_gt|| / IOD  (when detected)
Combined  = mean(Eye NME, Mouth NME)              (when both detected)
```
where `IOD` = inter-ocular distance between ground truth eye centers.
**Failure threshold:** NME > 0.1 (standard in 300-W benchmark [10]).

### Dataset — MTFL
12,995 face images annotated with 5 landmarks [8]. Download:
[http://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip](http://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip)

Pose distribution in full dataset: 74.4% right profile, 11.1% left
profile, 11.1% upward, 2.0% downward, **1.4% frontal (136 images)**.
The dataset is heavily skewed toward non-frontal — see
`analysis/plot_pose_distribution.png`.

We evaluate on 2000 images from `training.txt` (all poses) and
separately on the 136 frontal images (`results/frontal_only.txt`).

### Results — all poses (2000 images)

| Approach | Eye det.% | Mouth det.% | Eye NME | Mouth NME | Combined NME |
|---|---|---|---|---|---|
| A1 — HSV+YCbCr+Otsu | 72.1% | 63.0% | 1.121 | 0.803 | 0.982 |
| A2 — center seed | 57.7% | 50.2% | 0.801 | 0.717 | 0.765 |
| A2 — GT seed (upper bound) | 68.7% | 58.8% | 0.796 | 0.802 | 0.800 |
| A3 — VJ+classical | 95.5% | 78.3% | 0.743 | 0.324 | 0.568 |
| A4 — VJ full | 39.6% | 35.9% | 0.685 | 0.403 | 0.559 |

### Results — frontal only (136 images, pose==1)

| Approach | Eye det.% | Mouth det.% | Eye NME | Mouth NME | Combined NME |
|---|---|---|---|---|---|
| A1 — HSV+YCbCr+Otsu | 64.0% | 53.7% | 2.262 | 2.080 | 2.187 |
| A2 — center seed | 58.8% | 49.3% | 1.940 | 1.601 | 1.776 |
| A2 — GT seed (upper bound) | 51.5% | 43.4% | 1.864 | 1.672 | 1.768 |
| A3 — VJ+classical | 74.3% | 61.8% | 1.522 | 0.581 | 1.150 |
| A4 — VJ full | 16.2% | 14.0% | 1.091 | 0.570 | 0.886 |

### Key findings

**A3 dominates on detection rate** (95.5%) across all poses. Viola-Jones
face detection is robust to pose and lighting variation that defeats
hand-crafted skin detection. This directly confirms that trained appearance
models outperform classical thresholding for face localization.

**A4 best NME, worst detection rate.** A4's eye cascade achieves the
lowest eye NME (0.685) and mouth NME (0.403) when it detects — cascade
detectors are more accurate than darkness/redness thresholding. But
detection rate collapses to 39.6% because the eye cascade requires good
frontal alignment. Classic precision-recall trade-off.

**A3 mouth NME (0.324) significantly better than eyes (0.743).** On
well-detected faces with reliable bounding boxes, redness-based mouth
detection outperforms darkness-based eye detection. Mouth color is a
more discriminative feature than relative darkness within the face bbox.

**GT seed experiment isolates the bottleneck.** A2 with GT seed improves
detection rate (57.7% → 68.7%) but leaves eye NME essentially unchanged
(0.801 → 0.796). This confirms that **skin detection quality is not the
primary source of error** — the landmark localization algorithm itself
is the bottleneck.

**Frontal images are harder on this dataset.** Counter-intuitively, all
approaches achieve worse NME on the 136 frontal images than on the full
dataset. This is because MTFL's frontal images are rare and atypical — the
dataset is 74% right profile, so algorithms implicitly tune to that case.
A4's eye cascade collapses on frontal (16.2% detection) — the 136 frontal
images in MTFL may have unusual characteristics.

**Pose stratification** (see `analysis/plot_nme_by_pose.png`): upward-tilt
images (pose=4) are easier than profile images for all approaches. A3 and
A4 show the smallest NME degradation across poses, while A1 degrades most
severely on left/right profiles.

**Glasses reduce A4 detection rate significantly** (26.5% vs 42.8%
without glasses) — the eye cascade confuses glasses frames with eye
boundaries. A1-A3 are less sensitive to glasses.

**Smile has minimal effect** — smiling images are slightly easier than
neutral across all approaches, likely because smiling portraits tend to
be more frontal and well-lit.

**100% failure rate:** No approach reaches NME < 0.1 on any image.
Classical methods without trained landmark detectors cannot achieve
sub-10% IOD precision on a diverse real-world dataset.

---

## Analysis scripts

```bash
cd analysis

# generate all tables and charts from CSV results
python analyze_results.py

# extract frontal-only annotation subset from MTFL training.txt
python extract_frontal.py <path_to_training.txt> <output_path>
```

Charts saved to `analysis/`, tables printed to console.

---

## Setup

### Requirements
- Visual Studio 2019/2022
- OpenCV 4.9 (included in `OpenCV/` folder)

### Build
1. Open `OpenCVApplication.sln`
2. Build → x64 → Debug

### Usage
Run the project. A file dialog opens — pick any image from `Images/`.
Then select from the menu:

```
1. Approach 1 - HSV + YCbCr skin detection
2. Approach 2 - Region growing (click on skin in the popup window)
3. Approach 3 - Viola Jones face + classical landmarks
4. Approach 4 - Viola Jones full pipeline
5. Evaluate Approach 1 on MTFL
6. Evaluate Approach 2 on MTFL (auto center seed)
7. Evaluate A2 upper bound (GT seed — theoretical, uses ground truth)
8. Evaluate Approach 3 on MTFL
9. Evaluate Approach 4 on MTFL
0. Exit
```

For options 5–9, enter when prompted:
- Annotation file: `<extract_path>\MTFL\training.txt` (or `frontal_only.txt`)
- Image root: `<extract_path>\MTFL`
- Number of images
- Output CSV path (optional, leave empty to skip)

---

## References

**[1]** Hsu, R.-L., Abdel-Mottaleb, M., & Jain, A. K. (2002).
Face detection in color images.
*IEEE Transactions on Pattern Analysis and Machine Intelligence*, 24(5), 696–706.
https://doi.org/10.1109/34.1000242
*Pipeline structure. MouthMap concept (chrominance-based mouth detection).*

**[2]** Shaik, K. B., Ganesan, P., Kalist, V., Sathish, B. S., &
Jenitha, J. M. M. (2015).
Comparative study of skin color detection and segmentation in HSV and
YCbCr color space.
*Procedia Computer Science*, 57, 41–48.
https://doi.org/10.1016/j.procs.2015.07.362
*Justification for HSV over RGB for skin detection.*

**[3]** Hassan, E. K., & Saud, J. H. (2023).
HSV color model and logical filter for human skin detection.
*AIP Conference Proceedings*, 2457, 040003.
https://doi.org/10.1063/5.0120025
*HSV threshold ranges for skin detection.*

**[4]** Rahmat, R. F., Chairunnisa, T., Gunawan, D., & Sitompul, O. S.
(2016). Skin color segmentation using multi-color space threshold.
*3rd International Conference on Computer and Information Sciences.*
https://doi.org/10.1109/ICCOINS.2016.7783247
*Multi-color-space skin detection (HSV + YCbCr). YCbCr threshold values.*

**[5]** Saber, E., & Tekalp, A. M. (1998).
Frontal-view face detection and facial feature extraction using color,
shape and symmetry-based cost functions.
*Pattern Recognition Letters*, 19(8), 669–680.
https://doi.org/10.1016/S0167-8655(98)00044-0
*Symmetry-based cost functions for landmark localization.
selectEyePair is a discrete approximation of this framework.*

**[6]** Gonzalez, R. C., & Woods, R. E. (2018).
*Digital Image Processing* (4th ed.). Pearson.
*Morphological operations (Ch. 9), region growing (Ch. 10),
Otsu's thresholding method (Ch. 10).*

**[7]** Rosenfeld, A., & Pfaltz, J. L. (1966).
Sequential operations in digital picture processing.
*Journal of the ACM*, 13(4), 471–494.
https://doi.org/10.1145/321356.321357
*Two-pass connected component labeling algorithm.*

**[8]** Zhang, Z., Luo, P., Loy, C. C., & Tang, X. (2014).
Facial landmark detection by deep multi-task learning.
*European Conference on Computer Vision (ECCV).*
MTFL dataset: http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html
*Evaluation dataset (12,995 images, 5 landmarks per face).*

**[9]** Farkas, L. G. (1994).
*Anthropometry of the Head and Face* (2nd ed.). Raven Press.
WorldCat: https://www.worldcat.org/title/29600219
*Facial proportions motivating eye/mouth band fractions.*

**[10]** Sagonas, C., Antonakos, E., Tzimiropoulos, G., Zafeiriou, S.,
& Pantic, M. (2016).
300 faces in-the-wild challenge: Database and results.
*Image and Vision Computing*, Special Issue on Facial Landmark Localisation.
Annotations: https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
*NME metric and failure threshold (0.1) used in our evaluation.*

**[11]** Viola, P., & Jones, M. (2001).
Rapid object detection using a boosted cascade of simple features.
*IEEE Conference on Computer Vision and Pattern Recognition (CVPR).*
https://doi.org/10.1109/CVPR.2001.990517
*Viola-Jones face/eye/mouth detection: Haar features, integral image,
AdaBoost, cascade classifier.*

**[12]** Castrillón-Santana, M., Déniz-Suárez, O., Antón-Canalis, L.,
& Lorenzo-Navarro, J. (2007).
Face and facial feature detection evaluation.
*1st Spanish Workshop on Biometrics*, Girona.
*MCS mouth Haar cascade (haarcascade_mcs_mouth.xml),
trained on 7000 positive samples.*

---

## Known limitations

- **100% failure rate on MTFL** — NME never reaches the 0.1 threshold
- **Fixed thresholds** — HSV, YCbCr, and redness values not robust
  across all skin tones and lighting conditions
- **Eye cascade fails on non-frontal faces** — A4 detection rate drops
  on tilted or poorly lit faces
- **Fixed seed fails** — A2 center seed misses non-centered faces
- **No tilted face handling** — vertical bands are axis-aligned
- **Mouth systematic bias** — `minR` correction partially mitigates
  chin/neck redness but does not eliminate it
- **No subpixel precision** — landmark positions at integer coordinates
- **Custom implementations are slow** — morphology and labeling written
  from scratch; full dataset evaluation takes several hours