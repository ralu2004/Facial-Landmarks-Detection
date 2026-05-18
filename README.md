# Facial Landmarks Detection

Classical facial landmark detection in C++ using OpenCV.
Detects eyes and mouth on face images using two independent approaches,
evaluated on the MTFL dataset with Normalized Mean Error (NME).

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
The two approaches differ only in how the skin/face mask is produced.
Everything from `extractFace` onward is shared code.

---

## Project structure

```
OpenCVApplication.cpp     — interactive menu, dispatches to approaches
FacialLandmarks.h         — shared types (FaceGeometry, Landmarks,
                            Component, LandmarkParams) + declarations
FacialLandmarksUtils.cpp  — shared primitives and full landmark pipeline
Approach1_HSV.cpp         — Approach 1: HSV skin thresholding
Approach2_Interactive.cpp — Approach 2: region growing from user click
Evaluation.h              — GTLandmarks, EvalResult structs
Evaluator.cpp             — MTFL parser, NME computation, eval loop
```

---

## Approach 1 — HSV skin detection

### What it does
Converts BGR to HSV manually, then classifies each pixel as skin if:
```
H ∈ [0, 25]   (hue — skin tones, 0–180 scale)
S ∈ [40, 255] (saturation — excludes near-grey pixels)
V ∈ [60, 255] (value — excludes very dark pixels)
```
The largest connected component of the cleaned skin mask is taken as
the face.

### Why HSV
RGB mixes luminance and chrominance, making skin look very different
under different lighting. HSV separates them: hue captures color
independently of brightness, saturation distinguishes vivid colors
from grey shadows. This is the standard justification in the skin
detection literature [2][3].

### Threshold values
**Empirically tuned**, informed by ranges reported in [3][4]. The
specific values `H ≤ 25, S ≥ 40, V ≥ 60` are not lifted from any
single paper — they were selected by testing on dataset images. 
Different datasets may require different values. This is a known 
limitation of fixed-threshold approaches [2].

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
The resulting mask is then processed identically to Approach 1.

### Why region growing
Region growing is a classical segmentation technique from Gonzalez &
Woods [6], Chapter 10. Unlike fixed HSV thresholds, it adapts to the
specific image — the seed pixel's color becomes the reference, so
any skin tone or lighting condition is handled automatically, as long
as the seed is on skin.

### Tolerance values
**Empirically chosen.** The V tolerance is intentionally loose (50)
because the same face can have intensity variation of 60+ between
forehead and cheek due to lighting. The H tolerance (10) is tighter
because hue is more stable across the face. No single citation
specifies these exact values for skin; they were tuned on the test
images.

### GT seed experiment
To isolate the contribution of seed selection vs landmark localization,
we also ran Approach 2 with the ground truth eye midpoint as seed
(theoretical upper bound). Results were almost identical to the center
seed (see Evaluation results), confirming that **seed quality is not
the primary bottleneck** — the landmark localization is.

---

## Shared landmark pipeline

### Face extraction (`extractFace`)
After the skin mask is produced, morphological cleanup is applied:
- **Opening** (erode then dilate) removes small speckles — hands,
  neck patches, background noise. 
- **Closing** (dilate then erode) fills small holes so the face is
  one connected component — eyes, mouth, glasses frames appear as
  holes in the skin mask.

Order matters: opening first, then closing. Opening first removes
noise that closing would otherwise amplify.

A **circular structuring element** is used (not square) because it
treats all directions equally, as appropriate for a rotationally 
symmetric object like a face.

`skinOnly` stores the opened-only mask (before closing). This
preserves eye/mouth holes for downstream feature detection, while
`mask` (after closing) is used for the face region boundary. This
separation is necessary because closing fills the eye holes that
the feature detectors depend on.

### Eye detection (`darkFeatureMask` + `selectEyePair`)

**Feature mask:** inside the eye band (20–55% of face height from
top of bbox), pixels that are:
1. Not classified as skin in `skinOnly`
2. Inside the face contour (`isInsideFace` scanline test)
3. Darker than `faceMean - 15` in grayscale

The darkness threshold is **empirically tuned** to `15`. Higher
values (stricter) missed light-colored eyes; lower values let in too
much skin shadow. The face-mean-relative threshold (rather than a
fixed value) adapts to per-image brightness.

The **eye band fractions** `[0.20, 0.55]` are anthropometrically
motivated — eyes are anatomically in the upper half of the face,
below the hairline. The specific fractions are empirically tuned,
informed by the proportions in Farkas (1994) [9].

**Eye pair selection:** every candidate component pair is scored:
```
score = |Δy| × 1.0
      + asymmetry × 1.0
      + (1 - areaRatio) × 30.0
      - yReward × 5.0
```
Lower score = better pair. The scoring concept — using bilateral
facial symmetry as a cost function — follows Saber & Tekalp (1998)
[5], who introduced symmetry-based cost functions for eye/nose/mouth
localization. Our implementation is a **discrete approximation**:
rather than a continuous pixel-level symmetry energy, we score
connected component pairs. The weights (1.0, 30.0, 5.0) are
**empirical** — tuned so that area dissimilarity is the dominant
term (eyes are the same size), with position terms as tiebreakers.

Hard constraints (not scored, just filtered):
- Must straddle the face vertical midline (left and right eye)
- `|Δy| ≤ 10%` of face height (eyes are level)
- Separation ∈ `[20%, 65%]` of face width (interpupillary distance)
  — anthropometric range from Farkas [9]

**Single-eye fallback:** if no valid pair is found, the best single
candidate is mirrored across the face midline. This handles partial
occlusion and asymmetric lighting.

### Mouth detection (`mouthFeatureMask` + `selectMouth`)

**Feature mask:** in the mouth band (68–90% of face height), pixels
where `R - G > 30 AND R > B`. This detects redness characteristic
of lips. The approach is inspired by the MouthMap concept from Hsu
et al. [1], which uses chrominance to detect mouths. Our
implementation is a **simplified approximation** using raw BGR
channels instead of the full YCbCr chrominance ratio.

The threshold `R - G > 30` is **empirically tuned** — lower values
let in too much cheek/chin redness, higher values missed pale lips.

**Mouth selection:** largest redness blob near the face midline,
wider than tall (lips are wider than they are tall), width between
15% and 90% of face width.

The mouth y-coordinate uses `blob.minR` (top of blob) rather than
the centroid. This is an **empirical correction** — the large
redness region's centroid is systematically pulled below the lips
by chin and neck pixels. Using the top of the blob places the mark
closer to the upper lip.

---

## Evaluation

### Metric — Normalized Mean Error (NME)
```
NME = (1/N) × Σ ||detected_i − gt_i|| / IOD
```
where `IOD` = inter-ocular distance (Euclidean distance between
ground truth eye centers). Normalizing by IOD makes the error
scale-independent across different image sizes and face scales.
**Failure threshold:** NME > 0.1 (standard in the 300-W benchmark
[10]).

### Dataset — MTFL
12,995 face images annotated with 5 landmarks: left eye, right eye,
nose, left mouth corner, right mouth corner [8]. Download:
`http://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip`

We compare our detected `leftEye`, `rightEye` against the MTFL
eye annotations, and `mouth` against the midpoint of the two mouth
corner annotations.

### Results (1000 images from `training.txt`)

| Approach | Detection rate | Mean NME | Failure rate |
|---|---|---|---|
| A1 — HSV skin | 69.7% | 1.121 | 100% |
| A2 — center seed | 57.8% | 0.776 | 100% |
| A2 — GT seed (upper bound) | 69.3% | 0.786 | 100% |

### Key findings

**A2 outperforms A1 on NME** (0.776 vs 1.121) when it detects a
face. Region growing produces a cleaner face mask because it adapts
to each image's actual skin color rather than applying fixed thresholds.

**Detection rate trade-off:** A1 detects more faces (69.7% vs 57.8%)
because its fixed HSV thresholds reliably find *some* skin on most
images, even if the resulting mask is noisy. A2's center seed
occasionally misses the face entirely.

**GT seed experiment:** Using the ground truth eye midpoint as seed
improves A2's detection rate to 69.3% (matching A1) but leaves NME
unchanged (0.786 ≈ 0.776). This confirms that the seed selection
is not the primary source of error — **the landmark localization
algorithm itself is the bottleneck**.

**100% failure rate:** Neither approach reaches NME < 0.1 on any
image. The 0.1 threshold corresponds to error < 10% of inter-ocular
distance — roughly 3px on a 250×250 image with 30px IOD. Our
detections are in the right face region but not precise enough for
this strict threshold. The primary causes:
- Fixed redness/darkness thresholds are not robust across diverse
  faces, lighting, and facial hair
- The symmetry scoring selects approximate eye locations, not
  precise pupil centers
- No subpixel refinement

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
1. Approach 1 - HSV skin detection
2. Approach 2 - Region growing (click on skin in the popup window)
3. Evaluate Approach 1 on MTFL
4. Evaluate Approach 2 on MTFL (auto center seed)
5. Evaluate A2 upper bound (GT seed — theoretical, uses ground truth)
0. Exit
```

For options 3–5, enter when prompted:
- Annotation file: `<extract_path>\MTFL\training.txt`
- Image root: `<extract_path>\MTFL`
- Number of images (e.g. 1000)

---

## References

**[1]** Hsu, R.-L., Abdel-Mottaleb, M., & Jain, A. K. (2002).
Face detection in color images.
*IEEE Transactions on Pattern Analysis and Machine Intelligence*, 24(5), 696–706.
https://doi.org/10.1109/34.1000242
*Pipeline structure (skin → face candidate → feature map → landmarks).
MouthMap concept (chrominance-based mouth detection).*

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
*Multi-color-space skin detection (HSV + YCbCr + normalized RGB).*

**[5]** Saber, E., & Tekalp, A. M. (1998).
Frontal-view face detection and facial feature extraction using color,
shape and symmetry-based cost functions.
*Pattern Recognition Letters*, 19(8), 669–680.
https://doi.org/10.1016/S0167-8655(98)00044-0
*Symmetry-based cost functions for eye/nose/mouth localization.
Our selectEyePair is a discrete approximation of this framework.*

**[6]** Gonzalez, R. C., & Woods, R. E. (2018).
*Digital Image Processing* (4th ed.). Pearson.
*Morphological operations (Ch. 9) and region growing (Ch. 10).*

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
*Image and Vision Computing*, Special Issue on Facial Landmark
Localisation.
Annotations: https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
*NME metric and failure threshold (0.1) used in our evaluation.*

---

## Known limitations

- **100% failure rate on MTFL** — NME never reaches the 0.1 threshold
- **Fixed thresholds** — HSV and redness values tuned on a small set
  of images, not robust across diverse skin tones and lighting
- **Brown/dark eyes on tan skin** — misclassified as skin by HSV
  thresholding, leaving no eye holes for detection
- **Fixed seed fails** — A2 center seed misses faces that aren't
  centered in the image
- **No tilted face handling** — vertical bands are axis-aligned
- **Mouth systematic bias** — redness extends to chin/neck, pushing
  the detected mouth point slightly below the lips despite the
  `minR` correction
- **No subpixel precision** — landmark positions are at integer
  pixel coordinates

## What would improve results

- **Viola-Jones face detector** (OpenCV `CascadeClassifier`) —
  reliable bounding box without skin detection, removes the entire
  skin-detection failure mode
- **Multi-color-space skin** — AND HSV with YCbCr thresholds, as in
  [4], significantly more robust across skin tones
- **Adaptive thresholding** — Otsu's method per-band rather than
  fixed `mean - offset`
- **dlib 68-point detector** — production-grade classical (non-deep)
  landmark localization, ~1ms per image