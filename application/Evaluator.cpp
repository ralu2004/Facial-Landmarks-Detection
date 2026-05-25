// Evaluator.cpp
// MTFL annotation parser, NME computation, and evaluation loops
// for all four approaches.

#include "stdafx.h"
#include "Evaluation.h"
#include <fstream>
#include <sstream>

// parse one line of MTFL annotation into GTLandmarks struct
// format: image_path x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 gender smile glasses pose
// points: leftEye, rightEye, nose, mouthLeft, mouthRight
static bool parseLine(const string& line, const string& imageRoot, GTLandmarks& gt) {
	istringstream ss(line);
	string path;
	float x1, y1, x2, y2, x3, y3, x4, y4, x5, y5;
	int gender, smile, glasses, pose;

	if (!(ss >> path >> x1 >> y1 >> x2 >> y2 >> x3 >> y3 >> x4 >> y4 >> x5 >> y5
		>> gender >> smile >> glasses >> pose))
		return false;

	if (path.empty()) return false;

	gt.imagePath = imageRoot + "\\" + path;
	gt.leftEye = Point((int)x1, (int)y1);
	gt.rightEye = Point((int)x2, (int)y2);
	gt.nose = Point((int)x3, (int)y3);
	gt.mouthLeft = Point((int)x4, (int)y4);
	gt.mouthRight = Point((int)x5, (int)y5);
	return true;
}

// normalized error for one point: ||detected - gt|| / IOD
static double pointNME(Point detected, Point gt, double iod) {
	double dx = detected.x - gt.x;
	double dy = detected.y - gt.y;
	return sqrt(dx * dx + dy * dy) / iod;
}

// accumulate per-image NME into result fields
static void accumulateNME(EvalResult& result, const Landmarks& lm, const GTLandmarks& gt, double iod, double& totalEyeNME,
	double& totalMouthNME, double& totalCombinedNME, int& failures) {

	// eye NME
	double eyeNME = 0;
	eyeNME += pointNME(lm.leftEye, gt.leftEye, iod);
	eyeNME += pointNME(lm.rightEye, gt.rightEye, iod);
	eyeNME /= 2.0;
	totalEyeNME += eyeNME;

	// mouth NME 
	double combinedNME = eyeNME;
	if (lm.mouthOk) {
		double mouthNME = pointNME(lm.mouth, gt.mouthCenter(), iod);
		totalMouthNME += mouthNME;
		result.mouthDetected++;
		// combined = mean of eye NME and mouth NME
		combinedNME = (eyeNME + mouthNME) / 2.0;
	}

	totalCombinedNME += combinedNME;
	if (combinedNME > 0.1) failures++;
}

// finalize result fields from accumulators
static void finalizeResult(EvalResult& result, double totalEyeNME, double totalMouthNME, double totalCombinedNME, int failures)
{
	if (result.detected > 0) {
		result.eyeNME = totalEyeNME / result.detected;
		result.meanNME = totalCombinedNME / result.detected;
		result.failureRate = (double)failures / result.detected * 100.0;
	}
	if (result.mouthDetected > 0) {
		result.mouthNME = totalMouthNME / result.mouthDetected;
	}
}

// Approach 1 + 2
// skin-mask based evaluation
EvalResult runEvaluation(const string& annotationFile, const string& imageRoot, int maxImages, function<Mat_<uchar>(Mat_<Vec3b>)> skinDetector, const LandmarkParams& p) {
	EvalResult result;

	ifstream file(annotationFile);
	if (!file.is_open()) {
		cout << "Cannot open annotation file: " << annotationFile << "\n";
		return result;
	}

	double totalEyeNME = 0, totalMouthNME = 0, totalCombinedNME = 0;
	int failures = 0;
	string line;

	while (getline(file, line) && result.total < maxImages) {
		size_t start = line.find_first_not_of(" \t\r\n");
		if (start == string::npos) continue;
		line = line.substr(start);
		if (line.empty()) continue;

		GTLandmarks gt;
		if (!parseLine(line, imageRoot, gt)) continue;
		result.total++;

		Mat_<Vec3b> img = imread(gt.imagePath, IMREAD_COLOR);
		if (img.empty()) continue;

		Mat_<uchar>  skin = skinDetector(img);
		FaceGeometry face = extractFace(skin, p);
		if (!face.valid) continue;

		Landmarks lm = detectLandmarks(img, face, p);
		if (!lm.eyesOk) continue;
		result.detected++;

		double iod = gt.interocularDistance();
		if (iod < 1.0) continue;

		accumulateNME(result, lm, gt, iod, totalEyeNME, totalMouthNME, totalCombinedNME, failures);

		if (result.total % 10 == 0)
			cout << "Evaluated " << result.total << "/" << maxImages
			<< "  eye NME: " << totalEyeNME / result.detected << "\n";
	}

	finalizeResult(result, totalEyeNME, totalMouthNME, totalCombinedNME, failures);
	return result;
}

// Approach 2 upper bound
// GT eye midpoint as seed (theoretical only)
EvalResult runEvaluationGTSeed(const string& annotationFile, const string& imageRoot, int maxImages, const LandmarkParams& p) {
	EvalResult result;

	ifstream file(annotationFile);
	if (!file.is_open()) {
		cout << "Cannot open annotation file: " << annotationFile << "\n";
		return result;
	}

	double totalEyeNME = 0, totalMouthNME = 0, totalCombinedNME = 0;
	int    failures = 0;
	string line;

	while (getline(file, line) && result.total < maxImages) {
		size_t start = line.find_first_not_of(" \t\r\n");
		if (start == string::npos) continue;
		line = line.substr(start);
		if (line.empty()) continue;

		GTLandmarks gt;
		if (!parseLine(line, imageRoot, gt)) continue;
		result.total++;

		Mat_<Vec3b> img = imread(gt.imagePath, IMREAD_COLOR);
		if (img.empty()) continue;

		// GT seed: midpoint between ground truth eyes
		// theoretical upper bound
		Point gtSeed(
			(gt.leftEye.x + gt.rightEye.x) / 2,
			(gt.leftEye.y + gt.rightEye.y) / 2
		);

		Mat_<uchar>  skin = regionGrowingPublic(img, gtSeed);
		FaceGeometry face = extractFace(skin, p);
		if (!face.valid) continue;

		Landmarks lm = detectLandmarks(img, face, p);
		if (!lm.eyesOk) continue;
		result.detected++;

		double iod = gt.interocularDistance();
		if (iod < 1.0) continue;

		accumulateNME(result, lm, gt, iod, totalEyeNME, totalMouthNME, totalCombinedNME, failures);

		if (result.total % 10 == 0)
			cout << "Evaluated " << result.total << "/" << maxImages
			<< "  eye NME: " << totalEyeNME / result.detected << "\n";
	}

	finalizeResult(result, totalEyeNME, totalMouthNME, totalCombinedNME, failures);
	return result;
}

// Approach 3 — VJ face + classical landmarks
EvalResult runEvaluationVJ(const string& annotationFile, const string& imageRoot, const string& cascadePath, int maxImages, const LandmarkParams& p) {
	EvalResult result;

	CascadeClassifier cascade;
	if (!cascade.load(cascadePath)) {
		cout << "Could not load cascade: " << cascadePath << "\n";
		return result;
	}

	ifstream file(annotationFile);
	if (!file.is_open()) {
		cout << "Cannot open annotation file: " << annotationFile << "\n";
		return result;
	}

	double totalEyeNME = 0, totalMouthNME = 0, totalCombinedNME = 0;
	int failures = 0;
	string line;

	while (getline(file, line) && result.total < maxImages) {
		size_t start = line.find_first_not_of(" \t\r\n");
		if (start == string::npos) continue;
		line = line.substr(start);
		if (line.empty()) continue;

		GTLandmarks gt;
		if (!parseLine(line, imageRoot, gt)) continue;
		result.total++;

		Mat_<Vec3b> img = imread(gt.imagePath, IMREAD_COLOR);
		if (img.empty()) continue;

		FaceGeometry face = detectFaceVJ(img, cascade);
		if (!face.valid) continue;

		Landmarks lm = detectLandmarks(img, face, p);
		if (!lm.eyesOk) continue;
		result.detected++;

		double iod = gt.interocularDistance();
		if (iod < 1.0) continue;

		accumulateNME(result, lm, gt, iod, totalEyeNME, totalMouthNME, totalCombinedNME, failures);

		if (result.total % 10 == 0)
			cout << "Evaluated " << result.total << "/" << maxImages
			<< "  eye NME: " << totalEyeNME / result.detected << "\n";
	}

	finalizeResult(result, totalEyeNME, totalMouthNME, totalCombinedNME, failures);
	return result;
}

// Approach 4
// full VJ pipeline: face + eye + mouth cascades
EvalResult runEvaluationVJFull(const string& annotationFile, const string& imageRoot, const string& faceCascadePath, const string& eyeCascadePath, 
	const string& mouthCascadePath, int maxImages) {
	EvalResult result;

	CascadeClassifier faceCascade, eyeCascade, mouthCascade;
	if (!faceCascade.load(faceCascadePath)) { 
		cout << "face cascade failed\n"; 
		return result; 
	}

	if (!eyeCascade.load(eyeCascadePath)) { 
		cout << "eye cascade failed\n";  
		return result; 
	}

	if (!mouthCascade.load(mouthCascadePath)) { 
		cout << "mouth cascade failed\n"; 
		return result; 
	}

	ifstream file(annotationFile);
	if (!file.is_open()) {
		cout << "Cannot open annotation file: " << annotationFile << "\n";
		return result;
	}

	double totalEyeNME = 0, totalMouthNME = 0, totalCombinedNME = 0;
	int failures = 0;
	string line;

	while (getline(file, line) && result.total < maxImages) {
		size_t start = line.find_first_not_of(" \t\r\n");
		if (start == string::npos) continue;
		line = line.substr(start);
		if (line.empty()) continue;

		GTLandmarks gt;
		if (!parseLine(line, imageRoot, gt)) continue;
		result.total++;

		Mat_<Vec3b> img = imread(gt.imagePath, IMREAD_COLOR);
		if (img.empty()) continue;

		FaceGeometry face = detectFaceVJ(img, faceCascade);
		if (!face.valid) continue;

		Landmarks lm = detectLandmarksVJ(img, face, eyeCascade, mouthCascade);
		if (!lm.eyesOk) continue;
		result.detected++;

		double iod = gt.interocularDistance();
		if (iod < 1.0) continue;

		accumulateNME(result, lm, gt, iod, totalEyeNME, totalMouthNME, totalCombinedNME, failures);

		if (result.total % 10 == 0)
			cout << "Evaluated " << result.total << "/" << maxImages
			<< "  eye NME: " << totalEyeNME / result.detected << "\n";
	}

	finalizeResult(result, totalEyeNME, totalMouthNME, totalCombinedNME, failures);
	return result;
}