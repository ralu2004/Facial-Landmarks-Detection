// Evaluator.cpp
// MTFL annotation parser, NME computation, and evaluation loops
// for all four approaches. Supports optional CSV export for analysis.

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
	gt.gender = gender;
	gt.smile = smile;
	gt.glasses = glasses;
	gt.pose = pose;
	return true;
}

// normalized error for one point: ||detected - gt|| / IOD
static double pointNME(Point detected, Point gt, double iod) {
	double dx = detected.x - gt.x;
	double dy = detected.y - gt.y;
	return sqrt(dx * dx + dy * dy) / iod;
}

// write CSV header
static void writeCSVHeader(ofstream& csv) {
	csv << "image,pose,smile,glasses,gender,"
		<< "eye_detected,mouth_detected,"
		<< "eye_nme,mouth_nme,combined_nme\n";
}

// write one row to CSV
static void writeCSVRow(ofstream& csv, const GTLandmarks& gt, bool eyeDetected, bool mouthDetected, double eyeNME, double mouthNME, double combinedNME) {
	
	string name = gt.imagePath;
	size_t sep = name.find_last_of("\\/");
	if (sep != string::npos) name = name.substr(sep + 1);

	csv << name << ","
		<< gt.pose << ","
		<< gt.smile << ","
		<< gt.glasses << ","
		<< gt.gender << ","
		<< eyeDetected << ","
		<< mouthDetected << ","
		<< eyeNME << ","
		<< mouthNME << ","
		<< combinedNME << "\n";
}

// accumulate per-image NME into result fields
static void accumulateNME(EvalResult& result, const Landmarks& lm, const GTLandmarks& gt, double iod,
	double& totalEyeNME, double& totalMouthNME, double& totalCombinedNME, int& failures,
	ofstream& csv)
{
	// eye NME
	double eyeNME = 0;
	eyeNME += pointNME(lm.leftEye, gt.leftEye, iod);
	eyeNME += pointNME(lm.rightEye, gt.rightEye, iod);
	eyeNME /= 2.0;
	totalEyeNME += eyeNME;

	// mouth NME
	double mNME = 0;
	double combinedNME = eyeNME;
	if (lm.mouthOk) {
		mNME = pointNME(lm.mouth, gt.mouthCenter(), iod);
		totalMouthNME += mNME;
		result.mouthDetected++;
		combinedNME = (eyeNME + mNME) / 2.0;
	}

	totalCombinedNME += combinedNME;
	if (combinedNME > 0.1) failures++;

	if (csv.is_open())
		writeCSVRow(csv, gt, true, lm.mouthOk, eyeNME, lm.mouthOk ? mNME : -1, combinedNME);
}

// write a row for images where eye detection failed
static void writeFailedRow(ofstream& csv, const GTLandmarks& gt) {
	if (!csv.is_open()) return;
	string name = gt.imagePath;
	size_t sep = name.find_last_of("\\/");
	if (sep != string::npos) name = name.substr(sep + 1);
	csv << name << ","
		<< gt.pose << ","
		<< gt.smile << ","
		<< gt.glasses << ","
		<< gt.gender << ","
		<< "0,0,-1,-1,-1\n";  // -1 = not detected
}

// finalize result fields from accumulators
static void finalizeResult(EvalResult& result, double totalEyeNME, double totalMouthNME, double totalCombinedNME, int failures) {
	if (result.detected > 0) {
		result.eyeNME = totalEyeNME / result.detected;
		result.meanNME = totalCombinedNME / result.detected;
		result.failureRate = (double)failures / result.detected * 100.0;
	}
	if (result.mouthDetected > 0)
		result.mouthNME = totalMouthNME / result.mouthDetected;
}

// Approach 1 + 2
// skin-mask based evaluation
EvalResult runEvaluation(const string& annotationFile, const string& imageRoot, int maxImages, function<Mat_<uchar>(Mat_<Vec3b>)> skinDetector,
	const LandmarkParams& p, const string& csvPath) {
	EvalResult result;

	ifstream file(annotationFile);
	if (!file.is_open()) {
		cout << "Cannot open annotation file: " << annotationFile << "\n";
		return result;
	}

	ofstream csv;
	if (!csvPath.empty()) {
		csv.open(csvPath);
		writeCSVHeader(csv);
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
		if (!face.valid) { 
			writeFailedRow(csv, gt); 
			continue; 
		}

		Landmarks lm = detectLandmarks(img, face, p);
		if (!lm.eyesOk) { 
			writeFailedRow(csv, gt); 
			continue; 
		}
		result.detected++;

		double iod = gt.interocularDistance();
		if (iod < 1.0) continue;

		accumulateNME(result, lm, gt, iod, totalEyeNME, totalMouthNME, totalCombinedNME, failures, csv);

		if (result.total % 50 == 0)
			cout << "Evaluated " << result.total << "/" << maxImages << "\n";
	}

	finalizeResult(result, totalEyeNME, totalMouthNME, totalCombinedNME, failures);
	if (csv.is_open()) csv.close();
	return result;
}

// Approach 2 upper bound 
// GT eye midpoint as seed (theoretical only)
EvalResult runEvaluationGTSeed(const string& annotationFile, const string& imageRoot, int maxImages,
	const LandmarkParams& p, const string& csvPath) {
	EvalResult result;

	ifstream file(annotationFile);
	if (!file.is_open()) {
		cout << "Cannot open annotation file: " << annotationFile << "\n";
		return result;
	}

	ofstream csv;
	if (!csvPath.empty()) {
		csv.open(csvPath);
		writeCSVHeader(csv);
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

		Point gtSeed(
			(gt.leftEye.x + gt.rightEye.x) / 2,
			(gt.leftEye.y + gt.rightEye.y) / 2
		);

		Mat_<uchar>  skin = regionGrowingPublic(img, gtSeed);
		FaceGeometry face = extractFace(skin, p);
		if (!face.valid) { 
			writeFailedRow(csv, gt); 
			continue; 
		}

		Landmarks lm = detectLandmarks(img, face, p);
		if (!lm.eyesOk) { 
			writeFailedRow(csv, gt); 
			continue; 
		}
		result.detected++;

		double iod = gt.interocularDistance();
		if (iod < 1.0) continue;

		accumulateNME(result, lm, gt, iod, totalEyeNME, totalMouthNME, totalCombinedNME, failures, csv);

		if (result.total % 50 == 0)
			cout << "Evaluated " << result.total << "/" << maxImages << "\n";
	}

	finalizeResult(result, totalEyeNME, totalMouthNME, totalCombinedNME, failures);
	if (csv.is_open()) csv.close();
	return result;
}

// Approach 3 
// VJ face + classical landmarks
EvalResult runEvaluationVJ(const string& annotationFile, const string& imageRoot, const string& cascadePath, int maxImages,
	const LandmarkParams& p, const string& csvPath) {
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

	ofstream csv;
	if (!csvPath.empty()) {
		csv.open(csvPath);
		writeCSVHeader(csv);
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
		if (!face.valid) {
			writeFailedRow(csv, gt); 
			continue; 
		}

		Landmarks lm = detectLandmarks(img, face, p);
		if (!lm.eyesOk) { 
			writeFailedRow(csv, gt); 
			continue; 
		}
		result.detected++;

		double iod = gt.interocularDistance();
		if (iod < 1.0) continue;

		accumulateNME(result, lm, gt, iod, totalEyeNME, totalMouthNME, totalCombinedNME, failures, csv);

		if (result.total % 50 == 0)
			cout << "Evaluated " << result.total << "/" << maxImages << "\n";
	}

	finalizeResult(result, totalEyeNME, totalMouthNME, totalCombinedNME, failures);
	if (csv.is_open()) csv.close();
	return result;
}

// Approach 4
// full VJ pipeline: face + eye + mouth cascades
EvalResult runEvaluationVJFull(const string& annotationFile, const string& imageRoot, const string& faceCascadePath,
	const string& eyeCascadePath, const string& mouthCascadePath, int maxImages, const string& csvPath) {
	EvalResult result;

	CascadeClassifier faceCascade, eyeCascade, mouthCascade;
	if (!faceCascade.load(faceCascadePath)) { cout << "face cascade failed\n";  return result; }
	if (!eyeCascade.load(eyeCascadePath)) { cout << "eye cascade failed\n";   return result; }
	if (!mouthCascade.load(mouthCascadePath)) { cout << "mouth cascade failed\n"; return result; }

	ifstream file(annotationFile);
	if (!file.is_open()) {
		cout << "Cannot open annotation file: " << annotationFile << "\n";
		return result;
	}

	ofstream csv;
	if (!csvPath.empty()) {
		csv.open(csvPath);
		writeCSVHeader(csv);
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
		if (!face.valid) { 
			writeFailedRow(csv, gt); 
			continue; 
		}

		Landmarks lm = detectLandmarksVJ(img, face, eyeCascade, mouthCascade);
		if (!lm.eyesOk) { 
			writeFailedRow(csv, gt); 
			continue; 
		}
		result.detected++;

		double iod = gt.interocularDistance();
		if (iod < 1.0) continue;

		accumulateNME(result, lm, gt, iod, totalEyeNME, totalMouthNME, totalCombinedNME, failures, csv);

		if (result.total % 50 == 0)
			cout << "Evaluated " << result.total << "/" << maxImages << "\n";
	}

	finalizeResult(result, totalEyeNME, totalMouthNME, totalCombinedNME, failures);
	if (csv.is_open()) csv.close();
	return result;
}