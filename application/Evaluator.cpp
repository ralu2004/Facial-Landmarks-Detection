#include "stdafx.h"
#include "Evaluation.h"
#include <fstream>
#include <sstream>

// parse one line of MTFL annotation into GTLandmarks struct
static bool parseLine(const string& line, const string& imageRoot, GTLandmarks& gt) {
	istringstream ss(line);
	string path;
	float x1, y1, x2, y2, x3, y3, x4, y4, x5, y5;
	int gender, smile, glasses, pose;

	if (!(ss >> path >> x1 >> y1 >> x2 >> y2 >> x3 >> y3 >> x4 >> y4 >> x5 >> y5 >> gender >> smile >> glasses >> pose))
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

// normalized error for one point
// ||detected - gt|| / interocular_distance
static double pointNME(Point detected, Point gt, double iod) {
	double dx = detected.x - gt.x;
	double dy = detected.y - gt.y;
	return sqrt(dx * dx + dy * dy) / iod;
}

EvalResult runEvaluation(const string& annotationFile, const string& imageRoot, int maxImages, function<Mat_<uchar>(Mat_<Vec3b>)> skinDetector, const LandmarkParams& p) {
	EvalResult result;

	ifstream file(annotationFile);
	if (!file.is_open()) {
		cout << "Cannot open annotation file: " << annotationFile << "\n";
		return result;
	}

	double totalNME = 0;
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
		if (img.empty()) {
			cout << "Failed to load: " << gt.imagePath << "\n";
			continue;
		}

		Mat_<uchar> skin = skinDetector(img);
		FaceGeometry face = extractFace(skin, p);
		if (!face.valid) continue;

		Landmarks lm = detectLandmarks(img, face, p);

		// only score if both eyes detected
		if (!lm.eyesOk) continue;
		result.detected++;

		double iod = gt.interocularDistance();
		if (iod < 1.0) continue; // degenerate annotation

		// NME = mean over detected landmarks
		double nme = 0;
		int count = 0;

		nme += pointNME(lm.leftEye, gt.leftEye, iod); 
		count++;
		nme += pointNME(lm.rightEye, gt.rightEye, iod); 
		count++;

		nme /= count;
		totalNME += nme;
		if (nme > 0.1) failures++;

		if (result.total % 10 == 0) 
			cout << "Evaluated " << result.total << "/" << maxImages << "  mean NME so far: " << totalNME / result.detected << "\n";
	}

	if (result.detected > 0) {
		result.meanNME = totalNME / result.detected;
		result.failureRate = (double)failures / result.detected * 100.0;
	}

	return result;
}

EvalResult runEvaluationGTSeed(
	const string& annotationFile,
	const string& imageRoot,
	int maxImages,
	const LandmarkParams& p)
{
	EvalResult result;
	ifstream file(annotationFile);
	if (!file.is_open()) {
		cout << "Cannot open annotation file: " << annotationFile << "\n";
		return result;
	}

	double totalNME = 0;
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

		// GT seed — midpoint between ground truth eyes
		// this is a theoretical
		// upper bound only, not a fair evaluation
		Point gtSeed(
			(gt.leftEye.x + gt.rightEye.x) / 2,
			(gt.leftEye.y + gt.rightEye.y) / 2
		);

		Mat_<uchar> skin = regionGrowingPublic(img, gtSeed);
		FaceGeometry face = extractFace(skin, p);
		if (!face.valid) continue;

		Landmarks lm = detectLandmarks(img, face, p);
		if (!lm.eyesOk) continue;
		result.detected++;

		double iod = gt.interocularDistance();
		if (iod < 1.0) continue;

		double nme = 0;
		nme += pointNME(lm.leftEye, gt.leftEye, iod);
		nme += pointNME(lm.rightEye, gt.rightEye, iod);
		nme /= 2;

		totalNME += nme;
		if (nme > 0.1) failures++;

		if (result.total % 10 == 0)
			cout << "Evaluated " << result.total << "/" << maxImages
			<< "  mean NME so far: " << totalNME / result.detected << "\n";
	}

	if (result.detected > 0) {
		result.meanNME = totalNME / result.detected;
		result.failureRate = (double)failures / result.detected * 100.0;
	}
	return result;
}

EvalResult runEvaluationVJ(
	const string& annotationFile,
	const string& imageRoot,
	const string& cascadePath,
	int maxImages,
	const LandmarkParams& p)
{
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

	double totalNME = 0;
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

		// VJ face detection — no skin mask needed
		Mat_<uchar> gray = convertToGray(img);
		vector<Rect> faces;
		cascade.detectMultiScale(gray, faces, 1.1, 4, 0, Size(60, 60));
		if (faces.empty()) continue;

		// largest face
		Rect best = faces[0];
		for (const Rect& r : faces)
			if (r.area() > best.area()) best = r;

		FaceGeometry face;
		face.bbox = best;
		face.midRow = best.y + best.height / 2;
		face.midCol = best.x + best.width / 2;
		face.valid = true;
		face.mask = Mat_<uchar>(img.size(), (uchar)0);
		face.skinOnly = Mat_<uchar>(img.size(), (uchar)0);
		for (int i = best.y; i < best.y + best.height; ++i)
			for (int j = best.x; j < best.x + best.width; ++j)
				face.mask(i, j) = 255;

		Landmarks lm = detectLandmarks(img, face, p);
		if (!lm.eyesOk) continue;
		result.detected++;

		double iod = gt.interocularDistance();
		if (iod < 1.0) continue;

		double nme = 0;
		nme += pointNME(lm.leftEye, gt.leftEye, iod);
		nme += pointNME(lm.rightEye, gt.rightEye, iod);
		nme /= 2;

		totalNME += nme;
		if (nme > 0.1) failures++;

		if (result.total % 10 == 0)
			cout << "Evaluated " << result.total << "/" << maxImages
			<< "  mean NME: " << totalNME / result.detected << "\n";
	}

	if (result.detected > 0) {
		result.meanNME = totalNME / result.detected;
		result.failureRate = (double)failures / result.detected * 100.0;
	}
	return result;
}