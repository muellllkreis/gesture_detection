#include "binary_mask_creator.h"

using namespace cv;
using namespace std;

binary_mask_creator::binary_mask_creator()
{

}

Mat binary_mask_creator::computeBinaryMask(vector<Hand_ROI> roi, Mat I_HSV)
{
	// create binary threshold image per ROI
	float min_h = std::numeric_limits<float>::infinity();
	float max_h = 0.f;
	float min_s = std::numeric_limits<float>::infinity();
	float max_s = 0.f;
	for (Hand_ROI r : roi) {
		if (r.roi_mean[0] < min_h) {
			min_h = r.roi_mean[0];
		}
		if (r.roi_mean[0] > max_h) {
			max_h = r.roi_mean[0];
		}
		if (r.roi_mean[1] < min_s) {
			min_s = r.roi_mean[1];
		}
		if (r.roi_mean[1] > max_s) {
			max_s = r.roi_mean[1];
		}
	}

	Mat binary_mask;
	inRange(I_HSV, Scalar(min_h - thresholdCorrectionLow, min_s - thresholdCorrectionLow*0.01f, 0), Scalar(max_h + thresholdCorrectionHigh, max_s + thresholdCorrectionHigh*0.01f, 1), binary_mask);

	// Blur binary image to smoothen it
	Mat binary_blur(I_HSV.size(), CV_8UC1);
	Mat binary_blur_uc;	
	medianBlur(binary_mask, binary_blur, 5);
	binary_blur.convertTo(binary_blur_uc, CV_8UC1);	

	return binary_blur_uc;
}

Mat binary_mask_creator::removeFacesFromMask(Mat& binary_mask, Mat& frameGray)
{
	String faceClassifierFileName = "face_classifier/haarcascade_frontalface_alt.xml";
	String profileClassifierFileName = "face_classifier/haarcascade_profileface.xml";
	CascadeClassifier faceCascadeClassifier, profileCascadeClassifier;

	if (!faceCascadeClassifier.load(faceClassifierFileName))
		throw runtime_error("can't load file " + faceClassifierFileName);

	if (!profileCascadeClassifier.load(profileClassifierFileName))
		throw runtime_error("can't load file " + profileClassifierFileName);

	vector<Rect> faces;
	vector<Rect> profile;

	faceCascadeClassifier.detectMultiScale(frameGray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(120, 120));
	profileCascadeClassifier.detectMultiScale(frameGray, profile, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(120, 120));

	faces.insert(faces.end(), profile.begin(), profile.end());

	for (Rect f : faces) {
		cout << "found face" << endl;
		// note that we are removing the face from binary_blur_uc as it is the Mat used in the next step for contour detection
		rectangle(binary_mask, f, Scalar(0, 0, 0), -1);
	}
	return binary_mask;
}


std::vector<Mat> binary_mask_creator::createBinaryMask(VideoCapture& cap, bool removeFace)
{
	Mat I1;
	Mat I_BGR;
	Mat I_HSV;
	vector<Hand_ROI> roi;
	vector<Hand_ROI>().swap(roi);
	while (true)
	{
		cap >> I1;
		//flip(I1, I1, 1);
		double col_offset = I1.cols * 0.05;
		double row_offset = I1.rows * 0.05;
		// create bgr and hsv version of image
		I1.convertTo(I_BGR, CV_32F, 1.0 / 255.0, 0.);
		cvtColor(I_BGR, I_HSV, COLOR_BGR2HSV);

		// add points of interest that we are using to compute color averages. later, for a webcam feed,
		// the hand will have to be positioned over ROIs like this
		Point hand_center = Point(3 * (int)(I1.cols / 4), (int)(I1.rows / 2));
		//roi.push_back(Hand_ROI(hand_center, I_HSV));
		roi.push_back(Hand_ROI(Point(hand_center.x + col_offset, hand_center.y - row_offset), I_HSV));
		roi.push_back(Hand_ROI(Point(hand_center.x - col_offset, hand_center.y + row_offset), I_HSV));
		roi.push_back(Hand_ROI(Point(hand_center.x + col_offset, hand_center.y + row_offset), I_HSV));
		roi.push_back(Hand_ROI(Point(hand_center.x - col_offset, hand_center.y - row_offset), I_HSV));
		//roi.push_back(Hand_ROI(Point(hand_center.x, hand_center.y - row_offset), I_HSV));
		//roi.push_back(Hand_ROI(Point(hand_center.x, hand_center.y - row_offset * 2), I_HSV));

		if (waitKey(10) == 32) {
			break;
		}

		for (Hand_ROI r : roi) {
			r.draw_rectangle(I_BGR);
		}

		namedWindow("Image", CV_WINDOW_KEEPRATIO);
		resizeWindow("Image", 960, 540);
		imshow("Image", I_BGR);
	}

	//compute binary mask
	Mat BM = computeBinaryMask(roi, I_HSV);

	if (removeFace)
	{
		Mat frameGray;
		cvtColor(I1, frameGray, CV_BGR2GRAY);
		cout << "type: " << frameGray.type() << endl;
		equalizeHist(frameGray, frameGray);
		//Remove face from matrix
		removeFacesFromMask(BM, frameGray);
	}
	std::vector<Mat> output(2);
	output[0] = I_BGR;
	output[1] = BM;
	imshow("Image", BM);
	waitKey(0);
	return output;
}