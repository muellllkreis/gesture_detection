#include <iostream>
#include <sstream>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include "hand_roi.hpp"
using namespace cv;
using namespace std;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
     if  (event == EVENT_LBUTTONDOWN) {
          cout << "Position (" << x << ", " << y << ")" << endl;
     }
}

int main(int argc, char* argv[])
{
    // Section to include webcam feed for later
    /*
    VideoCapture cap;
    if(!cap.open(0))
        return 0;
    for(;;)
    {
          Mat frame;
          cap >> frame;
          if( frame.empty() ) break; // end of video stream
          namedWindow("Feed", CV_WINDOW_KEEPRATIO);
          resizeWindow("Feed", 1000, 800);
          imshow("Feed", frame);
          if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC
    }
    */

    vector<Hand_ROI> roi;

    Mat I1 = imread("hand.jpg");
    cout << I1.type() << endl;
    cout << "Works: " << I1.at<Vec3b>(0, 0) << endl;
    cout << "Size: " << I1.size << endl;
    I1.at<Vec3b>(165, 660) = Vec3b(0.f, 0.f, 0.f);
    I1.at<Vec3b>(166, 660) = Vec3b(0.f, 0.f, 0.f);
    I1.at<Vec3b>(167, 660) = Vec3b(0.f, 0.f, 0.f);
    I1.at<Vec3b>(164, 660) = Vec3b(0.f, 0.f, 0.f);
    Mat I;
    I1.convertTo(I, CV_32F, 1.0 / 255.0, 0.);
    roi.push_back(Hand_ROI(Point(645, 150), I));
    roi.push_back(Hand_ROI(Point(615, 244), I));
    roi.push_back(Hand_ROI(Point(725, 262), I));
    roi.push_back(Hand_ROI(Point(623, 333), I));
    roi.push_back(Hand_ROI(Point(690, 333), I));
    for (Hand_ROI r : roi) {
        r.draw_rectangle(I);  //rectangle(I, r, Scalar(0, 255, 0));
    }

    Mat binaryMat(I.size(), CV_8U);
    cout << roi[0].roi_mean << endl;
    inRange(I, roi[0].roi_mean, Scalar(roi[0].roi_mean[0] - 12, roi[0].roi_mean[1] - 7, roi[0].roi_mean[2] - 10), binaryMat);

    imshow("Image", I);
    imshow("Binary", binaryMat);
    setMouseCallback("Image", CallBackFunc, &I);
    waitKey(0);

    return 0;
}
