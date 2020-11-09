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
    Mat I_BGR;
    Mat I_HSV;
    // create bgr and hsv version of image
    I1.convertTo(I_BGR, CV_32F, 1.0 / 255.0, 0.);
    cvtColor(I_BGR, I_HSV, COLOR_BGR2HSV);

    // add points of interest that we are using to compute color averages. later, for a webcam feed,
    // the hand will have to be positioned over ROIs like this
    roi.push_back(Hand_ROI(Point(645, 150), I_HSV));
    roi.push_back(Hand_ROI(Point(615, 244), I_HSV));
    roi.push_back(Hand_ROI(Point(725, 262), I_HSV));
    roi.push_back(Hand_ROI(Point(623, 333), I_HSV));
    roi.push_back(Hand_ROI(Point(690, 333), I_HSV));
    roi.push_back(Hand_ROI(Point(660, 280), I_HSV));
    roi.push_back(Hand_ROI(Point(620, 300), I_HSV));
    roi.push_back(Hand_ROI(Point(770, 170), I_HSV));
    roi.push_back(Hand_ROI(Point(710, 108), I_HSV));
    roi.push_back(Hand_ROI(Point(662, 319), I_HSV));

    // create binary threshold image per ROI
    vector<Mat> binaries;
    for (Hand_ROI r : roi) {
        //r.draw_rectangle(I_HSV);
        vector<Mat> split_HSV;
        split(I_HSV, split_HSV);
        Mat H_bin, S_bin, V_bin, result;
        threshold(split_HSV[0], H_bin, r.roi_mean[0], 1, THRESH_BINARY_INV);
        threshold(split_HSV[1], S_bin, r.roi_mean[1], 1, THRESH_BINARY);
        threshold(split_HSV[2], V_bin, r.roi_mean[2] - 0.2, 1, THRESH_BINARY);
        binaries.push_back(H_bin & S_bin & V_bin);
    }

    // sum of all binary images
    int count = 0;
    Mat binary_sum = binaries[0];
    for(Mat m : binaries) {
        binary_sum += m;
        //imshow(to_string(count), m);
        count++;
    }

    // blur binary image to smoothen it
    Mat binary_blur(I_HSV.size(), CV_8UC1);
    Mat binary_blur_uc;
    cout << binary_blur.type() << endl;
    medianBlur(binary_sum, binary_blur, 5);
    binary_blur.convertTo(binary_blur_uc, CV_8UC1);
    cout << binary_blur_uc.type() << endl;

    // detect contours
    vector<vector<Point>> all_contours;
    findContours(binary_blur_uc, all_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // for now we only want to keep the biggest one (hand). this might pose problems later, we will see.
    vector<vector<Point>> contours;
    for(vector<Point> c: all_contours) {
        if(contourArea(c) > 40000)
            contours.push_back(c);
    }

    // get hull and defects for our hand contour
    vector<vector<int>> hull_int(contours.size());
    vector<vector<Point>> hull(contours.size());
    vector<vector<Vec4i>> defects(contours.size());
    for(int i = 0; i < contours.size(); i++) {
        //cout << "AREA: " << contourArea(contours[i]) << endl;
        convexHull(contours[i], hull[i], true);
        convexHull(contours[i], hull_int[i], false);
        convexityDefects(contours[i], hull_int[i], defects[i]);
    }

    // draw contour and hull
    drawContours(I_BGR, hull, -1, Scalar(0, 255, 0), 1);
    drawContours(I_BGR, contours, -1, Scalar(255, 0, 0), 1);

    for(vector<Vec4i> v: defects) {
        for(Vec4i d: v) {
            cout << d << " ";
        }
        cout << endl;
    }

    // draw defects
    for(int i = 0; i< contours.size(); i++) {
        int count = contours[i].size();
        if( count <300 )
            continue;

        vector<Vec4i>::iterator d = defects[i].begin();
        while(d != defects[i].end()) {
            Vec4i& v = (*d);
            int startidx = v[0]; Point ptStart(contours[i][startidx]);
            int endidx = v[1]; Point ptEnd(contours[i][endidx]);
            int faridx = v[2]; Point ptFar(contours[i][faridx]);
            float depth = v[3] / 256;

            //line(I_BGR, ptStart, ptEnd, Scalar(0, 0, 255), 1);
            //line(I_BGR, ptStart, ptFar, Scalar(0, 0, 255), 1);
            //line(I_BGR, ptEnd, ptFar, Scalar(0, 0, 255), 1);
            circle(I_BGR, ptFar, 4, Scalar(0, 0, 255), 2);
            d++;
        }
    }

    imshow("Image", I_BGR);
    imshow("Binary Sum", binary_sum);
    imshow("Binary Blur", binary_blur);
    setMouseCallback("Image", CallBackFunc, &I_HSV);
    waitKey(0);

    return 0;
}
