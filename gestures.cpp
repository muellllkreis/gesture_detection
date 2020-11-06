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

    Mat I = imread("hand.jpg");
    roi.push_back(Hand_ROI(Rect(645,150,30,30), I));
    roi.push_back(Hand_ROI(Rect(615,244,30,30), I));
    roi.push_back(Hand_ROI(Rect(725,262,30,30), I));
    roi.push_back(Hand_ROI(Rect(623,333,30,30), I));
    roi.push_back(Hand_ROI(Rect(690,333,30,30), I));
    for(Hand_ROI r : roi) {
        r.draw_rectangle(I);  //rectangle(I, r, Scalar(0, 255, 0));
    }

    imshow("Image", I);
    setMouseCallback("Image", CallBackFunc, NULL);
    waitKey(0);

    return 0;
}
