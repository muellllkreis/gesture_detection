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
#define _USE_MATH_DEFINES
#include <math.h>
#include <limits>

#include "hand_roi.h"
#include "binary_mask_creator.h"
#include "gesture_detector.h"


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
    //start video feed
    VideoCapture cap;
    if (!cap.open(0)) {
        return 0;
    }

    //Create binary Mask (+ remove face)
    binary_mask_creator BMC;
    vector<Mat> BMC_output = BMC.createBinaryMask(cap, true);
    Mat I_BGR = BMC_output[0];
    Mat binary_mask = BMC_output[1];

    //Get Hand Contour & Count fingers
    gesture_detector GD;
    vector<Point> handContour;
    bool handFound = GD.findHandContour(binary_mask, handContour);    
    if (handFound)
    {
        //find fingertips
        vector<Point>fingertips;
        fingertips = GD.findFingerTips(handContour, I_BGR);
        //showing results
        if (fingertips.size() > 0)
        {
            Point p;            
            for (int i = 0; i < fingertips.size(); i++) {
                p = fingertips[i];
                circle(I_BGR, p, 5, Scalar(100, 255, 100), 4);
            }
        }
        imshow("Image", I_BGR);
        imshow("Binary Blur", binary_mask);
        setMouseCallback("Image", CallBackFunc, &I_BGR);
        waitKey(0);
    }
    return 0;
}
