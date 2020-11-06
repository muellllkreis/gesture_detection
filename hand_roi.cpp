#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include "hand_roi.hpp"

using namespace cv;
using namespace std;

Hand_ROI::Hand_ROI(){
}

Hand_ROI::Hand_ROI(Rect rect, Mat src){
        roi_rect = rect;
        roi_ptr=src(rect);
}

void Hand_ROI::draw_rectangle(Mat src){
        rectangle(src, roi_rect, Scalar(0,255,0), 2);
}
