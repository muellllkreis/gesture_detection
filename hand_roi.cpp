#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include "hand_roi.hpp"

static int RECT_LEN = 20;

using namespace cv;
using namespace std;

Hand_ROI::Hand_ROI(){
}

Hand_ROI::Hand_ROI(Point upper, Mat src){
    roi_src = src;
    upper_corner = upper;
    lower_corner = Point(upper.x, upper.y - RECT_LEN);
    roi_rect = Rect(this->upper_corner, Size(RECT_LEN, RECT_LEN));
    roi_ptr = src(this->roi_rect);
    roi_mean = calculate_average(src);
}

void Hand_ROI::draw_rectangle(Mat src){
    rectangle(src, roi_rect, Scalar(0,255,0), 2);
}

Scalar Hand_ROI::calculate_average(Mat src) {
    Mat1b mask(this->roi_ptr.cols, this->roi_ptr.rows);
    try {
        //Scalar average = mean(this->roi_ptr, mask);
        vector<float> average(3, 0.f);
        for (int i = 0; i < RECT_LEN; i++) {
            for (int j = 0; j < RECT_LEN; j++) {
                Vec3f px = src.at<Vec3f>(this->upper_corner.y + i, this->upper_corner.x + j);
                //cout << px << endl;
                average[0] += px[0];
                average[1] += px[1];
                average[2] += px[2];
                //src.at<Vec3b>(this->upper_corner.y + i, this->upper_corner.x + j) = Vec3b(0.f, 0.f, 0.f);
            }
        }

/*        cout << "B: " << ((float) average[0] / (RECT_LEN * RECT_LEN)) << endl;
        cout << "G: " << ((float) average[1] / (RECT_LEN * RECT_LEN)) << endl;
        cout << "R: " << ((float) average[2] / (RECT_LEN * RECT_LEN)) << endl*/;

        return Scalar(((float)average[0] / (RECT_LEN * RECT_LEN)), ((float)average[1] / (RECT_LEN * RECT_LEN)), ((float)average[2] / (RECT_LEN * RECT_LEN)));
    }
    catch (cv::Exception & e) {
        cerr << e.what() << endl;
        return 0;
    }
}
