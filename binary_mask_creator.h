#ifndef BINARY_MASK_CREATOR_H
#define BINARY_MASK_CREATOR_H

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
#include "hand_roi.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <limits>

class binary_mask_creator
{
    public : 
        binary_mask_creator();
        Mat computeBinaryMask(std::vector<Hand_ROI> roi, Mat I);
        std::vector<Mat> createBinaryMask(VideoCapture& cap, bool removeFace);
        Mat removeFacesFromMask(Mat& binary_blur_uc, Mat& I1);
        Mat removeBackGround(Mat input);
        void calibrateBackground(Mat frame);
    private :
        float thresholdCorrectionHigh = 60;
        float thresholdCorrectionLow = 20;
        float backGroundThresholdOffset = 12;
        bool calibrated = false;
        Mat backgroundReference;
};

#endif
