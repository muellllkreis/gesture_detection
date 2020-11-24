#ifndef GESTURE_DETECTOR_H
#define GESTURE_DETECTOR_H

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

using namespace std;

class gesture_detector
{
    public : 
        gesture_detector();
        bool findHandContour(Mat& I, vector<Point>& handContour);
        vector<Point> findFingerTips(vector<Point> handContour, Mat& I);
        vector<Point> filterFalsePositiveFingertips(vector<Point> fingertips, float limitDistance);        
    private :
        vector<Point> neighborhoodAverage(vector<Point> initialPoints, float neighborhoodRadius);        
        vector<Point> findClosestOnX(vector<Point> points, Point pivot);
        bool isFinger(Point a, Point b, Point c, double minAngle, double maxAngle, Point handCentroid, double minDistFromCentroid);
        bool isHand(vector<Point> contour, Rect boundRect);
        int findLargestCont(vector<vector<Point>> contours);
        double findPointsDistanceOnX(Point a, Point b);        
        float distance(Point a, Point b);
        float getAngle(Point s, Point f, Point e);
};

#endif