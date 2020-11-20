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
#define _USE_MATH_DEFINES
#include <math.h>
#include <limits>

using namespace cv;
using namespace std;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
     if  (event == EVENT_LBUTTONDOWN) {
          cout << "Position (" << x << ", " << y << ")" << endl;
     }
}

float distance(Point a, Point b){
        float d = sqrt(fabs(pow(a.x - b.x, 2) + pow(a.y - b.y, 2))) ;
        return d;
}

float getAngle(Point s, Point f, Point e){
        float l1 = distance(f,s);
        float l2 = distance(f,e);
        float dot=(s.x-f.x)*(e.x-f.x) + (s.y-f.y)*(e.y-f.y);
        float angle = acos(dot/(l1*l2));
        angle = angle*180/M_PI;
        return angle;
}

bool isHand(vector<Point> contour, Rect boundRect) {
    double bounding_width = boundRect.width;
    double bounding_height = boundRect.height;
    if(bounding_width == 0 || bounding_height == 0)
        return false;
    else if((bounding_height / bounding_width > 4) || (bounding_width / bounding_height > 4))
        return false;
    else if(boundRect.x < 20)
        return false;
    else
        return true;
}

int findLargestCont(vector<vector<Point>> contours){
    int max_index = -1;
    int max_size = 0;
    for (int i = 0; i < contours.size(); i++){
        if(contours[i].size() > max_size){
            max_size = contours[i].size();
            max_index = i;
        }
    }
    return max_index;
}

bool isFinger(Point a, Point b, Point c, double minAngle, double maxAngle, Point handCentroid, double minDistFromCentroid) {
    float angle = getAngle(a, b, c);
    //threshold angle values
    if (angle > maxAngle || angle < minAngle)
        return false;

    // the finger point should not be under the two far points
    if (b.y - a.y > 0 && b.y - c.y > 0)
        return false;

    // the two far points should not be both under the center of the hand
    if (handCentroid.y - a.y < 0 && handCentroid.y - c.y < 0)
        return false;
    
    if (distance(b, handCentroid) < minDistFromCentroid)
        return false;

    // this should be the case when no fingers are up
    if (distance(a, handCentroid) < minDistFromCentroid / 4 || distance(c, handCentroid) < minDistFromCentroid / 4)
        return false;

    return true;
}

vector<Point> neighborhoodAverage(vector<Point> initialPoints, float neighborhoodRadius)
{
    vector<Point> averagePoints;

    // we start with the first point
    Point reference = initialPoints[0];
    Point median = initialPoints[0];

    for (int i = 1; i < initialPoints.size(); i++) {
        if (distance(reference, initialPoints[i]) > neighborhoodRadius) {

            // the point is not in range, we save the median
            averagePoints.push_back(median);

            // we swap the reference
            reference = initialPoints[i];
            median = initialPoints[i];
        }
        else
            median = (initialPoints[i] + median) / 2;
    }
    // last median
    averagePoints.push_back(median);
    return averagePoints;
}

double findPointsDistanceOnX(Point a, Point b) {
    double to_return = 0.0;

    if (a.x > b.x)
        to_return = a.x - b.x;
    else
        to_return = b.x - a.x;

    return to_return;
}


vector<Point> findClosestOnX(vector<Point> points, Point pivot) {
    vector<Point> to_return(2);

    if (points.size() == 0)
        return to_return;

    double distance_x_1 = DBL_MAX;
    double distance_1 = DBL_MAX;
    double distance_x_2 = DBL_MAX;
    double distance_2 = DBL_MAX;
    int index_found = 0;

    for (int i = 0; i < points.size(); i++) {
        double distance_x = findPointsDistanceOnX(pivot, points[i]);
        double totalDistance = distance(pivot, points[i]);

        if (distance_x < distance_x_1 && distance_x != 0 && totalDistance <= distance_1) {
            distance_x_1 = distance_x;
            distance_1 = totalDistance;
            index_found = i;
        }
    }

    to_return[0] = points[index_found];

    for (int i = 0; i < points.size(); i++) {
        double distance_x = findPointsDistanceOnX(pivot, points[i]);
        double totalDistance = distance(pivot, points[i]);

        if (distance_x < distance_x_2 && distance_x != 0 && totalDistance <= distance_2 && distance_x != distance_x_1) {
            distance_x_2 = distance_x;
            distance_2 = totalDistance;
            index_found = i;
        }
    }

    to_return[1] = points[index_found];

    return to_return;
}


int main(int argc, char* argv[])
{
    String faceClassifierFileName = "face_classifier/haarcascade_frontalface_alt.xml";
    String profileClassifierFileName = "face_classifier/haarcascade_profileface.xml";
    CascadeClassifier faceCascadeClassifier, profileCascadeClassifier;

    if (!faceCascadeClassifier.load(faceClassifierFileName))
        throw runtime_error("can't load file " + faceClassifierFileName);

    if (!profileCascadeClassifier.load(profileClassifierFileName))
        throw runtime_error("can't load file " + profileClassifierFileName);


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

    //Mat I1 = imread("hand.jpg");
    Mat I1;
    Mat I_BGR;
    Mat I_HSV;

    VideoCapture cap;
    if(!cap.open(0)) {
        return 0;
    }

    cap >> I1;
    double col_offset = I1.cols * 0.05;
    double row_offset = I1.rows * 0.05;

    while(true) {
        vector<Hand_ROI>().swap(roi);
        cap >> I1;
        //flip(I1, I1, 1);
        // create bgr and hsv version of image
        I1.convertTo(I_BGR, CV_32F, 1.0 / 255.0, 0.);
        cvtColor(I_BGR, I_HSV, COLOR_BGR2HSV);

        // add points of interest that we are using to compute color averages. later, for a webcam feed,
        // the hand will have to be positioned over ROIs like this
        Point hand_center = Point(3 * (int) (I1.cols / 4), (int) (I1.rows / 2));
        //roi.push_back(Hand_ROI(hand_center, I_HSV));
        roi.push_back(Hand_ROI(Point(hand_center.x + col_offset, hand_center.y-row_offset), I_HSV));
        roi.push_back(Hand_ROI(Point(hand_center.x - col_offset, hand_center.y+row_offset), I_HSV));
        roi.push_back(Hand_ROI(Point(hand_center.x + col_offset, hand_center.y + row_offset), I_HSV));
        roi.push_back(Hand_ROI(Point(hand_center.x - col_offset, hand_center.y - row_offset), I_HSV));
        //roi.push_back(Hand_ROI(Point(hand_center.x, hand_center.y - row_offset), I_HSV));
        //roi.push_back(Hand_ROI(Point(hand_center.x, hand_center.y - row_offset * 2), I_HSV));

        if(waitKey(10) == 32) {
            break;
        }

        for (Hand_ROI r : roi) {
            r.draw_rectangle(I_BGR);
        }

        namedWindow("Image", CV_WINDOW_KEEPRATIO);
        resizeWindow("Image", 960, 540);
        imshow("Image", I_BGR);
    }

    // create binary threshold image per ROI
    float min_h = std::numeric_limits<float>::infinity();
    float max_h = 0.f;
    float min_s = std::numeric_limits<float>::infinity();
    float max_s = 0.f;
    for(Hand_ROI r: roi) {
        if(r.roi_mean[0] < min_h) {
            min_h = r.roi_mean[0];
        }
        if(r.roi_mean[0] > max_h) {
            max_h = r.roi_mean[0];
        }
        if(r.roi_mean[1] < min_s) {
            min_s = r.roi_mean[1];
        }
        if(r.roi_mean[1] > max_s) {
            max_s = r.roi_mean[1];
        }
    }

    cout << "min h: " << min_h << endl;
    cout << "max h: " << max_h << endl;
    cout << "min s: " << min_s << endl;
    cout << "max h: " << max_s << endl;

    Mat binary_mask;
    inRange(I_HSV, Scalar(min_h - 30, min_s - 0.3, 0), Scalar(max_h + 80, max_s + 0.8, 1), binary_mask);

    // blur binary image to smoothen it
    Mat binary_blur(I_HSV.size(), CV_8UC1);
    Mat binary_blur_uc;
    cout << binary_blur.type() << endl;
    medianBlur(binary_mask, binary_blur, 5);
    binary_blur.convertTo(binary_blur_uc, CV_8UC1);
    cout << binary_blur_uc.type() << endl;

    // use openCV classifier to find face(s) and remove them
    // this will enable us to afterwards assume that the biggest remaining part of skin will be the hand
    vector<Rect> faces;
    vector<Rect> profile;
    Mat frameGray;

    cvtColor(I1, frameGray, CV_BGR2GRAY);
    cout << "type: " << frameGray.type() << endl;
    equalizeHist(frameGray, frameGray);

    faceCascadeClassifier.detectMultiScale(frameGray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(120, 120));
    profileCascadeClassifier.detectMultiScale(frameGray, profile, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(120, 120));

    faces.insert(faces.end(), profile.begin(), profile.end());

    for(Rect f: faces) {
        cout << "found face" << endl;
        // note that we are removing the face from binary_blur_uc as it is the Mat used in the next step for contour detection
        rectangle(binary_blur_uc, f, Scalar(0, 0, 0), -1);
    }

    // detect contours
    vector<vector<Point>> all_contours;
    findContours(binary_blur_uc, all_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // for now we only want to keep the biggest one (hand). this might pose problems later, we will see.
    vector<vector<Point>> contours;
    for(int i = 0; i < all_contours.size(); i++) {
        if(contourArea(all_contours[i]) > 40000)
            contours.push_back(all_contours[i]);
    }

    // get hull and defects for our hand contour
    //vector<vector<int>> hull_int(contours.size());
    //vector<vector<Point>> hull(contours.size());
    //vector<vector<Vec4i>> defects(contours.size());
    //vector<Rect> boundRect(contours.size());

    int check = 0;
    bool foundHand = false;
    int handIndex = -1;
    Rect boundRect;
    while(!foundHand && check < contours.size()) {
        handIndex = findLargestCont(contours);
        boundRect = boundingRect(contours[handIndex]);
        if(isHand(contours[handIndex], boundRect)) {
            foundHand = true;
        }
        check++;
    }
    if(!foundHand) {
        cout << "No hand in frame" << endl;
        return -1;
    }

    cout << "AREA: " << contourArea(contours[handIndex]) << endl;
    vector<Point> hull;
    convexHull(contours[handIndex], hull, true);
    vector<int> hull_int;
    convexHull(contours[handIndex], hull_int, false);
    vector<Vec4i> defects;
    convexityDefects(contours[handIndex], hull_int, defects);
    //Rect boundRect = boundingRect(hull);

//    // draw contour and hull
    drawContours(I_BGR, vector<vector<Point>>(1, hull), -1, Scalar(0, 255, 0), 1);
//    //drawContours(I_BGR, contours, -1, Scalar(255, 0, 0), 1);

    Rect boundingRectangle = boundingRect(Mat(hull));
    // take the center of the rectangle which is approximately the center of the hand (tl = top left etc)
    Point boundingRectangeCenter((boundingRectangle.tl().x + boundingRectangle.br().x) / 2, (boundingRectangle.tl().y + boundingRectangle.br().y) / 2);

    //points corresponding to bigger and smaller distances
    vector<Point> startPoints;
    vector<Point> farPoints;

    for (int i = 0; i < defects.size(); i++) {
        startPoints.push_back(contours[handIndex][defects[i].val[0]]);

        // filtering the far point based on the distance from the center of the bounding rectangle
        if (distance(contours[handIndex][defects[i].val[2]], boundingRectangeCenter) < boundingRectangle.height * 0.3)
            farPoints.push_back(contours[handIndex][defects[i].val[2]]);
    }

    //we want only one point in a given neighbourhood of points --> filter the points
    vector<Point> filteredStartPoints = neighborhoodAverage(startPoints, boundingRectangle.height * 0.05);
    vector<Point> filteredFarPoints = neighborhoodAverage(farPoints, boundingRectangle.height * 0.05);

    vector<Point> filteredFingerPoints;
    vector<Point> fingerPoints;

    //test if the remaining filtered points are actually fingertips
    for (int i = 0; i < filteredStartPoints.size(); i++)
    {
        vector<Point> closestPoints = findClosestOnX(filteredFarPoints, filteredStartPoints[i]);
        if (isFinger(closestPoints[0], filteredStartPoints[i], closestPoints[1], 5, 50, boundingRectangeCenter, boundingRectangle.height * 0.3))
        {
            fingerPoints.push_back(filteredStartPoints[i]);
        }
    }

    if (fingerPoints.size() > 0)
    {
        while (fingerPoints.size() > 5) //remove potential 6,7 fingers occurences 
        {
            fingerPoints.pop_back(); 
        }

        //filter out points too close to each other
        for (int i = 0; i < fingerPoints.size() - 1; i++)
        {
            if (findPointsDistanceOnX(fingerPoints[i], fingerPoints[i + 1]) > boundingRectangle.height * 0.3 * 1.5)
            {
                filteredFingerPoints.push_back(fingerPoints[i]);
            }
        }
        
        if (fingerPoints.size() > 2)
        {
            if (findPointsDistanceOnX(fingerPoints[0], fingerPoints[fingerPoints.size() - 1]) > boundingRectangle.height * 0.3 * 1.5)
            {
                filteredFingerPoints.push_back(fingerPoints[fingerPoints.size() - 1]);
            }
        }
        else
        {
            filteredFingerPoints.push_back(fingerPoints[fingerPoints.size() - 1]);
        }
    }

    // we need the bounding box to get rid of defects that we don't need (in the end we only want the space
    // between fingers
    //rectangle(I_BGR, boundRect, Scalar(0, 0, 255), 1);
    //double bounding_width = boundRect.width;
    //double bounding_height = boundRect.height;

    //vector<Vec4i> newDefects;

    //int tolerance =  bounding_height / 5;
    //float angleTol = 95;
    //int startidx, endidx, faridx;

    //might not be needed with new method
    //vector<Vec4i>::iterator d = defects.begin();
    //while(d != defects.end()) {
    //    Vec4i& v = (*d);
    //    startidx = v[0]; Point ptStart(contours[handIndex][startidx]);
    //    endidx = v[1]; Point ptEnd(contours[handIndex][endidx]);
    //    faridx = v[2]; Point ptFar(contours[handIndex][faridx]);
    //    if(distance(ptStart, ptFar) > tolerance && distance(ptEnd, ptFar) > tolerance && getAngle(ptStart, ptFar, ptEnd) < angleTol){
    //        if(ptEnd.y > (boundRect.y + bounding_height - bounding_height / 4)) {
    //        } else if(ptStart.y > (boundRect.y + bounding_height - bounding_height / 4)){
    //        } else {
    //            newDefects.push_back(v);
    //        }
    //    }
    //    d++;
    //}

    //// draw defects
    //int count = contours[handIndex].size();
    //if(count < 300)
    //    return -1;

    //for(Vec4i v : newDefects) {
    //    int startidx = v[0]; Point ptStart(contours[handIndex][startidx]);
    //    int endidx = v[1]; Point ptEnd(contours[handIndex][endidx]);
    //    int faridx = v[2]; Point ptFar(contours[handIndex][faridx]);
    //    float depth = v[3] / 256;

    //    //line(I_BGR, ptStart, ptEnd, Scalar(0, 0, 255), 1);
    //    //line(I_BGR, ptStart, ptFar, Scalar(0, 0, 255), 1);
    //    //line(I_BGR, ptEnd, ptFar, Scalar(0, 0, 255), 1);
    //    circle(I_BGR, ptFar, 4, Scalar(0, 0, 255), 2);
    //}

    //vector <Point> fingerTips;
    //int i = 0;

    //for(Vec4i v : newDefects) {
    //    int startidx=v[0]; Point ptStart(contours[handIndex][startidx]);
    //    int endidx=v[1]; Point ptEnd(contours[handIndex][endidx]);
    //    int faridx=v[2]; Point ptFar(contours[handIndex][faridx]);
    //    if(i == 0){
    //            fingerTips.push_back(ptStart);
    //            i++;
    //    }
    //    fingerTips.push_back(ptEnd);
    //    i++;
    //}

    //Point center_bounding_rect(
    //        (boundRect.tl().x + boundRect.br().x) / 2,
    //        (boundRect.tl().y + boundRect.br().y) / 2
    //);

    circle(I_BGR, boundingRectangeCenter, 5, Scalar(255, 0, 0), 4);

    Point p;
    int k=0;
    for(int i = 0; i < fingerPoints.size(); i++){
            p = fingerPoints[i];
            circle(I_BGR, p, 5, Scalar(100,255,100), 4);
     }

    imshow("Image", I_BGR);
    imshow("Binary Sum", binary_mask);
    imshow("Binary Blur", binary_blur);
    setMouseCallback("Image", CallBackFunc, &I_BGR);
    waitKey(0);

    return 0;
}
