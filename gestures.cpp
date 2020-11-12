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
        roi.push_back(Hand_ROI(hand_center, I_HSV));
        roi.push_back(Hand_ROI(Point(hand_center.x + 150, hand_center.y), I_HSV));
        roi.push_back(Hand_ROI(Point(hand_center.x - 150, hand_center.y), I_HSV));
        roi.push_back(Hand_ROI(Point(hand_center.x + 120, hand_center.y + 100), I_HSV));
        roi.push_back(Hand_ROI(Point(hand_center.x - 120, hand_center.y + 100), I_HSV));
        roi.push_back(Hand_ROI(Point(hand_center.x, hand_center.y - 100), I_HSV));
        roi.push_back(Hand_ROI(Point(hand_center.x, hand_center.y - 200), I_HSV));

        for (Hand_ROI r : roi) {
            r.draw_rectangle(I_BGR);
        }

        if(waitKey(10) == 32) {
            break;
        }

        namedWindow("Image", CV_WINDOW_KEEPRATIO);
        resizeWindow("Image", 848, 480);
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
    inRange(I_HSV, Scalar(min_h - 12, min_s - 0.12, 0), Scalar(max_h + 30, max_s + 0.3, 1), binary_mask);

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
    vector<vector<int>> hull_int(contours.size());
    vector<vector<Point>> hull(contours.size());
    vector<vector<Vec4i>> defects(contours.size());
    vector<Rect> boundRect(contours.size());

    for(int i = 0; i < contours.size(); i++) {
        //cout << "AREA: " << contourArea(contours[i]) << endl;
        convexHull(contours[i], hull[i], true);
        convexHull(contours[i], hull_int[i], false);
        convexityDefects(contours[i], hull_int[i], defects[i]);
        boundRect[i] = boundingRect(hull[i]);

    }

    // draw contour and hull
    drawContours(I_BGR, hull, -1, Scalar(0, 255, 0), 1);
    //drawContours(I_BGR, contours, -1, Scalar(255, 0, 0), 1);

//    for(vector<Vec4i> v: defects) {
//        for(Vec4i d: v) {
//            cout << d << " ";
//        }
//        cout << endl;
//    }

//    // we need the bounding box to get rid of defects that we don't need (in the end we only want the space
//    // between fingers
//    rectangle(I_BGR, boundRect[0], Scalar(0, 0, 255), 1);
//    double bounding_width = boundRect[0].width;
//    double bounding_height = boundRect[0].height;
//    vector<Vec4i> newDefects;

//    for(int i = 0; i < contours.size(); i++) {
//        int tolerance =  bounding_height / 5;
//        float angleTol = 95;
//        int startidx, endidx, faridx;
//        vector<Vec4i>::iterator d=defects[i].begin();
//        while(d != defects[i].end()) {
//            Vec4i& v=(*d);
//            startidx=v[0]; Point ptStart(contours[i][startidx]);
//            endidx=v[1]; Point ptEnd(contours[i][endidx]);
//            faridx=v[2]; Point ptFar(contours[i][faridx]);
//            if(distance(ptStart, ptFar) > tolerance && distance(ptEnd, ptFar) > tolerance && getAngle(ptStart, ptFar, ptEnd  ) < angleTol ){
//                if(ptEnd.y > (boundRect[0].y + bounding_height - bounding_height / 4)) {
//                } else if(ptStart.y > (boundRect[0].y + bounding_height - bounding_height / 4)){
//                } else {
//                    newDefects.push_back(v);
//                }
//            }
//            d++;
//        }

//    // draw defects
//        int count = contours[i].size();
//        if( count <300 )
//            continue;

//        for(Vec4i v : newDefects) {
//            int startidx = v[0]; Point ptStart(contours[i][startidx]);
//            int endidx = v[1]; Point ptEnd(contours[i][endidx]);
//            int faridx = v[2]; Point ptFar(contours[i][faridx]);
//            float depth = v[3] / 256;

//            //line(I_BGR, ptStart, ptEnd, Scalar(0, 0, 255), 1);
//            //line(I_BGR, ptStart, ptFar, Scalar(0, 0, 255), 1);
//            //line(I_BGR, ptEnd, ptFar, Scalar(0, 0, 255), 1);
//            circle(I_BGR, ptFar, 4, Scalar(0, 0, 255), 2);
//        }
//    }

//    vector <Point> fingerTips;
//    int i=0;

//    for(Vec4i v : newDefects) {
//        int startidx=v[0]; Point ptStart(contours[0][startidx] );
//        int endidx=v[1]; Point ptEnd(contours[0][endidx] );
//        int faridx=v[2]; Point ptFar(contours[0][faridx] );
//        if(i==0){
//                fingerTips.push_back(ptStart);
//                i++;
//        }
//        fingerTips.push_back(ptEnd);
//        i++;
//    }

//    Point p;
//    int k=0;
//    for(int i=0;i<fingerTips.size();i++){
//            p=fingerTips[i];
//            circle( I_BGR,p,   5, Scalar(100,255,100), 4 );
//     }

    imshow("Image", I_BGR);
    imshow("Binary Sum", binary_mask);
    imshow("Binary Blur", binary_blur);
    setMouseCallback("Image", CallBackFunc, &I_HSV);
    waitKey(0);

    return 0;
}

//    // draw defects

//        int count = contours[i].size();
//        if( count <300 )
//            continue;

//        vector<Vec4i>::iterator d2 = newDefects[i].begin();
//        while(d2 != newDefects[i].end()) {
//            Vec4i& v = (*d2);
//            int startidx = v[0]; Point ptStart(contours[i][startidx]);
//            int endidx = v[1]; Point ptEnd(contours[i][endidx]);
//            int faridx = v[2]; Point ptFar(contours[i][faridx]);
//            float depth = v[3] / 256;

//            //line(I_BGR, ptStart, ptEnd, Scalar(0, 0, 255), 1);
//            //line(I_BGR, ptStart, ptFar, Scalar(0, 0, 255), 1);
//            //line(I_BGR, ptEnd, ptFar, Scalar(0, 0, 255), 1);
//            circle(I_BGR, ptFar, 4, Scalar(0, 0, 255), 2);
//            d++;
//        }
//    }
