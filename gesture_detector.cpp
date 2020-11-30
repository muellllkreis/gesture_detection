#include "gesture_detector.h"

using namespace std;

//CONSTRUCTOR
gesture_detector::gesture_detector()
{
    
}

//MAIN METHODS

bool gesture_detector::findHandContour(Mat& binary_blur_uc, vector<Point>& handContour)
{
      //Detect hand contours
    vector<vector<Point>> all_contours;
    findContours(binary_blur_uc, all_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // for now we only want to keep the biggest one (hand). this might pose problems later, we will see.
    vector<vector<Point>> contours;
    for (int i = 0; i < all_contours.size(); i++) {
        if (contourArea(all_contours[i]) > 20000)
            contours.push_back(all_contours[i]);
    }

    int check = 0;
    bool foundHand = false;
    int handIndex = -1;
    Rect boundRect;
    while (!foundHand && check < contours.size()) 
    {
        handIndex = findLargestCont(contours);
        boundRect = boundingRect(contours[handIndex]);
        if (isHand(contours[handIndex], boundRect)) 
        {
            foundHand = true;
        }
        check++;
    }
    if (!foundHand) 
    {
        std::cout << "No hand in frame" << std::endl;
        return false;
    }
    else
    {
        handContour = contours[handIndex];
        std::cout << "hand found" << contours.size()<< std::endl;
        std::cout << "hand found" << contours[handIndex].size()<< std::endl;
        return true;
    }    
}

vector<Point> gesture_detector::findFingerTips(vector<Point> handContour, Mat& I_BGR)
{    
    std::cout << "hand found" << handContour.size()<< std::endl;
    vector<Point> hull;
    convexHull(handContour, hull, true);
    vector<int> hull_int;
    convexHull(handContour, hull_int, false);
    vector<Vec4i> defects;
    convexityDefects(handContour, hull_int, defects);    

    // draw contour and hull
    drawContours(I_BGR, vector<vector<Point>>(1, hull), -1, Scalar(0, 255, 0), 1);    
    Rect boundingRectangle = boundingRect(Mat(hull));

    // take the center of the rectangle which is approximately the center of the hand (tl = top left etc)
    Point boundingRectangleCenter((boundingRectangle.tl().x + boundingRectangle.br().x) / 2, (boundingRectangle.tl().y + boundingRectangle.br().y) / 2);
    circle(I_BGR, boundingRectangleCenter, 5, Scalar(255, 0, 0), 4);

    //points corresponding to bigger and smaller distances
    vector<Point> startPoints;
    vector<Point> farPoints;

    for (int i = 0; i < defects.size(); i++) 
    {
        startPoints.push_back(handContour[defects[i].val[0]]);

        // filtering the far point based on the distance from the center of the bounding rectangle
        if (distance(handContour[defects[i].val[2]], boundingRectangleCenter) < boundingRectangle.height * limitDistanceRatio)
            farPoints.push_back(handContour[defects[i].val[2]]);
    }


    //we want only one point in a given neighbourhood of points --> filter the points
    vector<Point> filteredStartPoints = neighborhoodAverage(startPoints, boundingRectangle.height * neighboorhoudSize);
    vector<Point> filteredFarPoints = neighborhoodAverage(farPoints, boundingRectangle.height * neighboorhoudSize);


    vector<Point> fingerPoints;
    //test if the remaining filtered points are actually fingertips
    for(int i = 0; i < filteredStartPoints.size(); i++)
    {
        vector<Point> closestPoints = findClosestOnX(filteredFarPoints, filteredStartPoints[i]);
        if (isFinger(closestPoints[0], filteredStartPoints[i], closestPoints[1], 5, 50, boundingRectangleCenter, boundingRectangle.height * limitDistanceRatio))
        {
            fingerPoints.push_back(filteredStartPoints[i]);
        }
    }

    vector<Point> filteredFingerPoints = neighborhoodAverage(fingerPoints, boundingRectangle.height * neighboorhoudSize*5);

    float minDistance = limitFingertipDistanceRatio * boundingRectangle.height;
    filterFalsePositiveFingertips(filteredFingerPoints, minDistance);

    return filteredFingerPoints;
}

//TEST RESULTS
void gesture_detector::filterFalsePositiveFingertips(vector<Point>& fingerPoints, float limitDistance)
{
    vector<Point> filteredFingerPoints;
    if (fingerPoints.size() > 0)
    {        
        //filter out points too close to each other
        //int lastIndex = ;
        for (int i = 0; i < fingerPoints.size() - 1; i++)
        {
            if (distance(fingerPoints[i], fingerPoints[i + 1]) < limitDistance)
            {
                fingerPoints.erase(fingerPoints.begin()+i+1);                
                //lastIndex--;
            }
        }

        //while (filteredFingerPoints.size() > 5) //remove potential 6,7 fingers occurences 
        //{
        //    filteredFingerPoints.pop_back();
        //}

        //if (fingerPoints.size() > 2)
        //{
        //    if (findPointsDistanceOnX(fingerPoints[0], fingerPoints[fingerPoints.size() - 1]) > limitDistance)
        //    {
        //        filteredFingerPoints.push_back(fingerPoints[fingerPoints.size() - 1]);
        //    }
        //}
        //else
        //{
        //    filteredFingerPoints.push_back(fingerPoints[fingerPoints.size() - 1]);
        //}
    }
}


bool gesture_detector::isFinger(Point a, Point b, Point c, double minAngle, double maxAngle, Point handCentroid, double minDistFromCentroid) {
    float angle = getAngle(a, b, c);
    ////threshold angle values
    //if (angle > maxAngle || angle < minAngle)
    //    return false;

    // the finger point should not be under the two far points
    if (b.y - a.y > 0 && b.y - c.y > 0)
        return false;

    // the two far points should not be both under the center of the hand
    if (handCentroid.y - a.y < 0 && handCentroid.y - c.y < 0)
        return false;
    
    if (distance(b, handCentroid) < minDistFromCentroid)
        return false;

    // this should be the case when no fingers are up (hand uspide down)
    if (distance(a, handCentroid) < minDistFromCentroid / 4 || distance(c, handCentroid) < minDistFromCentroid / 4)
        return false;

    return true;
}


//GEOMETRY METHODS
vector<Point> gesture_detector::neighborhoodAverage(vector<Point> initialPoints, float neighborhoodRadius)
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

int gesture_detector::findLargestCont(vector<vector<Point>> contours){
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

vector<Point> gesture_detector::findClosestOnX(vector<Point> points, Point pivot) 
{
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


double gesture_detector::findPointsDistanceOnX(Point a, Point b) {
    double to_return = 0.0;

    if (a.x > b.x)
        to_return = a.x - b.x;
    else
        to_return = b.x - a.x;

    return to_return;
}

float gesture_detector::distance(Point a, Point b){
        float d = sqrt(fabs(pow(a.x - b.x, 2) + pow(a.y - b.y, 2))) ;
        return d;
}

float gesture_detector::getAngle(Point s, Point f, Point e){
        float l1 = distance(f,s);
        float l2 = distance(f,e);
        float dot=(s.x-f.x)*(e.x-f.x) + (s.y-f.y)*(e.y-f.y);
        float angle = acos(dot/(l1*l2));
        angle = angle*180/M_PI;
        return angle;
}

bool gesture_detector::isHand(vector<Point> contour, Rect boundRect) {
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