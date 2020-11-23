#ifndef HAND_ROI_H
#define HAND_ROI_H


#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>

using namespace cv;

class Hand_ROI{
        public:
            Hand_ROI();
            Hand_ROI(Point upper, Mat src);
            Rect roi_rect;
            Point upper_corner;
            Point lower_corner;
            Mat roi_ptr;
            Mat roi_src;
            Scalar roi_mean;
            void draw_rectangle(Mat src);
            Scalar calculate_average(Mat src);
};

#endif // !HAND_ROI