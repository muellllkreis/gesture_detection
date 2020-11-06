
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>

using namespace cv;

class Hand_ROI{
        public:
                Hand_ROI();
                Hand_ROI(Rect rect, Mat src);
                Rect roi_rect;
                Mat roi_ptr;
                void draw_rectangle(Mat src);
};
