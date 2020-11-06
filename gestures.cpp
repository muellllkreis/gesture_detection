#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
using namespace cv;
using namespace std;


int main(int argc, char* argv[])
{
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
    return 0;
}
