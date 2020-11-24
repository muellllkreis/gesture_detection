CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(CMAKE_BUILD_TYPE Release)
SET(OpenCV_STATIC OFF)
FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(GESTURE_TRACKER)

# ADD_EXECUTABLE(sample sample.cpp)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
ADD_EXECUTABLE(gestures_recognition main.cpp hand_roi.cpp binary_mask_creator.cpp gesture_detector.cpp)
TARGET_LINK_LIBRARIES(gestures_recognition ${OpenCV_LIBS})
