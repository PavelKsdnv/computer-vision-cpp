// -*- coding:utf-8; mode:c++; mode:auto-fill; fill-column:80; -*-

/// @file      cascade-classifier.cpp
/// @brief     OpenCV object recognition example.
/// @author    J. Arrieta <juan.arrieta@nablazerolabs.com>
/// @date      October 04, 2017
/// @copyright (c) 2017 Nabla Zero Labs
/// @license   MIT License.
///
/// I wrote this example program for my later reference.
///
/// Compilation:
///
///     clang++ cascade-classifier.cpp -o cascade-classifier \
///     -std=c++1z -Wall -Wextra -Ofast -march=native \
///     -lopencv_objdetect -lopencv_highgui \
///     -lopencv_imgproc -lopencv_core -lopencv_videoio
///
/// The Haar cascade XML description is provided as a command-line argument; the
/// examples I used are in GitHub:
///
///     https://github.com/opencv/opencv/tree/master/data/haarcascades
///

// C++ Standard Library
#include <cstdlib>
#include <iostream>
#include <vector>

// OpenCV
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {

    // Load a classifier from its XML description
    cv::CascadeClassifier classifier("D:\\Programming\\C++\\computer-vision-cpp\\data\\haarcascade_frontalface_default.xml");

    // Prepare a display window
    const char* const window_name{ "Facial Recognition Window" };

    cv::namedWindow(window_name);

    // Prepare a video capture device
    cv::VideoCapture capture(0); // `0` means "default video capture"
    if (not capture.isOpened()) {
        std::cerr << "cannot open video capture device\n";
        std::exit(EXIT_FAILURE);
    }

    // Prepare an image where to store the video frames, and an image to store a
    // grayscale version
    cv::Mat image;
    cv::Mat grayscale_image;

    // Prepare a vector where the detected features will be stored
    std::vector<cv::Rect> features;

    // Main loop
    while (capture.read(image) && (!image.empty())) {
        // Create a normalized, gray-scale version of the captured image
        cv::cvtColor(image, grayscale_image, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(grayscale_image, grayscale_image);

        // Detect the features in the normalized, gray-scale version of the
        // image. You don't need to clear the previously-found features because the
        // detectMultiScale method will clear before adding new features.
        classifier.detectMultiScale(grayscale_image, features, 1.1, 2,
            0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

        // Draw each feature as a separate green rectangle
        for (auto&& feature : features) {
            cv::rectangle(image, feature, cv::Scalar(0, 255, 0), 2);
        }

        // Show the captured image and the detected features
        cv::imshow(window_name, image);

        // Wait for input or process the next frame
        switch (cv::waitKey(10)) {
        case 'q':
            std::exit(EXIT_SUCCESS);
        case 'Q':
            std::exit(EXIT_SUCCESS);
        default:
            break;
        }
    }
    return EXIT_SUCCESS;
}