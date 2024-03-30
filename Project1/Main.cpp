// C++ Standard Library
#include <cstdlib>
#include <iostream>
#include <vector>

// OpenCV
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {

    // Load a classifier from its XML description
    cv::CascadeClassifier classifier("D:\\Programming\\C++\\computer-vision-cpp\\data\\haarcascade_upperbody.xml");

    // Prepare a display window
    const char* const window_name{ "Facial Recognition Window" };

    cv::namedWindow(window_name);

    cv::VideoCapture capture("D:\\Programming\\C++\\computer-vision-cpp\\data\\tokyo-walk1.mp4");
    if (not capture.isOpened()) {
        std::cerr << "cannot open video file\n";
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
        //image = cv::imread("D:\\Programming\\C++\\computer-vision-cpp\\data\\adabmp.bmp");
        cv::cvtColor(image, grayscale_image, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(grayscale_image, grayscale_image);
        cv::GaussianBlur(grayscale_image, grayscale_image, cv::Size(5, 5), 0);
        cv::Size targetSize(640, 480); // Or any other standard size
        cv::resize(grayscale_image, grayscale_image, targetSize);

        // detectMultiScale method will clear before adding new features.
        classifier.detectMultiScale(grayscale_image, features, 1.1, 3,
            0 | cv::CASCADE_SCALE_IMAGE, cv::Size(50, 50));

        // Draw each feature as a separate green rectangle
        for (auto&& feature : features) {
            feature.x *= (image.cols / targetSize.width);
            feature.y *= (image.rows / targetSize.height);
            feature.width *= (image.cols / targetSize.width);
            feature.height *= (image.rows / targetSize.height);

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