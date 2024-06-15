#include <iostream>
#include <fstream>
#include <sstream>
#include <Kinect.h>
#include <opencv2/opencv.hpp>
#include <windows.h> // Include this for GetAsyncKeyState

// Kinect related variables
IKinectSensor* pSensor = nullptr;
IColorFrameReader* pColorFrameReader = nullptr;
IDepthFrameReader* pDepthFrameReader = nullptr;

// Constants for color and depth frame dimensions
const int COLOR_WIDTH = 1920;
const int COLOR_HEIGHT = 1080;
const int DEPTH_WIDTH = 512;
const int DEPTH_HEIGHT = 424;
int imageCount = 0;

// Depth data array
UINT16 depthData[DEPTH_WIDTH * DEPTH_HEIGHT];

bool InitializeKinect() {
    HRESULT hr = GetDefaultKinectSensor(&pSensor);
    if (FAILED(hr)) {
        std::cerr << "Failed to get default Kinect sensor!" << std::endl;
        return false;
    }

    if (pSensor) {
        hr = pSensor->Open();
        if (FAILED(hr)) {
            std::cerr << "Failed to open Kinect sensor!" << std::endl;
            return false;
        }

        // Initialize color and depth frame readers
        IColorFrameSource* pColorFrameSource = nullptr;
        IDepthFrameSource* pDepthFrameSource = nullptr;

        if (SUCCEEDED(pSensor->get_ColorFrameSource(&pColorFrameSource))) {
            pColorFrameSource->OpenReader(&pColorFrameReader);
            pColorFrameSource->Release();
        }
        else {
            std::cerr << "Failed to get color frame source!" << std::endl;
        }

        if (SUCCEEDED(pSensor->get_DepthFrameSource(&pDepthFrameSource))) {
            pDepthFrameSource->OpenReader(&pDepthFrameReader);
            pDepthFrameSource->Release();
        }
        else {
            std::cerr << "Failed to get depth frame source!" << std::endl;
        }

        return true;
    }

    return false;
}

int main() {
    if (!InitializeKinect()) {
        std::cerr << "Failed to initialize Kinect!" << std::endl;
        return 1;
    }

    cv::Mat colorImage(COLOR_HEIGHT, COLOR_WIDTH, CV_8UC4);
    cv::Mat depthImage(DEPTH_HEIGHT, DEPTH_WIDTH, CV_16UC1);
    cv::Mat coloredDepthImage(DEPTH_HEIGHT, DEPTH_WIDTH, CV_8UC3);
    bool saveData = false;

    while (true) {
        IColorFrame* pColorFrame = nullptr;
        IDepthFrame* pDepthFrame = nullptr;

        if (SUCCEEDED(pColorFrameReader->AcquireLatestFrame(&pColorFrame))) {
            pColorFrame->CopyConvertedFrameDataToArray(COLOR_WIDTH * COLOR_HEIGHT * 4, colorImage.data, ColorImageFormat_Bgra);
            pColorFrame->Release();
        }

        if (SUCCEEDED(pDepthFrameReader->AcquireLatestFrame(&pDepthFrame))) {
            pDepthFrame->CopyFrameDataToArray(DEPTH_WIDTH * DEPTH_HEIGHT, depthData);
            pDepthFrame->Release();
        }

        if (GetAsyncKeyState(VK_SPACE) & 0x8000) {
            saveData = true;
        }

        if (saveData) {
            // JSON saving without external libraries
            std::ofstream jsonFile("depth_data_17mayis" + std::to_string(imageCount) + ".json");
            jsonFile << "[\n";
            for (int y = 0; y < DEPTH_HEIGHT; ++y) {
                jsonFile << "  [";
                for (int x = 0; x < DEPTH_WIDTH; ++x) {
                    jsonFile << depthData[y * DEPTH_WIDTH + x];
                    if (x < DEPTH_WIDTH - 1) jsonFile << ", ";
                }
                jsonFile << "]";
                if (y < DEPTH_HEIGHT - 1) jsonFile << ",";
                jsonFile << "\n";
            }
            jsonFile << "]";
            jsonFile.close();

            // Visualize and reset flag

            // Save RGB image

            std::ostringstream filename_color;
            std::ostringstream filename_depth;
            filename_color << "color_image_17mayis" << imageCount << ".jpg"; // Generate unique file name
            filename_depth << "depth_image_17mayis" << imageCount << ".jpg"; // Generate unique file name

            cv::imwrite(filename_color.str(), colorImage);
            cv::imwrite(filename_depth.str(), coloredDepthImage);
            // Reset flag
            imageCount++;
            saveData = false;
        }

        // Color the depth data for visualization
        for (int y = 0; y < DEPTH_HEIGHT; ++y) {
            for (int x = 0; x < DEPTH_WIDTH; ++x) {
                float normalizedDepth = static_cast<float>(depthData[y * DEPTH_WIDTH + x]) / 2047.0f;
                cv::Vec3b color;
                color[0] = static_cast<uchar>(255 * (1.0f - normalizedDepth)); // Blue
                color[1] = 0; // Green (optional)
                color[2] = static_cast<uchar>(255 * normalizedDepth); // Red
                coloredDepthImage.at<cv::Vec3b>(y, x) = color;
            }
        }

        // Display images
        cv::imshow("Color Image", colorImage);
        cv::imshow("Colored Depth Image", coloredDepthImage);

        if (cv::waitKey(30) == 27) {
            break; // Exit on ESC key
        }
    }

    // Cleanup
    pSensor->Close();
    pSensor->Release();
    cv::destroyAllWindows();

    return 0;
}
