#include <iostream>
#include <fstream>
#include <sstream>
#include <Kinect.h>
#include <opencv2/opencv.hpp>
#include <windows.h> // Include this for GetAsyncKeyState
#include <thread> // Include this for threading
#include <cstdlib> // Include this for system function

// Kinect related variables
IKinectSensor* pSensor = nullptr;
IColorFrameReader* pColorFrameReader = nullptr;
IDepthFrameReader* pDepthFrameReader = nullptr;
ICoordinateMapper* pCoordinateMapper = nullptr;

// Constants for color and depth frame dimensions
const int COLOR_WIDTH = 1920;
const int COLOR_HEIGHT = 1080;
const int DEPTH_WIDTH = 512;
const int DEPTH_HEIGHT = 424;
int imageCount = 0;

// Depth data array
UINT16 depthData[DEPTH_WIDTH * DEPTH_HEIGHT];
UINT16 alignedDepthData[COLOR_WIDTH * COLOR_HEIGHT];

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

        // Initialize coordinate mapper
        hr = pSensor->get_CoordinateMapper(&pCoordinateMapper);
        if (FAILED(hr)) {
            std::cerr << "Failed to get coordinate mapper!" << std::endl;
            return false;
        }

        return true;
    }

    return false;
}

void AlignDepthToColor(UINT16* depthData, UINT16* alignedDepthData) {
    DepthSpacePoint* depthSpacePoints = new DepthSpacePoint[COLOR_WIDTH * COLOR_HEIGHT];

    // Map color frame to depth space
    HRESULT hr = pCoordinateMapper->MapColorFrameToDepthSpace(DEPTH_WIDTH * DEPTH_HEIGHT, depthData, COLOR_WIDTH * COLOR_HEIGHT, depthSpacePoints);
    if (SUCCEEDED(hr)) {
        for (int y = 0; y < COLOR_HEIGHT; ++y) {
            for (int x = 0; x < COLOR_WIDTH; ++x) {
                DepthSpacePoint point = depthSpacePoints[y * COLOR_WIDTH + x];
                int depthX = static_cast<int>(point.X + 0.5f);
                int depthY = static_cast<int>(point.Y + 0.5f);
                if ((depthX >= 0 && depthX < DEPTH_WIDTH) && (depthY >= 0 && depthY < DEPTH_HEIGHT)) {
                    alignedDepthData[y * COLOR_WIDTH + x] = depthData[depthY * DEPTH_WIDTH + depthX];
                }
                else {
                    alignedDepthData[y * COLOR_WIDTH + x] = 0;
                }
            }
        }
    }

    delete[] depthSpacePoints;
}

void RunPythonScript() {
    // Activate the virtual environment and run the Python script
    //system("D:\\cse496-graduation-project\\cse-496\\cse496\\Scripts\\activate && python inference.py");
    system("D:\\cse496-graduation-project\\cse-496\\cse496\\Scripts\\activate && python inference2_calculation.py");
}

int main() {
    if (!InitializeKinect()) {
        std::cerr << "Failed to initialize Kinect!" << std::endl;
        return 1;
    }

    cv::Mat colorImage(COLOR_HEIGHT, COLOR_WIDTH, CV_8UC4);
    cv::Mat depthImage(COLOR_HEIGHT, COLOR_WIDTH, CV_16UC1); // Adjusted to match color image dimensions
    cv::Mat coloredDepthImage(COLOR_HEIGHT, COLOR_WIDTH, CV_8UC3);
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

            // Align depth data to color space
            AlignDepthToColor(depthData, alignedDepthData);
        }

        if (GetAsyncKeyState(VK_SPACE) & 0x8000) {
            saveData = true;
        }

        if (saveData) {
            // JSON saving without external libraries
            std::ofstream jsonFile("depth_data_17haz.json");
            jsonFile << "[\n";
            for (int y = 0; y < COLOR_HEIGHT; ++y) {
                jsonFile << "  [";
                for (int x = 0; x < COLOR_WIDTH; ++x) {
                    jsonFile << alignedDepthData[y * COLOR_WIDTH + x];
                    if (x < COLOR_WIDTH - 1) jsonFile << ", ";
                }
                jsonFile << "]";
                if (y < COLOR_HEIGHT - 1) jsonFile << ",";
                jsonFile << "\n";
            }
            jsonFile << "]";
            jsonFile.close();

            // Save RGB image
            std::ostringstream filename_color;
            std::ostringstream filename_depth;
            filename_color << "color_image_17haz.jpg"; // Generate unique file name
            filename_depth << "depth_image_17haz.jpg"; // Generate unique file name

            cv::imwrite(filename_color.str(), colorImage);
            cv::imwrite(filename_depth.str(), coloredDepthImage);
            // Reset flag
            imageCount++;
            saveData = false;

            // Start a new thread to run the Python script
            std::thread pythonThread(RunPythonScript);
            pythonThread.detach(); // Detach the thread to run independently
        }

        // Color the depth data for visualization
        for (int y = 0; y < COLOR_HEIGHT; ++y) {
            for (int x = 0; x < COLOR_WIDTH; ++x) {
                float normalizedDepth = static_cast<float>(alignedDepthData[y * COLOR_WIDTH + x]) / 2047.0f;
                cv::Vec3b color;
                color[0] = static_cast<uchar>(255 * (1.0f - normalizedDepth)); // Blue
                color[1] = 0; // Green (optional)
                color[2] = static_cast<uchar>(255 * normalizedDepth); // Red
                coloredDepthImage.at<cv::Vec3b>(y, x) = color;
            }
        }

        //// Add a rectangle to the same coordinates in both images
        //cv::Rect rect(500, 300, 200, 200);
        //cv::rectangle(colorImage, rect, cv::Scalar(0, 255, 0), 2);
        //cv::rectangle(coloredDepthImage, rect, cv::Scalar(0, 255, 0), 2);

        // Display images
        cv::imshow("Color Image", colorImage);
        //cv::imshow("Colored Depth Image", coloredDepthImage);

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
