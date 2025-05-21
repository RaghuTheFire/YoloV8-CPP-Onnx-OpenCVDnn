// Multi-threaded YOLOv5/YOLOv8 with TBB queues and OpenVINO/OpenCL/CUDA support + RTSP input + exception handling + alpha blended bounding box

/***********************************************************************************************************************************************
  To compile your updated multi-threaded YOLOv5/YOLOv8 C++ application with OpenCV, TBB, and FFMPEG support, use the following command:
  g++ -std=c++17 -O2 -o yolov_app yolov5_multithread_tbb.cpp `pkg-config --cflags --libs opencv4 tbb`
  g++ -std=c++17 -O2 yolov5_multithread_tbb.cpp -o yolov_app -I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_dnn -ltbb
***********************************************************************************************************************************************/

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <tbb/concurrent_queue.h>
#include <thread>
#include <atomic>
#include <iostream>
#include <filesystem>
#include <exception>

using namespace cv;
using namespace dnn;
using namespace std;
using namespace tbb;
namespace fs = std::filesystem;

concurrent_bounded_queue<Mat> frame_queue;
concurrent_bounded_queue<Mat> processed_queue;
atomic<bool> keep_running{true};

struct Net_config 
{
    float confThreshold;
    float nmsThreshold;
    float objThreshold;
    string modelpath;
    string backend; // "cuda", "cpu", "openvino", "opencl"
};

class YOLO 
{
public:
    YOLO(Net_config config);
    void detect(Mat& frame);

private:
    Net net;
    float confThreshold;
    float nmsThreshold;
    float objThreshold;
    int inpWidth = 640;
    int inpHeight = 640;
    vector<string> class_names;
    string version;

    void drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid);
    void autoDetectVersion();
};

YOLO::YOLO(Net_config config) 
{
    if (!fs::exists(config.modelpath)) 
    {
        cerr << "[Error] Model file not found: " << config.modelpath << endl;
        exit(EXIT_FAILURE);
    }

    string classFile = "classes.txt";
    if (!fs::exists(classFile)) 
    {
        cerr << "[Error] Class label file not found: " << classFile << endl;
        exit(EXIT_FAILURE);
    }

    confThreshold = config.confThreshold;
    nmsThreshold = config.nmsThreshold;
    objThreshold = config.objThreshold;
    net = readNet(config.modelpath);

    if (config.backend == "cuda")
    {
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
    } 
    else 
    if (config.backend == "openvino") 
    {
        net.setPreferableBackend(DNN_BACKEND_INFERENCE_ENGINE);
        net.setPreferableTarget(DNN_TARGET_CPU);
    } 
    else 
    if (config.backend == "opencl") 
    {
        net.setPreferableBackend(DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(DNN_TARGET_OPENCL);
    } 
    else 
    {
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }

    ifstream ifs(classFile);
    string line;
    while (getline(ifs, line)) class_names.push_back(line);
    if (class_names.empty()) 
    {
        cerr << "[Warning] No class labels found in " << classFile << endl;
    }

    autoDetectVersion();
    cout << "Detected YOLO version: " << version << endl;
}

void YOLO::autoDetectVersion() 
{
    Mat dummy = Mat::zeros(inpHeight, inpWidth, CV_8UC3);
    Mat blob = blobFromImage(dummy, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    vector<Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    if (!outs.empty())
    {
        int cols = outs[0].cols;
        version = (cols > 6 && cols < 84) ? "v8" : "v5";
    } 
    else 
    {
        version = "v5";
        cerr << "[Warning] Could not determine model output shape. Falling back to YOLOv5 logic." << endl;
    }
}

void YOLO::drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid) 
{
    // Create alpha blended rectangle
    float alpha = 0.4f;
    Scalar color(0, 255, 0); // Green with alpha
    Mat overlay;
    frame.copyTo(overlay);
    rectangle(overlay, Point(left, top), Point(right, bottom), color, FILLED);
    addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame);

    // Draw border
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 2);

    // Draw label
    string label = class_names[classid] + ": " + format("%.2f", conf);
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height - 4), Point(left + labelSize.width + 4, top + baseLine), Scalar(0, 255, 0), FILLED);
    putText(frame, label, Point(left + 2, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
}

void YOLO::detect(Mat& frame) 
{
    try 
    {
        Mat blob = blobFromImage(frame, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
        net.setInput(blob);
        vector<Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;

        for (int i = 0; i < outs.size(); ++i) 
        {
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) 
            {
                int classOffset = (version == "v8") ? 4 : 5;
                Mat scores = outs[i].row(j).colRange(classOffset, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }

        vector<int> indices;
        NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
        for (int idx : indices) 
        {
            Rect box = boxes[idx];
            drawPred(confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame, classIds[idx]);
        }
    } 
    catch (const std::exception& e) 
    {
        cerr << "[Exception] Error during detection: " << e.what() << endl;
    }
}

void capture_thread(VideoCapture& cap) 
{
    while (keep_running) 
    {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        frame_queue.push(frame);
    }
    keep_running = false;
}

void detection_thread(YOLO& model) 
{
    while (keep_running) 
    {
        Mat frame;
        if (frame_queue.try_pop(frame)) 
        {
            model.detect(frame);
            processed_queue.push(frame);
        }
    }
}

void display_thread() 
{
    while (keep_running) {
        Mat frame;
        if (processed_queue.try_pop(frame)) 
        {
            imshow("YOLO Detection", frame);
            if (waitKey(1) == 27) 
            {
                keep_running = false;
            }
        }
    }
}

int main() 
{
    Net_config config = {0.3, 0.5, 0.3, "best_640.onnx", "opencl"};
    YOLO model(config);

    string rtsp_url = "rtsp://username:password@192.168.1.100:554/stream"; // Replace with actual RTSP URL
    VideoCapture cap(rtsp_url, cv::CAP_FFMPEG);
    if (!cap.isOpened()) 
    {
        cerr << "Error opening RTSP stream." << endl;
        return -1;
    }

    frame_queue.set_capacity(10);
    processed_queue.set_capacity(10);

    thread t1(capture_thread, ref(cap));
    thread t2(detection_thread, ref(model));
    thread t3(display_thread);

    t1.join();
    t2.join();
    t3.join();

    cap.release();
    destroyAllWindows();
    return 0;
}
