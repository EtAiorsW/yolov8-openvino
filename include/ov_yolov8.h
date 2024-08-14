#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <openvino.hpp>

const std::vector<std::string> class_names = { "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
            "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush" };
const int shape = 640;
const int outputLength = 8400;
const int class_num = 80;
//const std::vector<std::string> class_names = { "bubble", "impurity" };
//const int shape = 1920;
//const int outputLength = 136000;
//const int class_num = 2;


struct DetResult
{

    cv::Rect bbox;
    float conf;
    int label;
    DetResult(cv::Rect bbox, float conf, int lable) :bbox(bbox), conf(conf), label(lable) {}
};

cv::Mat draw_bbox(cv::Mat& img, std::vector<DetResult>& res);

void pre_process(cv::Mat* img, int length, float* factor, std::vector<float>& data);

std::vector<DetResult> post_process(float* result, float factor);

void warmup(ov::InferRequest request);

void yolov8_async_infer();
