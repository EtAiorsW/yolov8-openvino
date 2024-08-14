#include <iostream>
#include <iomanip>
#include <filesystem>
#include "ov_yolov8.h"



cv::Mat draw_bbox(cv::Mat& img, std::vector<DetResult>& res)
{
	for (auto& result : res)
	{
		cv::rectangle(img, result.bbox, cv::Scalar(0, 255, 0), 2);
		cv::putText(img, class_names[result.label], cv::Point(result.bbox.x, result.bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));
	}
	return img;
}


void pre_process(cv::Mat* img, int length, float* factor, std::vector<float>& data)
{
    cv::Mat mat;
    int rh = img->rows;
    int rw = img->cols;
    int rc = img->channels();
    cv::cvtColor(*img, mat, cv::COLOR_BGR2RGB);
    int max_image_length = rw > rh ? rw : rh;
    cv::Mat max_image = cv::Mat::zeros(max_image_length, max_image_length, CV_8UC3);
    max_image = max_image * 255;
    cv::Rect roi(0, 0, rw, rh);
    mat.copyTo(cv::Mat(max_image, roi));
    cv::Mat resize_img;
    cv::resize(max_image, resize_img, cv::Size(length, length), 0.0f, 0.0f, cv::INTER_LINEAR);

    *factor = (float)((float)max_image_length / (float)length);
    resize_img.convertTo(resize_img, CV_32FC3, 1 / 255.0);
    rh = resize_img.rows;
    rw = resize_img.cols;
    rc = resize_img.channels();
    for (int i = 0; i < rc; ++i)
    {
        cv::extractChannel(resize_img, cv::Mat(rh, rw, CV_32FC1, data.data() + i * rh * rw), i);
    }
}


std::vector<DetResult> post_process(float* result, float factor)
{
	cv::Mat output = cv::Mat(4 + class_num, outputLength, CV_32F, result);
	cv::transpose(output, output);

	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;
	for (int i = 0; i < output.rows; i++)
	{
		cv::Mat confidences_all = output.row(i).colRange(4, 4 + class_num);
		cv::Point classIdsPoint;
		double confidence;
		cv::minMaxLoc(confidences_all, 0, &confidence, 0, &classIdsPoint);
		if (confidence > 0.25)
		{
			cv::Rect box;
			float cx = output.at<float>(i, 0);
			float cy = output.at<float>(i, 1);
			float ow = output.at<float>(i, 2);
			float oh = output.at<float>(i, 3);
			box.x = static_cast<int>((cx - 0.5 * ow) * factor);
			box.y = static_cast<int>((cy - 0.5 * oh) * factor);
			box.width = static_cast<int>(ow * factor);
			box.height = static_cast<int>(oh * factor);

			boxes.push_back(box);
			classIds.push_back(classIdsPoint.x);
			confidences.push_back(confidence);
		}
	}
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indices);
    std::vector<DetResult> re;
    for (int i = 0; i < indices.size(); i++)
    {
        int index = indices[i];
        DetResult det(boxes[index], confidences[index], classIds[index]);
        re.push_back(det);
    }
    return re;
}


void warmup(ov::InferRequest request)
{
	std::vector<float> inputData(shape * shape * 3);
	memcpy(request.get_input_tensor().data<float>(), inputData.data(), shape * shape * 3);
	request.infer();
}


void yolov8_async_infer()
{
    std::string model_path = "/home/gqwang/DL/openvino/yolov8-openvino/yolov8n.xml";
    ov::Core core;
    auto model = core.read_model(model_path);
    auto compiled_model = core.compile_model(model, "GPU");
    std::vector<ov::InferRequest>requests = { compiled_model.create_infer_request(), compiled_model.create_infer_request() };
    cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
        std::cerr << "Failed to open camera" << std::endl;
        return;
    }
    float factor = 0;
    requests[0].get_input_tensor().set_shape(std::vector<size_t>{1, 3, shape, shape});
    requests[1].get_input_tensor().set_shape(std::vector<size_t>{1, 3, shape, shape});
    cv::Mat frame;
    capture.read(frame);
    std::vector<float> inputData(shape * shape * 3);
    pre_process(&frame, shape, &factor, inputData);
    memcpy(requests[0].get_input_tensor().data<float>(), inputData.data(), shape * shape * 3 * sizeof(float));
    requests[0].start_async();
    while (true)
    {
        cv::Mat next_frame;
        if (!capture.read(next_frame)) {
            break;
        }

        auto start = std::chrono::high_resolution_clock::now();
        pre_process(&next_frame, shape, &factor, inputData);
        memcpy(requests[1].get_input_tensor().data<float>(), inputData.data(), shape * shape * 3 * sizeof(float));
        requests[1].start_async();
        requests[0].wait();
        float* output_data = requests[0].get_output_tensor().data<float>();
        std::vector<DetResult> result = post_process(output_data, factor);

        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
        int fps = int(1000.0 / time);
        std::cout << "fps: " << std::right << std::setw(3) << fps << "  time: " << std::right << std::setw(2) << time << "ms" << std::endl;

        draw_bbox(frame, result);
        cv::imshow("frame", frame);
        cv::waitKey(1);
        frame = next_frame;
        std::swap(requests[0], requests[1]);
    }
    cv::destroyAllWindows();
}

int main()
{
	yolov8_async_infer();
	return 0;
}

