#include "trt_utils.h"
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>

using namespace nvinfer1;

template <typename T>
using SampleUniquePtr = std::unique_ptr<T>;

nvinfer1::Dims mInputDims;
nvinfer1::Dims mOutputDims;

class MinimalLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Log only warnings and errors
        if (severity == Severity::kERROR) {
            std::cerr << "[ERROR] " << msg << std::endl;
        } else if (severity == Severity::kWARNING) {
            std::cerr << "[WARNING] " << msg << std::endl;
        }
    }
};

MinimalLogger gLogger;

class ManualBufferManager {
public:
    ManualBufferManager(const std::shared_ptr<nvinfer1::ICudaEngine>& engine)
        : mEngine(engine) {
        initializeBuffers();
    }

    ~ManualBufferManager() {
        freeBuffers();
    }

    void* getHostBuffer(int index) {
        return mHostBuffers[index];
    }

    void* getDeviceBuffer(int index) {
        return mDeviceBuffers[index];
    }

    void copyInputToDevice(int index) {
        size_t size = mBufferSizes[index];
        cudaMemcpy(mDeviceBuffers[index], mHostBuffers[index], size, cudaMemcpyHostToDevice);
    }

    void copyOutputToHost(int index) {
        size_t size = mBufferSizes[index];
        cudaMemcpy(mHostBuffers[index], mDeviceBuffers[index], size, cudaMemcpyDeviceToHost);
    }

    std::vector<void*>& getDeviceBindings() {
        return mDeviceBuffers;
    }

private:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    std::vector<void*> mHostBuffers;
    std::vector<void*> mDeviceBuffers;
    std::vector<size_t> mBufferSizes;

    void initializeBuffers() {
        int nbBindings = mEngine->getNbIOTensors();
        mHostBuffers.resize(nbBindings, nullptr);
        mDeviceBuffers.resize(nbBindings, nullptr);
        mBufferSizes.resize(nbBindings, 0);

        for (int i = 0; i < nbBindings; ++i) {
            const char* tensorName = mEngine->getIOTensorName(i);
            auto dims = mEngine->getTensorShape(tensorName);
            size_t size = getBufferSize(dims, mEngine->getTensorDataType(tensorName));
            mBufferSizes[i] = size;

            mHostBuffers[i] = malloc(size);
            cudaMalloc(&mDeviceBuffers[i], size);
        }
    }

    void freeBuffers() {
        for (void* buffer : mHostBuffers) {
            if (buffer) free(buffer);
        }

        for (void* buffer : mDeviceBuffers) {
            if (buffer) cudaFree(buffer);
        }
    }

    size_t getBufferSize(const nvinfer1::Dims& dims, nvinfer1::DataType dataType) {
        size_t size = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            size *= dims.d[i];
        }

        size_t elementSize = getElementSize(dataType);
        return size * elementSize;
    }

    size_t getElementSize(nvinfer1::DataType dataType) {
        switch (dataType) {
            case nvinfer1::DataType::kFLOAT: return 4;
            case nvinfer1::DataType::kHALF: return 2;
            case nvinfer1::DataType::kINT8: return 1;
            case nvinfer1::DataType::kINT32: return 4;
            default: throw std::runtime_error("Unsupported data type.");
        }
    }
};

// Preprocess input image for YOLO
void preprocessImage(const cv::Mat& inputMat, ManualBufferManager& buffers) {
    const int32_t inputC = mInputDims.d[1];
    const int32_t inputH = mInputDims.d[2];
    const int32_t inputW = mInputDims.d[3];
    std::cout << "Input dimensions: C=" << inputC << ", H=" << inputH << ", W=" << inputW << std::endl;

    cv::Mat resizedImage;
    cv::resize(inputMat, resizedImage, cv::Size(inputW, inputH));
    resizedImage.convertTo(resizedImage, CV_32FC3, 1.0 / 255.0);
    cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGR2RGB);

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(0));
    size_t requiredSize = inputC * inputH * inputW * sizeof(float);

    for (int h = 0; h < inputH; ++h) {
        for (int w = 0; w < inputW; ++w) {
            cv::Vec3f pixel = resizedImage.at<cv::Vec3f>(h, w);
            for (int c = 0; c < inputC; ++c) {
                hostDataBuffer[c * inputH * inputW + h * inputW + w] = pixel[c];
            }
        }
    }

    std::cout << "Preprocessing completed successfully." << std::endl;
}

// Postprocess YOLO output
bool postprocessOutput(ManualBufferManager& buffers, cv::Mat& image) {
    // const float confThreshold = 0.5;
    // const float nmsThreshold = 0.4;
    float confThreshold = 0.1;
    const float nmsThreshold = 0.1;

    float* output = static_cast<float*>(buffers.getHostBuffer(1));

    int numDetections = mOutputDims.d[1];  // Number of detections
    int numClasses = mOutputDims.d[2] - 5; // Number of classes
    std::cout << "Number of detections: " << numDetections << std::endl;
    std::cout << "Number of classes: " << numClasses << std::endl;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    for (int i = 0; i < numDetections; ++i) {
        float confidence = output[i * (numClasses + 5) + 4]; // Object confidence score
        if (confidence > confThreshold) {
            // std::cout << "Detection " << i << ": Confidence = " << confidence << std::endl;

            float centerX = output[i * (numClasses + 5) + 0]; // Scale to original width
            float centerY = output[i * (numClasses + 5) + 1]; // Scale to original height
            float width = output[i * (numClasses + 5) + 2];   // Scale to original width
            float height = output[i * (numClasses + 5) + 3];  // Scale to original height

            int x1 = static_cast<int>(centerX - width / 2);
            int y1 = static_cast<int>(centerY - height / 2);
            int x2 = static_cast<int>(centerX + width / 2);
            int y2 = static_cast<int>(centerY + height / 2);

            std::cout << "Box: (" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << ")" << std::endl;

            boxes.emplace_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));

            // Determine the class with the highest score
            float maxClassScore = 0.0f;
            int classId = -1;
            for (int c = 0; c < numClasses; ++c) {
                float classScore = output[i * (numClasses + 5) + 5 + c];
                if (classScore > maxClassScore) {
                    maxClassScore = classScore;
                    classId = c;
                }
            }

            confidences.push_back(confidence * maxClassScore);
            classIds.push_back(classId);

            std::cout << "Class ID: " << classId << ", Class confidence: " << maxClassScore << std::endl;
        }
    }

    std::cout << "Boxes after filtering: " << boxes.size() << std::endl;

    // Perform Non-Maximum Suppression (NMS)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    cv::Mat displayImage;
    cv::resize(image, displayImage, cv::Size(640, 640)); // Resize to 800x800 for display

    // for (int idx : indices) {
    //     const auto& box = boxes[idx];
    //     cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);
    //     cv::putText(image, "Class: " + std::to_string(classIds[idx]), cv::Point(box.x, box.y - 10),
    //                 cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    // }
    for (int idx : indices) {
        const auto& box = boxes[idx];
        cv::rectangle(displayImage, box, cv::Scalar(0, 255, 0), 2);
        cv::putText(displayImage, "Class: " + std::to_string(classIds[idx]), 
                    cv::Point(box.x, box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    // Display the resized image with bounding boxes
    cv::imshow("YOLO Detection", displayImage);
    cv::waitKey(0);

    return true;
}

bool runYOLOInference(const std::string& modelPath, const std::string& enginePath, const std::string& imagePath) {

    std::ifstream engineFile(enginePath, std::ios::binary);
    if (!engineFile) {
        throw std::runtime_error("Failed to open engine file: " + enginePath);
    }

    engineFile.seekg(0, std::ios::end);
    size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);

    std::vector<char> engineData(engineSize);
    engineFile.read(engineData.data(), engineSize);
    engineFile.close();

    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));

    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(engineData.data(), engineSize), TRT::InferDeleter());
    
    ManualBufferManager buffers(engine);
    auto context = SampleUniquePtr<IExecutionContext>(engine->createExecutionContext());

    cv::Mat image = cv::imread(imagePath);

    const char* inputName = engine->getIOTensorName(0);
    const char* outputName = engine->getIOTensorName(1);

    mInputDims = context->getTensorShape(inputName);
    mOutputDims = context->getTensorShape(outputName);

    preprocessImage(image, buffers);
    buffers.copyInputToDevice(0);
    context->executeV2(buffers.getDeviceBindings().data());
    buffers.copyOutputToHost(1);
    postprocessOutput(buffers, image);

    return true;

}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <modelPath> <imagePath> <enginePath>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string modelPath = argv[1]; // Model path (first argument)
    std::string imagePath = argv[2]; // Image path (second argument)
    const std::string enginePath = argv[3]; // Engine path (third argument)

    if (runYOLOInference(modelPath, enginePath, imagePath)) {
        return EXIT_SUCCESS;
    }
    return EXIT_FAILURE;
}


