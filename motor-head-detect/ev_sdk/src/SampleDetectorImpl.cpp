//
// Created by hrh on 2019-09-02.
//

#include <cstring>
#include <iostream>
#include <fstream>
#include <glog/logging.h>
#include <sys/stat.h>
#include "SampleDetectorImpl.hpp"

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace InferenceEngine;

/**
 * @brief Gets filename without extension
 *
 * @param filepath - full file name
 * @return filename without extension
 */
static std::string fileNameNoExt(const std::string &filepath) {
    auto pos = filepath.rfind('.');
    if (pos == std::string::npos) return filepath;
    return filepath.substr(0, pos);
}

/**
 * Check if rect1 inside rect2, `overlap_thresh` is the least ratio of rect1 inside rect2
 *
 * @param rect1
 * @param rect2
 * @return true if rect1 inside rect2, false otherwise
 **/
static bool inside_rect(cv::Rect rect1, cv::Rect rect2, float overlap_thresh = 1.0f) {
    int left = std::max(rect1.x, rect2.x);
    int top = std::max(rect1.y, rect2.y);
    int right = std::min(rect1.y, rect2.y);
    int bottom = std::min(rect1.y, rect2.y);

    if (left >= right || top >= bottom) {
        return false;
    }

    double ratio = ((right - left) * (bottom - top) / rect1.area());
    return ratio <= 1 && ratio >= overlap_thresh;
}

STATUS SampleDetectorImpl::init() {
    // --------------------------- Replace to use your own model ------------------------
    const char *motorModelXMLPath = "/usr/local/ev_sdk/model/openvino/ssd_inception_v2_motor.xml";
    const char *headModelXMLPath = "/usr/local/ev_sdk/model/openvino/ssd_mobilenet_v2_head.xml";

    // ------------------------------ Add object name -----------------------------------
    mMotorIDNameMap.insert(std::make_pair<int, std::string>(1, "motor"));
    mHeadIDNameMap.insert(std::make_pair<int, std::string>(1, "head"));

    // -------------------------------- Initialize model --------------------------------
    if (motorModelXMLPath == nullptr) {
        LOG(ERROR) << "Invalid init args!";
        return ERROR_INIT;
    }
    if (headModelXMLPath == nullptr) {
        LOG(ERROR) << "Invalid init args!";
        return ERROR_INIT;
    }
    struct stat st;
    if (stat(motorModelXMLPath, &st) != 0) {
        LOG(ERROR) << motorModelXMLPath << " not found!";
        return ERROR_INIT;
    }
    if (stat(headModelXMLPath, &st) != 0) {
        LOG(ERROR) << headModelXMLPath << " not found!";
        return ERROR_INIT;
    }
    LOG(INFO) << "Loading model...";
    Core ie;
    std::string motorModelBinFileName = fileNameNoExt(motorModelXMLPath) + ".bin";
    std::string headModelBinFileName = fileNameNoExt(headModelXMLPath) + ".bin";
    if (stat(motorModelBinFileName.c_str(), &st) != 0) {
        LOG(ERROR) << motorModelBinFileName << " not found!";
        return ERROR_INIT;
    }
    if (stat(headModelBinFileName.c_str(), &st) != 0) {
        LOG(ERROR) << headModelBinFileName << " not found!";
        return ERROR_INIT;
    }
    LOG(INFO) << "Loading network files: motor model\n\t" << motorModelXMLPath << "\n\t" << motorModelBinFileName
              << "\n\t head model:\n\t" << headModelXMLPath << "\n\t" << headModelBinFileName;
    CNNNetwork motorNetwork = ie.ReadNetwork(motorModelXMLPath);
    CNNNetwork headNetwork = ie.ReadNetwork(headModelXMLPath);

    // -------------------------------- Get input info ----------------------------------
    LOG(INFO) << "Preparing input blobs";
    // motor model
    mMotorBatchSize = motorNetwork.getBatchSize();
    LOG(INFO) << "Batch size is " << std::to_string(mMotorBatchSize);
    InputsDataMap inputsInfo(motorNetwork.getInputsInfo());
    SizeVector inputImageDims;
    InputInfo::Ptr inputInfo;
    for (auto & item : inputsInfo) {
        mMotorInputName = item.first;
        inputInfo = item.second;
        LOG(INFO) << "Found input name:" << mMotorInputName;

        Precision inputPrecision = Precision::U8;
        item.second->setPrecision(inputPrecision);
    }
    if (inputInfo == nullptr) {
        inputInfo = inputsInfo.begin()->second;
    }
    mMotorModelInputWidth = inputInfo->getTensorDesc().getDims()[3];
    mMotorModelInputHeight = inputInfo->getTensorDesc().getDims()[2];

    // head model
    mHeadBatchSize = headNetwork.getBatchSize();
    LOG(INFO) << "Batch size is " << std::to_string(mHeadBatchSize);
    InputsDataMap headInputsInfo(headNetwork.getInputsInfo());
    SizeVector headInputImageDims;
    InputInfo::Ptr headInputInfo;
    for (auto & item : headInputsInfo) {
        mHeadInputName = item.first;
        headInputInfo = item.second;
        LOG(INFO) << "Found input name:" << mHeadInputName;

        Precision headInputPrecision = Precision::U8;
        item.second->setPrecision(headInputPrecision);
    }
    if (headInputInfo == nullptr) {
        headInputInfo = headInputsInfo.begin()->second;
    }
    mHeadModelInputWidth = headInputInfo->getTensorDesc().getDims()[3];
    mHeadModelInputHeight = headInputInfo->getTensorDesc().getDims()[2];
    LOG(INFO) << "Head model input:(" << mHeadModelInputWidth << ", " << mHeadModelInputHeight;

    // -------------------------------- Get output info ---------------------------------
    LOG(INFO) << "Preparing output blobs";
    // motor model
    OutputsDataMap motorOutputsInfo(motorNetwork.getOutputsInfo());
    DataPtr motorOutputInfo;
    for (const auto& out : motorOutputsInfo) {
        if (out.second->getCreatorLayer().lock()->type == "DetectionOutput") {
            mMotorOutputName = out.first;
            motorOutputInfo = out.second;
        }
    }

    if (motorOutputInfo == nullptr) {
        LOG(ERROR) << "Can't find a DetectionOutput layer in the topology";
    }
    SizeVector motorOutputDims = motorOutputInfo->getTensorDesc().getDims();
    if (motorOutputDims.size() != 4) {
        throw std::logic_error("Incorrect output dimensions for SSD model");
    }

    mMotorMaxProposalCount = motorOutputDims[2];
    mMotorObjectInfoDims = motorOutputDims[3];
    if (mMotorObjectInfoDims != 7) {
        throw std::logic_error("Output item should have 7 as a last dimension");
    }

    // Set the precision of output data provided by the user, should be called before load of the network to the device
    motorOutputInfo->setPrecision(Precision::FP32);

    // head model
    OutputsDataMap headOutputsInfo(headNetwork.getOutputsInfo());
    DataPtr headOutputInfo;
    for (const auto& out : headOutputsInfo) {
        if (out.second->getCreatorLayer().lock()->type == "DetectionOutput") {
            mHeadOutputName = out.first;
            headOutputInfo = out.second;
        }
    }

    if (headOutputInfo == nullptr) {
        LOG(ERROR) << "Can't find a DetectionOutput layer in the topology";
    }
    SizeVector headOutputDims = headOutputInfo->getTensorDesc().getDims();
    if (headOutputDims.size() != 4) {
        throw std::logic_error("Incorrect output dimensions for SSD model");
    }

    mHeadMaxProposalCount = headOutputDims[2];
    mHeadObjectInfoDims = headOutputDims[3];
    if (mHeadObjectInfoDims != 7) {
        throw std::logic_error("Output item should have 7 as a last dimension");
    }

    // Set the precision of output data provided by the user, should be called before load of the network to the device
    headOutputInfo->setPrecision(Precision::FP32);

    // -------------------------------- Load model to device ----------------------------
    LOG(INFO) << "Loading model to the device";
    mMotorExecNetwork = ie.LoadNetwork(motorNetwork, "CPU", {});
    mHeadExecNetwork = ie.LoadNetwork(headNetwork, "CPU", {});

    LOG(INFO) << "Init done.";
    return SUCCESS_INIT;
}

void SampleDetectorImpl::unInit() {
}

STATUS SampleDetectorImpl::processImage(const cv::Mat &cv_image, std::vector<Object> &result) {
    if (cv_image.empty()) {
        LOG(ERROR) << "Invalid input!";
        return ERROR_INVALID_INPUT;
    }
    LOG(INFO) << "Input image size:" << cv_image.size();

    // -------------------------------- Prepare input -----------------------------------
    std::vector<std::shared_ptr<unsigned char>> input_images; // 最终输入到推理引擎的数据

    int image_width = cv_image.cols;
    int image_height = cv_image.rows;

    // Resize image
    cv::Mat resized_img(cv_image);
    cv::resize(cv_image, resized_img, cv::Size(mMotorModelInputWidth, mMotorModelInputHeight));
    std::shared_ptr<unsigned char> input_data;
    size_t resized_size = resized_img.size().width * resized_img.size().height * resized_img.channels();
    input_data.reset(new unsigned char[resized_size], std::default_delete<unsigned char[]>());
    for (size_t id = 0; id < resized_size; ++id) {
        input_data.get()[id] = resized_img.data[id];
    }
    input_images.push_back(input_data);

    // Fill data into request
    InferRequest infer_request = mMotorExecNetwork.CreateInferRequest();
    fillInputData(infer_request, mMotorInputName, input_images, mMotorBatchSize);

    // -------------------------------- Inference ---------------------------------------
    LOG(INFO) << "Start inference";
    infer_request.Infer();

    // -------------------------------- Parse output ------------------------------------
    LOG(INFO) << "Processing output blobs";
    const Blob::Ptr output_blob = infer_request.GetBlob(mMotorOutputName);
    MemoryBlob::CPtr output_mem_blob = as<MemoryBlob>(output_blob);
    if (!output_mem_blob) {
        throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                               "but by fact we were not able to cast output to MemoryBlob");
    }
    // locked memory holder should be alive all time while access to its buffer happens
    auto output_mem_holder = output_mem_blob->rmap();
    const float *detection = output_mem_holder.as<const PrecisionTrait<Precision::FP32>::value_type *>();

    // Each detection has image_id that denotes processed image
    // mObjectInfoDims: (image_index, label, confidence, xmin, ymin, xmax, ymax)
    std::vector<SampleDetector::Object> motors;
    for (int cur_proposal = 0; cur_proposal < mMotorMaxProposalCount; cur_proposal++) {
        auto image_index = static_cast<int>(detection[cur_proposal * mMotorObjectInfoDims + 0]);
        if (image_index < 0) {
            break;
        }
        float confidence = detection[cur_proposal * mMotorObjectInfoDims + 2];
        if (confidence < mMotorThresh) {
            continue;
        }
        auto label_idx = static_cast<int>(detection[cur_proposal * mMotorObjectInfoDims + 1]);
        auto xmin = static_cast<int>(detection[cur_proposal * mMotorObjectInfoDims + 3] * image_width);
        auto ymin = static_cast<int>(detection[cur_proposal * mMotorObjectInfoDims + 4] * image_height);
        auto xmax = static_cast<int>(detection[cur_proposal * mMotorObjectInfoDims + 5] * image_width);
        auto ymax = static_cast<int>(detection[cur_proposal * mMotorObjectInfoDims + 6] * image_height);

        LOG(INFO) << "[" << cur_proposal << "," << mMotorIDNameMap[label_idx] << "] element, prob = " << confidence <<
                  " (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")";

        auto obj = SampleDetector::Object({
                                                  confidence,
                                                  mMotorIDNameMap[label_idx],
                                                  cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin)
                                          });
        if (obj.name == "motor") {
            motors.emplace_back(obj);
        }
    }

    // Use head model to further detect head inside motor bbox
    input_images.clear();
    int i = 0;
    for (auto motor : motors) {
        // Prepare input
        cv::Rect expanded_motor_rect(motor.rect);
        int expanded_xmin = std::max(0, motor.rect.x - 10);
        int expanded_ymin = std::max(0, motor.rect.y - 10);
        int expanded_xmax = std::min(image_width, motor.rect.x + motor.rect.width + 10);
        int expanded_ymax = std::min(image_height, motor.rect.y + motor.rect.height + 10);
        cv::Mat motor_image = cv_image(cv::Rect(expanded_xmin, expanded_ymin, expanded_xmax - expanded_xmin, expanded_ymax - expanded_ymin)).clone();
        int motor_image_width = expanded_xmax - expanded_xmin;
        int motor_image_height = expanded_ymax - expanded_ymin;
        cv::resize(motor_image, motor_image, cv::Size(mHeadModelInputWidth, mHeadModelInputHeight));
        std::shared_ptr<unsigned char> input_data;
        size_t resized_size = motor_image.size().width * motor_image.size().height * motor_image.channels();
        input_data.reset(new unsigned char[resized_size], std::default_delete<unsigned char[]>());
        for (size_t id = 0; id < resized_size; ++id) {
            input_data.get()[id] = motor_image.data[id];
        }
        input_images.push_back(input_data);

        // Inference
        InferRequest request = mHeadExecNetwork.CreateInferRequest();
        fillInputData(request, mHeadInputName, input_images, mHeadBatchSize);
        request.Infer();

        // Parse output
        const Blob::Ptr output_blob = request.GetBlob(mHeadOutputName);
        MemoryBlob::CPtr output_mem_blob = as<MemoryBlob>(output_blob);
        if (!output_mem_blob) {
            throw std::logic_error("We expect output to be inherited from MemoryBlob, but by fact we were not able to cast output to MemoryBlob");
        }
        auto output_mem_holder = output_mem_blob->rmap();
        const float *head_detection = output_mem_holder.as<const PrecisionTrait<Precision::FP32>::value_type *>();

        for (int cur_proposal = 0; cur_proposal < mHeadMaxProposalCount; cur_proposal++) {
            auto image_index = static_cast<int>(head_detection[cur_proposal * mHeadObjectInfoDims + 0]);
            if (image_index < 0) {
                break;
            }
            float confidence = head_detection[cur_proposal * mHeadObjectInfoDims + 2];
            // LOG(INFO) << "Found head, conf:" << confidence;
            if (confidence < mHeadThresh) {
                continue;
            }
            auto label_idx = static_cast<int>(head_detection[cur_proposal * mHeadObjectInfoDims + 1]);
            auto xmin = static_cast<int>(head_detection[cur_proposal * mHeadObjectInfoDims + 3] * motor_image_width);
            auto ymin = static_cast<int>(head_detection[cur_proposal * mHeadObjectInfoDims + 4] * motor_image_height);
            auto xmax = static_cast<int>(head_detection[cur_proposal * mHeadObjectInfoDims + 5] * motor_image_width);
            auto ymax = static_cast<int>(head_detection[cur_proposal * mHeadObjectInfoDims + 6] * motor_image_height);

            if (mHeadIDNameMap.find(label_idx) == mHeadIDNameMap.end()) {
                continue;
            }
            LOG(INFO) << "[" << cur_proposal << "," << mHeadIDNameMap[label_idx] << "] element, prob = " << confidence <<
                      "    (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")" << " batch id : " << image_index;

            auto obj = SampleDetector::Object({
                                                      confidence,
                                                      mHeadIDNameMap[label_idx],
                                                      cv::Rect(xmin + expanded_xmin, ymin + expanded_ymin, xmax - xmin, ymax - ymin)    // Fix coordinate
                                              });
            result.emplace_back(obj);
        }
    }

    return SUCCESS_PROCESS;
}

bool SampleDetectorImpl::fillInputData(InferRequest &inferRequest, std::string &inputName,
                                       std::vector<std::shared_ptr<unsigned char>> &input_images, size_t batchSize) {
    LOG(INFO) << "Batch size is " << std::to_string(batchSize);
    if (batchSize != input_images.size()) {
        LOG(WARNING) << "Number of images " + std::to_string(input_images.size()) + " doesn't match batch size " + std::to_string(batchSize);
        batchSize = std::min(batchSize, input_images.size());
        LOG(WARNING) << "Number of images to be processed is "<< std::to_string(batchSize);
    }
    Blob::Ptr input_blob = inferRequest.GetBlob(inputName);
    MemoryBlob::Ptr input_mem_blob = as<MemoryBlob>(input_blob);
    if (!input_mem_blob) {
        LOG(ERROR) << "We expect image blob to be inherited from MemoryBlob, but by fact we were not able "
                      "to cast input_blob to MemoryBlob";
        return false;
    }
    // locked memory holder should be alive all time while access to its buffer happens
    auto input_mem_holder = input_mem_blob->wmap();

    size_t num_channels = input_mem_blob->getTensorDesc().getDims()[1];
    size_t image_size = input_mem_blob->getTensorDesc().getDims()[3] * input_mem_blob->getTensorDesc().getDims()[2];

    auto *data = input_mem_holder.as<unsigned char *>();
    // Iterate over all input images
    for (size_t image_id = 0; image_id < std::min(input_images.size(), batchSize); ++image_id) {
        // Iterate over all pixel in image (b,g,r)
        for (size_t pid = 0; pid < image_size; pid++) {
            // Iterate over all channels
            for (size_t ch = 0; ch < num_channels; ++ch) {
                // [images stride + channels stride + pixel id ] all in bytes
                data[image_id * image_size * num_channels + ch * image_size + pid] = input_images.at(image_id).get()[pid * num_channels + ch];
            }
        }
    }
    return true;
}
