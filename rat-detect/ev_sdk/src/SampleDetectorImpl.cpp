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

STATUS SampleDetectorImpl::init() {
    // --------------------------- Replace to use your own model ------------------------
    const char *modelXMLPath = "/usr/local/ev_sdk/model/openvino/ssd_mobilenet_v2.xml";

    // ------------------------------ Add object name -----------------------------------
    mIDNameMap.insert(std::make_pair<int, std::string>(1, "rat"));

    // -------------------------------- Initialize model --------------------------------
    if (modelXMLPath == nullptr) {
        LOG(ERROR) << "Invalid init args!";
        return ERROR_INIT;
    }
    struct stat st;
    if (stat(modelXMLPath, &st) != 0) {
        LOG(ERROR) << modelXMLPath << " not found!";
        return ERROR_INIT;
    }
    LOG(INFO) << "Loading model...";
    Core ie;
    std::string binFileName = fileNameNoExt(modelXMLPath) + ".bin";
    if (stat(binFileName.c_str(), &st) != 0) {
        LOG(ERROR) << binFileName << " not found!";
        return ERROR_INIT;
    }
    LOG(INFO) << "Loading network files:\n\t" << modelXMLPath << "\n\t" << binFileName;
    CNNNetwork network = ie.ReadNetwork(modelXMLPath);

    // -------------------------------- Get input info ----------------------------------
    LOG(INFO) << "Preparing input blobs";
    mBatchSize = network.getBatchSize();
    LOG(INFO) << "Batch size is " << std::to_string(mBatchSize);
    InputsDataMap inputsInfo(network.getInputsInfo());
    SizeVector inputImageDims;
    InputInfo::Ptr inputInfo;
    for (auto & item : inputsInfo) {
        mImageInputName = item.first;
        inputInfo = item.second;
        LOG(INFO) << "Found input name:" << mImageInputName;

        Precision inputPrecision = Precision::U8;
        item.second->setPrecision(inputPrecision);
    }
    if (inputInfo == nullptr) {
        inputInfo = inputsInfo.begin()->second;
    }
    mModelInputWidth = inputInfo->getTensorDesc().getDims()[3];
    mModelInputHeight = inputInfo->getTensorDesc().getDims()[2];

    // -------------------------------- Get output info ---------------------------------
    LOG(INFO) << "Preparing output blobs";
    OutputsDataMap outputsInfo(network.getOutputsInfo());
    DataPtr outputInfo;
    for (const auto& out : outputsInfo) {
        if (out.second->getCreatorLayer().lock()->type == "DetectionOutput") {
            mOutputName = out.first;
            outputInfo = out.second;
        }
    }

    if (outputInfo == nullptr) {
        LOG(ERROR) << "Can't find a DetectionOutput layer in the topology";
    }
    SizeVector outputDims = outputInfo->getTensorDesc().getDims();
    if (outputDims.size() != 4) {
        throw std::logic_error("Incorrect output dimensions for SSD model");
    }

    mMaxProposalCount = outputDims[2];
    mObjectInfoDims = outputDims[3];
    if (mObjectInfoDims != 7) {
        throw std::logic_error("Output item should have 7 as a last dimension");
    }

    // Set the precision of output data provided by the user, should be called before load of the network to the device
    outputInfo->setPrecision(Precision::FP32);

    // -------------------------------- Load model to device ----------------------------
    LOG(INFO) << "Loading model to the device";
    mExecutableNetwork = ie.LoadNetwork(network, "CPU", {});
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
    std::vector<size_t> image_widths, image_heights;

    image_widths.push_back(cv_image.cols);
    image_heights.push_back(cv_image.rows);

    // Resize image
    cv::Mat resized_img(cv_image);
    cv::resize(cv_image, resized_img, cv::Size(mModelInputWidth, mModelInputHeight));
    std::shared_ptr<unsigned char> input_data;
    size_t resized_size = resized_img.size().width * resized_img.size().height * resized_img.channels();
    input_data.reset(new unsigned char[resized_size], std::default_delete<unsigned char[]>());
    for (size_t id = 0; id < resized_size; ++id) {
        input_data.get()[id] = resized_img.data[id];
    }
    input_images.push_back(input_data);
    if (input_images.empty()) throw std::logic_error("Valid input images were not found!");

    // Fill data into request
    InferRequest infer_request = mExecutableNetwork.CreateInferRequest();
    fillInputData(infer_request, mImageInputName, input_images, mBatchSize);

    // -------------------------------- Inference ---------------------------------------
    LOG(INFO) << "Start inference";
    infer_request.Infer();

    // -------------------------------- Parse output ------------------------------------
    LOG(INFO) << "Processing output blobs";
    const Blob::Ptr output_blob = infer_request.GetBlob(mOutputName);
    MemoryBlob::CPtr output_mem_blob = as<MemoryBlob>(output_blob);
    if (!output_mem_blob) {
        throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                               "but by fact we were not able to cast output to MemoryBlob");
    }
    // locked memory holder should be alive all time while access to its buffer happens
    auto output_mem_holder = output_mem_blob->rmap();
    const float *detection = output_mem_holder.as<const PrecisionTrait<Precision::FP32>::value_type *>();

    // Each detection has image_id that denotes processed image
    // mObjectInfoDims: (image_id, label, confidence, xmin, ymin, xmax, ymax)
    for (int cur_proposal = 0; cur_proposal < mMaxProposalCount; cur_proposal++) {
        auto image_index = static_cast<int>(detection[cur_proposal * mObjectInfoDims + 0]);
        if (image_index < 0) {
            break;
        }
        float confidence = detection[cur_proposal * mObjectInfoDims + 2];
        if (confidence < mThresh) {
            continue;
        }
        auto label = static_cast<int>(detection[cur_proposal * mObjectInfoDims + 1]);
        auto xmin = static_cast<int>(detection[cur_proposal * mObjectInfoDims + 3] * image_widths[image_index]);
        auto ymin = static_cast<int>(detection[cur_proposal * mObjectInfoDims + 4] * image_heights[image_index]);
        auto xmax = static_cast<int>(detection[cur_proposal * mObjectInfoDims + 5] * image_widths[image_index]);
        auto ymax = static_cast<int>(detection[cur_proposal * mObjectInfoDims + 6] * image_heights[image_index]);

        LOG(INFO) << "[" << cur_proposal << "," << label << "] element, prob = " << confidence <<
                  "    (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")" << " batch id : " << image_index;
        result.emplace_back(SampleDetector::Object({
            confidence,
            mIDNameMap[label],
            cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin)
        }));
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
    return false;
}
