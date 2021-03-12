//
// Created by hrh on 2019-09-02.
//

#ifndef JI_SAMPLEDETECTORIMPL_HPP
#define JI_SAMPLEDETECTORIMPL_HPP
#include <string>
#include <opencv2/core/mat.hpp>
#include <inference_engine.hpp>
#include <map>
#include "SampleDetector.hpp"

#define STATUS int

using namespace InferenceEngine;

/**
 * @brief 使用OpenVINO转换的行人检测模型，模型基于ssd inception v2 coco训练得到，模型转换请参考：
 *  https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html#ssd_single_shot_multibox_detector_topologies
 *
 *  这个示例使用OpenVINO Inference Engine的C++接口实现推理
 */
class SampleDetectorImpl: public SampleDetector {

public:
    STATUS init() override;

    void unInit() override;

    STATUS processImage(const cv::Mat &image, std::vector<Object> &detectResults) override;

private:
    ExecutableNetwork mExecutableNetwork;
    CNNNetwork mNetwork;

    InputInfo::Ptr mInputInfo{nullptr};
    DataPtr mOutputInfo{nullptr};
    SizeVector mOutputDims;
    std::string mImageInputName;
    std::string mOutputName;
    int mMaxProposalCount{0};
    int mObjectSize{0};

};

#endif //JI_SAMPLEDETECTORIMPL_HPP
