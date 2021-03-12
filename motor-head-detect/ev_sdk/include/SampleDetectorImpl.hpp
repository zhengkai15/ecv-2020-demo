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

    /**
     * 将inputImages 填充到inferRequest
     *
     * @param inferRequest
     * @param inputName 输入层的名称
     * @param inputImages 输入数据
     * @param batchSize 网络的batch size
     * @return 返回true如果处理正常，否则返回false
     */
    static bool fillInputData(InferRequest &inferRequest, std::string &inputName,
                              std::vector<std::shared_ptr<unsigned char>> &inputImages, size_t batchSize);

private:
    ExecutableNetwork mMotorExecNetwork;

    std::string mMotorInputName;
    std::string mMotorOutputName;
    size_t mMotorModelInputWidth{0};
    size_t mMotorModelInputHeight{0};
    size_t mMotorBatchSize{0};
    int mMotorMaxProposalCount{0};
    int mMotorObjectInfoDims{0};

    ExecutableNetwork mHeadExecNetwork;

    std::string mHeadInputName;
    std::string mHeadOutputName;
    size_t mHeadModelInputWidth{0};
    size_t mHeadModelInputHeight{0};
    size_t mHeadBatchSize{0};
    int mHeadMaxProposalCount{0};
    int mHeadObjectInfoDims{0};

    std::map<int, std::string> mMotorIDNameMap;     // motor id与label的映射表
    std::map<int, std::string> mHeadIDNameMap;      // head id与label的映射表

    double mMotorThresh{0.7};
    double mHeadThresh{0.2};
};

#endif //JI_SAMPLEDETECTORIMPL_HPP
