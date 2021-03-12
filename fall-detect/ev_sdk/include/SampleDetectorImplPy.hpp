//
// Created by rawk on 2020/4/13.
//

#ifndef JI_SAMPLEDETECTORIMPLPY_HPP
#define JI_SAMPLEDETECTORIMPLPY_HPP
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <string>
#include <opencv2/core/mat.hpp>
#include <map>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <object.h>
#include "SampleDetector.hpp"

#define STATUS int

/**
 * @brief 使用OpenVINO转换的行人检测模型，模型基于ssd inception v2 coco训练得到，模型转换请参考：
 *  https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html#ssd_single_shot_multibox_detector_topologies
 *
 *  这个示例使用OpenVINO Inference Engine的Python接口实现推理
 */

class SampleDetectorImplPy: public SampleDetector {

public:

    STATUS init() override;

    void unInit() override;

    STATUS processImage(const cv::Mat &image, std::vector<Object> &detectResults) override;

    /**
     * @brief 初始化Python环境，并加载ji.py
     */
    bool initPythonEnv();

    /**
     * @brief 释放Python解释器及ji.py模块
     */
    bool unInitPythonEnv();

private:
    PyObject *pModule{};
    PyObject *mPredictor{};

    // ji.py内的函数
    std::map<std::string, PyObject *> PY_APIS = {{"init",          nullptr},
                                                 {"process_image", nullptr}};

};

#endif //JI_SAMPLEDETECTORIMPLPY_HPP
