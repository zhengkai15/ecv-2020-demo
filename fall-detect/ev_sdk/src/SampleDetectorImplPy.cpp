//
// Created by hrh on 2020/4/13.
//

#include "SampleDetectorImplPy.hpp"
#include <glog/logging.h>

static uchar *pCacheImage = nullptr;
static unsigned int pCacheSize = 0;

/**
 * @brief Check if pyObject callable
 * @return true if pyObject is callable, else false
 */
static bool is_pyobj_callable(PyObject *pyObject) {
    return pyObject && PyCallable_Check(pyObject);
}

/**
 * @brief 将cv::Mat转换成numpy array
 *
 * @param img cv::Mat, BGR格式
 * @return numpy.ndarray PyObject
 */
static PyObject* mat2ndarray(const cv::Mat &img) {
    cv::Mat image = img.clone();
    image.convertTo(img, CV_8UC3);

    // Copy data from cv::Mat
    int image_len = image.cols * image.rows * image.channels();
    if (pCacheSize < image_len) {
        free(pCacheImage);
        pCacheImage = new uchar[image_len];
        pCacheSize = image_len;
    }

    memcpy(pCacheImage, image.data, image_len);
    // Construct ndarray
    npy_intp dimensions[3] = {image.rows, image.cols, image.channels()};    // h, w, c
    PyObject *img_arr = PyArray_SimpleNewFromData(image.dims + 1, &dimensions[0], NPY_UBYTE, pCacheImage);
    if (img_arr == nullptr) {
        LOG(ERROR) << "mat2ndarray failed!";
        PyErr_Print();
    }
    return img_arr;
}

STATUS SampleDetectorImplPy::init() {
    if (!initPythonEnv()) {
        LOG(ERROR) << "Failed to initialize python env!";
        return ERROR_INIT;
    }

    // -------------------------------- Initialize model --------------------------------
    mPredictor = PyObject_Call(PY_APIS["init"], PyTuple_New(0), PyDict_New());
    if (mPredictor == nullptr) {
        PyErr_Print();
        LOG(ERROR) << "init failed.";
        return ERROR_INIT;
    }

    return SUCCESS_INIT;
}

void SampleDetectorImplPy::unInit() {
    unInitPythonEnv();
}

STATUS SampleDetectorImplPy::processImage(const cv::Mat &image, std::vector<Object> &detectResults) {
    detectResults.clear();

    PyObject *pyInFrame = mat2ndarray(image);
    PyObject *pArgs = PyDict_New();
    PyDict_SetItem(pArgs, PyUnicode_DecodeFSDefault("net"), mPredictor);
    PyDict_SetItem(pArgs, PyUnicode_DecodeFSDefault("input_image"), pyInFrame);
    PyDict_SetItem(pArgs, PyUnicode_DecodeFSDefault("thresh"), PyFloat_FromDouble(mThresh));

    // ---------------------------- Call Python inference API ---------------------------
    if (!is_pyobj_callable(PY_APIS["process_image"])) {
        LOG(ERROR) << "process_image is not callable!";
        return ERROR_PROCESS;
    }
    PyObject *result = PyObject_Call(PY_APIS["process_image"], PyTuple_New(0), pArgs);
    if (result == nullptr) {
        Py_DECREF(pArgs);
        Py_DECREF(pyInFrame);
        PyErr_Print();
        LOG(ERROR) << "process_image failed.";
        return ERROR_PROCESS;
    }

    // -------------------------------- Parse output ------------------------------------
    for (int i = 0; i < PyList_Size(result); ++i) {
        PyObject * obj = PyList_GetItem(result, i);
        PyObject * labelObj = PyUnicode_AsEncodedString(PyDict_GetItemString(obj, "label"), "utf-8", "~E~");
        const char *labelStr = PyBytes_AS_STRING(labelObj);
        Py_XDECREF(labelObj);
        PyObject * boxObj = PyDict_GetItemString(obj, "box");
        int xmin = PyLong_AsLong(PyDict_GetItemString(boxObj, "xmin"));
        int ymin = PyLong_AsLong(PyDict_GetItemString(boxObj, "ymin"));
        int xmax = PyLong_AsLong(PyDict_GetItemString(boxObj, "xmax"));
        int ymax = PyLong_AsLong(PyDict_GetItemString(boxObj, "ymax"));
        double prob = PyFloat_AsDouble(PyDict_GetItemString(obj, "prob"));
        detectResults.push_back({
            static_cast<float>(prob),
            labelStr,
            cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin)});
    }
    return SUCCESS_PROCESS;
}

bool SampleDetectorImplPy::initPythonEnv() {
    Py_Initialize();
    import_array();

    LOG(INFO) << "Loading ji.py...";
    PyObject *sysPath = PySys_GetObject("path");
    PyList_Append(sysPath, PyUnicode_DecodeFSDefault("/usr/local/ev_sdk/src"));
    pModule = PyImport_ImportModule("ji");
    if (pModule == NULL) {
        LOG(INFO) << "ji python module does not exist!";
        PyErr_Print();
        return false;
    }
    LOG(INFO) << "ji python module successfully loaded.";

    for (auto &it : PY_APIS) {
        std::string funcName = it.first;
        PyObject *pFunc = PyObject_GetAttrString(pModule, funcName.c_str());
        if (pFunc && PyCallable_Check(pFunc)) {
            it.second = pFunc;
            LOG(INFO) << funcName << " loaded.";
        } else {
            it.second = nullptr;
            LOG(ERROR) << funcName << " is not callable.";
            return false;
        }
    }
    return true;
}

bool SampleDetectorImplPy::unInitPythonEnv() {
    for (auto &it : PY_APIS) {
        if (is_pyobj_callable(it.second)) {
            Py_DECREF(it.second);
            it.second = nullptr;
        }
    }
    Py_DECREF(pModule);
    pModule = nullptr;
    Py_Finalize();
    return true;
}
