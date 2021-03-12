# 老鼠检测SDK封装代码

此文档为老鼠检测算法的SDK封装代码使用说明。

在线上完成模型训练之后，后续步骤即为模型测试，而为了能够使用线上平台的自动测试功能，需要将模型封装成统一的接口。平台两种封装接口方式：

1. 使用C++语言封装模型；
2. 使用Python语言封装模型；

这份示例代码提供了两种实现。

## 算法原理

首先使用ssd_inception_v2检测Motor目标，然后使用ssd_mobilenet_v2检测Motor目标框内的未佩戴头盔的人头。

## 代码运行依赖

此代码仅适用于平台上的TensorFlow1.13.2的镜像，**且仅能用于OpenVINO模型封装和测试**。

## 代码目录结构

```shell
ev_sdk
|-- CMakeLists.txt          # 本项目的cmake构建文件
|-- README.md       # 本说明文件
|-- model           # 模型数据存放文件夹
|-- include         # 头文件目录
|   `-- ji.h        # 需要实现的接口头文件
|-- lib             # 本项目编译并安装之后，默认会将依赖的库放在该目录，包括libji.so
|-- convert_openvino.sh # OpenVINO模型转换脚本，发起测试时系统会调用这个脚本做转换
|-- src             # ji.h的接口实现代码，如ji.cpp、ji.py
`-- test            # 针对ji.h中所定义接口的测试代码，请勿修改！！！
```
## 示例代码的使用方法

示例代码使用的推理引擎为OpenVINO Inference Engine。

### 在编码环境中测试SDK

1. 在**编码环境中**运行**示例代码**的测试程序之前，需要手动下载预训练的模型文件：

   ```dockerfile
   RUN wget http://10.9.0.146:8888/group1/M00/00/B1/CgkAkl8PBw6ELROJAAAAAM8GK6M4836.pb -O /usr/local/ev_sdk/model/ssd_mobilenet_v2_head.pb \
       && wget http://10.9.0.146:8888/group1/M00/00/B1/CgkAkl8Pu7aEUBenAAAAAExDeqA1858.pb -O  /usr/local/ev_sdk/model/ssd_inception_v2_motor.pb
   ```

2. 手动执行模型转换程序：

   ```shell
   bash /usr/local/ev_sdk/model/convert_openvino.sh
   ```

### 在平台发起测试

示例代码可以直接拷贝到`/usr/local/ev_sdk/`下提交并发起测试，**注意：如果要使用预训练的模型进行测试，请确保在发起测试时所选择的模型不会覆盖预训练模型**。

## 如何基于示例代码封装自己的SDK

### 使用C++接口

根据自己实际的模型名称、模型输入输出、模型推理逻辑，修改`src/SampleDetectorImpl.cpp`

1. 实现模型初始化：

   ```c++
   int SampleDetector::init();
   ```

2. 实现模型推理：

   ```c++
   STATUS SampleDetector::processImage(const cv::Mat &cv_image, std::vector<Object> &result);
   ```

3. 根据实际项目，将结果封装成项目所规定的输入输出格式，例如，在摩托车未佩戴头盔的检测项目中，需要保证`ji_calc_frame`接口处理后，输出如下格式的数据（请参考示例代码）：

   ```json
   {
      "objects": [
          {
              "xmin": 320,
              "ymin": 430,
              "xmax": 500,
              "ymax": 700,
              "name": "rat"
          }
      ]
   }
   ```

4. 编译程序

   ```shell
   mkdir -p /usr/local/ev_sdk/build
   cd /usr/local/ev_sdk/build
   make install
   ```

5. 运行测试程序

   ```shell
   cd /usr/local/ev_sdk/bin
   ./test-ji-api -f 1 -i ../data/dog.jpg
   ```

### 使用Python接口

与C++接口不同，当使用Python接口发起测试时，系统仅会运行`src/ji.py`内的代码，用户需要根据自己的模型名称、模型输入输出、模型推理逻辑，修改`src/ji.py`

1. 实现模型初始化：

   ```python
   # src/ji.py
   def init()
   ```

2. 实现模型推理：

   ```python
   # src/ji.py
   def process_image(net, input_image)
   ```
   其中process_image接口返回值，必须是JSON格式的字符串，并且格式符合要求。

3. 根据实际项目，将结果封装成项目所规定的输入输出格式，例如，在摩托车未佩戴头盔的检测项目中，需要保证`ji_calc_frame`接口处理后，输出如下格式的数据（请参考示例代码）：

   ```python
    {
      "objects": [
          {
              "xmin": 320,
              "ymin": 430,
              "xmax": 500,
              "ymax": 700,
              "name": "rat"
          }
      ]
   }
   ```
   
4. 测试程序

   ```shell
   python ji.py
   ```

6. **注意：**使用Python接口发起测试时，需要在`Dockerfile`中设置一个环境变量：

   ```dockerfile
   ENV AUTO_TEST_USE_JI_PYTHON_API=1
   ```