# 使用指南

此文档为开发平台SDK示例代码使用说明文档。

## 用途

在线上完成模型训练之后，后续步骤即为模型测试，而为了能够使用线上平台的自动测试功能，需要将模型封装成统一的接口，即为EV_SDK。这份代码提供了两种SDK的封装方式

1. 使用C++语言封装模型；
2. 使用Python语言封装模型。

## 目录

### 代码目录结构

```
ev_sdk
|-- 3rd
|   `-- cJSON             	# c语言版json解析库
|-- CMakeLists.txt          # 本项目的cmake构建文件
|-- README.md       # 本说明文件
|-- model           # 模型数据存放文件夹
|-- include         # 头文件目录
|   `-- ji.h        # 需要实现的接口头文件
|-- lib             # 本项目编译并安装之后，默认会将依赖的库放在该目录，包括libji.so
|-- src             # ji.h的接口实现代码
`-- test            # 针对ji.h中所定义接口的测试代码，请勿修改！！！
```
## 示例代码的使用方法

示例代码使用一个**ssd_inception_v2**模型的行人检测算法，推理引擎为OpenVINO Inference Engine。

### C++版编译方法

```shell
mkdir -p /usr/local/ev_sdk/build
cd /usr/local/ev_sdk/build
cmake -DUSE_PYTHON_API=OFF ..
make install
```

### Python版编译方法

```shell
mkdir -p /usr/local/ev_sdk/build
cd /usr/local/ev_sdk/build
cmake -DUSE_PYTHON_API=ON ..
make install
```

### 测试

在运行**示例代码**的测试程序之前，需要手动下载模型文件：

```shell
wget http://10.9.0.146:8888/group1/M00/00/01/CgkA616-GFeEZMcNAAAAAHEKBs82984.gz -O /tmp/ssd_inception_v2_pedestrian.tar.gz && tar zxf /tmp/ssd_inception_v2_pedestrian.tar.gz -C /usr/local/ev_sdk/model
```

运行测试程序：

```shell
cd /usr/local/ev_sdk/bin
./test-ji-api -f 1 -i ../data/dog.jpg
```

样例输出内容：

```json
code: 0
json: {
	"objects":	[{
			"xmin":	116,
			"ymin":	557,
			"xmax":	557,
			"ymax":	860,
			"confidence":	0.988156,
			"name":	"class1"
		}]
}
```

**请根据实际比赛项目要求，调整上述`json`输出结果，自动测试会根据输出的`json`结果来测试模型。**

## 如何基于示例代码实现自己的SDK

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

3. 根据实际项目，将结果封装成项目所规定的输入输出格式

   示例代码中使用的是目标检测类项目，因此需要根据实际项目，添加检测类别信息：

   ```c++
   # src/SampleDetectorImpl.cpp, SampleDetector::init()
   mIDNameMap.insert(std::make_pair<int, std::string>(1, "class0"));	// id 1, 类别名称 class0
   mIDNameMap.insert(std::make_pair<int, std::string>(2, "class1")); // id 2, 类别名称 class1
   ```

### 使用Python接口

根据自己实际的模型名称、模型输入输出、模型推理逻辑，修改`src/ji.py`和`src/SampleDetectorPy.cpp`

1. 实现模型初始化：

   ```python
   # src/ji.py
   def init()
   ```

2. 实现模型推理：

   ```python
   # src/ji.py
   def process_image(net, input_image, thresh)
   ```

3. 根据实际项目，将结果封装成项目所规定的输入输出格式

   示例代码中使用的是目标检测类项目，因此需要根据实际项目，添加检测类别信息：

   ```python
   # src/ji.py
   label_id_map = {1: "class0", 2: "class1"}
   ```


在示例代码中，检测到的对象使用`SampleDetector::Object`结构体存储，最终会输出到`ji.cpp`内的`ji_calc_frame`接口函数中。