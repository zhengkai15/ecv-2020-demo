# 摩托车未佩戴头盔检测训练代码

## 说明
这份代码提供了一个检测摩托车中未佩戴头盔的人头的算法实现，代码可以直接拷贝到平台，提交后即可发起训练。

## 算法原理
首先使用ssd_inception_v2训练一个Motor目标的检测器，然后使用ssd_mobilenet_v2训练一个检测Motor目标框内的未佩戴头盔的人头检测器。

## 代码运行依赖
1. 此代码仅适用于平台上的TensorFlow1.13.2的镜像；
2. 示例代码使用[TensorFlow Object Detection API](https://github.com/tensorflow/models/) 的v1.13.0版本编写训练代码；

## 代码结构

```shell
├── convert_dataset_for_head_detector.py # 将数据集中Motor目标框内的Head目标提取出来并转换成tfrecord
├── convert_dataset_for_motor_detector.py	# 将数据集中的Motor目标提取出来并转换成tfrecord
├── Dockerfile
├── export_models.py	# 训练完成后的模型导出程序，将ckpt导出成pb文件
├── pipeline-config		# 训练模型的pipeline配置文件
├── pre-trained-model
├── README.md
├── requirements.txt
├── save_plots.py			# 读取TensorBoard文件并将损失更新数据导出成损失曲线图
├── start_train.sh		# 训练启动脚本
├── tf-models					# TensorFlow Object Detection API代码
├── train.py					# 训练代码
└── update_plots.py		# 调用平台的接口，将损失曲线图实时更新到用户界面
```

## 如何使用

此代码需要分别训练两个模型，在训练平台中按照如下设置发起两次训练，即可得到两个模型。

### 1. 训练Motor检测器

使用如下参数启动训练：

```shell
# 第一个参数10000是训练的steps数量，第二个参数的1表示训练Motor检测器，0表示训练Head检测器
bash /project/train/src_repo/start_train.sh 10000 1
```

训练完成后会生成Motor检测器的模型`ssd_inception_v2_motor.pb`。

### 2. 训练Head检测器

使用如下参数启动训练：

```shell
# 第一个参数10000是训练的steps数量，第二个参数的1表示训练Motor检测器，0表示训练Head检测器
bash /project/train/src_repo/start_train.sh 10000 0
```

训练完成后会生成Motor检测器的模型`ssd_inception_v2_motor.pb`。

