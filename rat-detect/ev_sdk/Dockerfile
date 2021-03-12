FROM uhub.service.ucloud.cn/eagle_nest/cuda10.0-cudnn7.4.2-dev-ubuntu16.04-opencv4.1.1-tensorflow1.13-openvino2020r1

RUN rm -rf /usr/local/ev_sdk && mkdir -p /usr/local/ev_sdk
COPY ./ /usr/local/ev_sdk

RUN \
    cd /usr/local/ev_sdk && mkdir -p build && rm -rf build/* \
    && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DUSE_PYTHON_API=OFF .. && make -j4 install && rm -rf ../build/*

# Download pretrained ssd mobilenet v2
RUN wget http://10.9.0.146:8888/group1/M00/00/AD/CgkAkl8GloaECYa_AAAAAKzBYhM6601.pb -O /usr/local/ev_sdk/model/ssd_mobilenet_v2.pb

# 如果使用TensorFlow镜像发起OpenVINO模型测试时，为了避免tensorflow-gpu和OpenVINO内部安装的CPU版tensorflow冲突，需要把tensorflow-gpu删除
RUN rm -rf /opt/tensorflow && pip uninstall --yes tensorflow-gpu

# 如果使用Python接口编写代码需要添加一个环境变量
# ENV AUTO_TEST_USE_JI_PYTHON_API=1