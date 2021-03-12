FROM uhub.service.ucloud.cn/eagle_nest/cuda10.0-cudnn7.4.2-dev-ubuntu16.04-opencv4.1.1-tensorflow1.13-openvino2020r2

# 卸载tensorflow
RUN rm -rf /opt/tensorflow && pip uninstall --yes tensorflow-gpu
RUN rm -rf /usr/local/ev_sdk && mkdir -p /usr/local/ev_sdk
# 将所有代码拷贝到镜像中对应路径
COPY ./ /usr/local/ev_sdk

#　在镜像中编译，安装sdk
RUN \
    cd /usr/local/ev_sdk && mkdir -p build && rm -rf build/* \
    && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j4 install && rm -rf ../build/*

# 如果使用python接口，需要设置此环境变量，如果使用c++接口，则不需要
ENV AUTO_TEST_USE_JI_PYTHON_API=1