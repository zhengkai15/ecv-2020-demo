#!/bin/bash

if [[ ! $# -eq 2 ]]; then
    echo "Usage: bash start_train.sh 10000 0"
    exit -1
fi

num_steps=$1
# 1表示训练motor检测器，0表示训练head检测器
train_motor_detector=$2

project_root_dir=/project/train/src_repo
dataset_dir=/home/data
log_file=/project/train/log/log.txt

pip install -i https://mirrors.aliyun.com/pypi/simple -r /project/train/src_repo/requirements.txt \
&& echo "Preparing..." \
&& cd ${project_root_dir}/tf-models/research/ && protoc object_detection/protos/*.proto --python_out=.
if [[ ${train_motor_detector} -eq 1 ]]; then
    # Train motor detector
    echo "Converting dataset..." \
    && python3 -u ${project_root_dir}/convert_dataset_for_motor_detector.py ${dataset_dir} | tee -a ${log_file}
    echo "Start update plot..." \
    && cd ${project_root_dir} && sh -c "python3 update_plots.py &" \
    && echo "Start training..." \
    && cd ${project_root_dir} && python3 -u train.py --logtostderr --model_dir=training/ --pipeline_config_path=pipeline-config/ssd_inception_v2_motor_detector.config 2>&1 | tee -a ${log_file} \
    && echo "Done" \
    && echo "Exporting and saving models to /project/train/models..." \
    && python3 -u ${project_root_dir}/export_models.py --ckpt_dir=training --output_model_prefix=ssd_inception_v2_motor  | tee -a ${log_file} \
    && echo "Saving plots..." \
    && python3 -u ${project_root_dir}/save_plots.py | tee -a ${log_file}
else
    # Train head inside motor detector
    echo "Converting dataset..." \
    && python3 -u ${project_root_dir}/convert_dataset_for_head_detector.py ${dataset_dir} | tee -a ${log_file}
    echo "Start update plot..." \
    && cd ${project_root_dir} && sh -c "python3 update_plots.py &" \
    && echo "Start training..." \
    && cd ${project_root_dir} && python3 -u train.py --logtostderr --model_dir=training/ --pipeline_config_path=pipeline-config/ssd_mobilenet_v2_head_detector.config 2>&1 | tee -a ${log_file} \
    && echo "Done" \
    && echo "Exporting and saving models to /project/train/models..." \
    && python3 -u ${project_root_dir}/export_models.py --ckpt_dir=training --output_model_prefix=ssd_mobilenet_v2_head | tee -a ${log_file} \
    && echo "Saving plots..." \
    && python3 -u ${project_root_dir}/save_plots.py | tee -a ${log_file}
fi
