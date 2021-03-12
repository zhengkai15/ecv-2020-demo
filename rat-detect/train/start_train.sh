#!/bin/bash

project_root_dir=/project/train/src_repo
dataset_dir=/home/data
log_file=/project/train/log/log.txt
if [ ! -z $1 ]; then
    num_train_steps=$1
else
    num_train_steps=100
fi
cd ${project_root_dir} && git pull
pip install -i https://mirrors.aliyun.com/pypi/simple -r /project/train/src_repo/requirements.txt \
&& echo "Preparing..." \
&& cd ${project_root_dir}/tf-models/research/ && protoc object_detection/protos/*.proto --python_out=. \
&& echo "Converting dataset..." \
&& python3 -u ${project_root_dir}/convert_dataset.py ${dataset_dir} | tee -a ${log_file} \
&& echo "Start update plot..." \
&& cd ${project_root_dir} && sh -c "python3 update_plots.py &" \
&& echo "Start training..." \
&& cd ${project_root_dir} && python3 -u train.py --logtostderr --model_dir=training/ --pipeline_config_path=pipeline-config/ssd_mobilenet_v2_rat_detector.config --num_train_steps ${num_train_steps} 2>&1 | tee -a ${log_file} \
&& echo "Done" \
&& echo "Exporting and saving models to /project/train/models..." \
&& python3 -u ${project_root_dir}/export_models.py --ckpt_dir=training --output_model_prefix=ssd_mobilenet_v2 | tee -a ${log_file} \
&& echo "Saving plots..." \
&& python3 -u ${project_root_dir}/save_plots.py | tee -a ${log_file}