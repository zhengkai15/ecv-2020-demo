#!/bin/bash

# Root directory of the script
source /opt/intel/openvino/deployment_tools/model_optimizer/venv/bin/activate
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

/opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
--input_model ${SCRIPT_DIR}/ssd_inception_v2.pb \
--transformations_config /opt/intel/openvino_2020.1.023/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json \
--tensorflow_object_detection_api_pipeline_config ${SCRIPT_DIR}/ssd_inception_v2_coco.config \
--output_dir ${SCRIPT_DIR}/openvino \
--model_name ssd_inception_v2 \
--input image_tensor