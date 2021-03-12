#!/bin/bash

# Root directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source /opt/intel/openvino/deployment_tools/model_optimizer/venv/bin/activate

# Convert motor model
/opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
--input_model ${SCRIPT_DIR}/ssd_inception_v2_motor.pb \
--transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json \
--tensorflow_object_detection_api_pipeline_config ${SCRIPT_DIR}/ssd_inception_v2_motor.config \
--output_dir ${SCRIPT_DIR}/openvino \
--model_name ssd_inception_v2_motor \
--input image_tensor

# Convert head model
/opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
--input_model ${SCRIPT_DIR}/ssd_mobilenet_v2_head.pb \
--transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json \
--tensorflow_object_detection_api_pipeline_config ${SCRIPT_DIR}/ssd_mobilenet_v2_head.config \
--output_dir ${SCRIPT_DIR}/openvino \
--model_name ssd_mobilenet_v2_head \
--input image_tensor