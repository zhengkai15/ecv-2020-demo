import os
from global_config import *

import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2
import pathlib
import shutil
import subprocess

import sys

sys.path.append('/opt/intel/openvino_2020.1.023/deployment_tools/model_optimizer/')
import mo_tf

model_save_dir = '/project/train/models'
model_prefix = 'ssd_inception_v2'

if __name__ == '__main__':
    # model_dir为保存checkpoint的路径
    model_dir = pathlib.Path(os.path.join(project_root, 'training'))
    ckpts = model_dir.glob('model.ckpt-*.meta')
    ckpt_list = []
    for ckpt in ckpts:
        ckpt_list.append({
            'ckpt_num': ckpt.stem.split('-')[1],
            'ckpt_name': ckpt.stem
        })
        print(f'Found ckpt:{ckpt_list[-1]}')
    sorted(ckpt_list, key=lambda x: int(x['ckpt_num']))
    for i, ckpt in enumerate(ckpt_list):
        # 创建模型保存路径/project/train/model/step{ckpt_num}
        if i == (len(ckpt_list) - 1):
            save_dir = os.path.join(model_save_dir, 'final')
        else:
            save_dir = os.path.join(model_save_dir, f"step{ckpt['ckpt_num']}")
        os.makedirs(save_dir, exist_ok=True)
        # 导出模型
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.gfile.GFile(os.path.join(project_root, 'pre-trained-model/ssd_inception_v2_coco.config'), 'r') as f:
            text_format.Merge(f.read(), pipeline_config)
        text_format.Merge('', pipeline_config)
        input_shape = None
        input_type = 'image_tensor'
        export_dir = os.path.join(project_root, 'exported_model')
        if os.path.isdir(export_dir):
            shutil.rmtree(export_dir)
        # 将所有保存在checkpoint路径下的模型导出到/project/train/models路径下
        exporter.export_inference_graph(input_type, pipeline_config,
                                        os.path.join(project_root, 'training', ckpt['ckpt_name']),
                                        export_dir, input_shape=input_shape)
        # 将模型保存到指定位置
        shutil.copy(src=os.path.join(export_dir, 'frozen_inference_graph.pb'),
                    dst=os.path.join(os.path.join(save_dir, f'{model_prefix}.pb')))
        tf.reset_default_graph()
        print('Saved.')
