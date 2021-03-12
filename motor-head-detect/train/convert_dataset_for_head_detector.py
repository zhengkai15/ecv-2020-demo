from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import shutil
import pathlib
import random

import PIL
from PIL import Image
import xml.etree.ElementTree as ET
import pandas as pd
import tensorflow as tf
import io
from collections import namedtuple, OrderedDict
import copy

from global_config import *
from object_detection.utils import dataset_util

train_data_dir = os.path.join(project_root, 'dataset/images/train/')
valid_data_dir = os.path.join(project_root, 'dataset/images/valid')
annotations_dir = os.path.join(project_root, 'dataset/annotations')
supported_fmt = ['.jpg', '.JPG']


"""用于训练检测Motor内部的head
1. 把所有Motor bbox裁剪出来，并保存到目录；
2. 对所有Motor bbox内部的head生成tfrecord；
"""


def class_text_to_int(row_label):
    if row_label == 'head':  # 标注文件里面的标签名称
        return 1
    elif row_label == 'helmet':
        return 2
    elif row_label == 'motor':
        return 3
    else:
        return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def inside_rect(rect1, rect2, overlap_threshold=1.0):
    left = max(rect1['xmin'], rect2['xmin'])
    top = max(rect1['ymin'], rect2['ymin'])
    right = min(rect1['xmax'], rect2['xmax'])
    bottom = min(rect1['ymax'], rect2['ymax'])
    if left >= right or top >= bottom:
        return False
    overlap_ratio = (right - left) * (bottom - top) / ((rect1['xmax'] - rect1['xmin']) * (rect1['ymax'] - rect1['ymin']))

    return 1 >= overlap_ratio >= overlap_threshold


def csv_to_record(output_path, img_path, csv_input):
    writer = tf.python_io.TFRecordWriter(output_path)
    path = os.path.join(os.getcwd(), img_path)
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


def xml_to_csv(data_list, motor_image_output_dir=None):
    """将data_list表示的(图片, 标签)对转换成pandas.Dataframe记录
    """
    os.makedirs(motor_image_output_dir, exist_ok=True)
    xml_list = []
    for i, data in enumerate(data_list):
        tree = ET.parse(data['label'])
        root = tree.getroot()
        try:
            img = Image.open(data['image'])
        except (FileNotFoundError, PIL.UnidentifiedImageError):
            print(f'打开{data["image"]}出错!')
            continue
        width, height = img.size

        # Find all objects
        motors = []
        heads = []
        objects = []
        for member in root.findall('object'):
            name = member.find('name')
            if name is None or name.text == '':
                print(f'empty name, skipped')
                continue
            name = name.text
            bndbox = member.find('bndbox')
            if bndbox is None or bndbox == '':
                print(f'empty bndbox, skipped')
                continue
            xmin_obj = bndbox.find('xmin')
            ymin_obj = bndbox.find('ymin')
            xmax_obj = bndbox.find('xmax')
            ymax_obj = bndbox.find('ymax')
            xmin, ymin, xmax, ymax = int(float(xmin_obj.text)), int(float(ymin_obj.text)), int(float(xmax_obj.text)), int(float(ymax_obj.text))
            objects.append({
                'name': name,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })
        for obj in objects:
            if obj['name'] == 'motor':
                motors.append(copy.deepcopy(obj))
            elif obj['name'] == 'head':
                heads.append(copy.deepcopy(obj))

        if len(motors) <= 0:
            print(f'count:{i},  empty motors in :{data["label"]}, skipped')
            continue
        # Cut out motor and save to image
        for i, motor in enumerate(motors):
            w, h = motor['xmax'] - motor['xmin'], motor['ymax'] - motor['ymin']
            if w > h:
                target_w, target_h = 300, int(300.0 / w * h)
            else:
                target_w, target_h = int(300.0 / h * w), 300
            resize_ratio = 300.0 / max(w, h)
            print(f'result target size: {(target_w, target_h)}, resize_ratio:{resize_ratio}')
            expand_x = 20
            expand_y = 20
            motor_box = (max(0, motor['xmin'] - expand_x), max(0, motor['ymin'] - expand_y), min(motor['xmax'] + expand_x, width), min(motor['ymax'] + expand_y, height))
            # motor_img = img.resize((target_w, target_h), PIL.Image.FLOYDSTEINBERG, box=(motor['xmin'], motor['ymin'], motor['xmax'], motor['ymax']))
            # motor_img = img.resize((target_w, target_h), PIL.Image.FLOYDSTEINBERG, box=motor_box)
            motor_img = img.crop(motor_box)

            motor_box_w, motor_box_h = motor_box[2] - motor_box[0], motor_box[3] - motor_box[1]

            orig_name = pathlib.Path(data['image'])
            # Output cropped motor image
            target_image_name = os.path.join(motor_image_output_dir, orig_name.stem + f'_motor_{i}.png')
            motor_img.save(target_image_name)
            print(f'{target_image_name} saved')

            # Save xml
            xml_filename = pathlib.Path(target_image_name).with_suffix('.xml').as_posix()
            xml_data = ET.Element('annotation')
            for head in heads:
                if inside_rect(head, motor, 0.99):
                    head_resized = copy.deepcopy(head)
                    print(f'{data["image"]}, {head_resized} , {motor}')
                    # Apply cut from original image
                    head_resized['xmin'] -= motor_box[0]
                    head_resized['ymin'] -= motor_box[1]
                    head_resized['xmax'] -= motor_box[0]
                    head_resized['ymax'] -= motor_box[1]

                    if head_resized['xmin'] < 0:
                        head_resized['xmin'] = 0
                    if head_resized['ymin'] < 0:
                        head_resized['ymin'] = 0
                    if head_resized['xmax'] > motor_box_w:
                        head_resized['xmax'] = motor_box_w
                    if head_resized['ymax'] > motor_box_h:
                        head_resized['ymax'] = motor_box_h
                    assert head_resized['xmin'] >= 0 and head_resized['ymin'] >= 0 and head_resized['xmax'] >= 0 and head_resized['ymax'] >= 0

                    value = (target_image_name,
                             target_w,
                             target_h,
                             head_resized['name'],
                             head_resized['xmin'],
                             head_resized['ymin'],
                             head_resized['xmax'],
                             head_resized['ymax'])
                    obj_data = ET.SubElement(xml_data, 'object')
                    name_obj = ET.SubElement(obj_data, 'name')
                    name_obj.text = head['name']
                    xmin_obj = ET.SubElement(obj_data, 'xmin')
                    ymin_obj = ET.SubElement(obj_data, 'ymin')
                    xmax_obj = ET.SubElement(obj_data, 'xmax')
                    ymax_obj = ET.SubElement(obj_data, 'ymax')
                    xmin_obj.text = str(head_resized['xmin'])
                    ymin_obj.text = str(head_resized['ymin'])
                    xmax_obj.text = str(head_resized['xmax'])
                    ymax_obj.text = str(head_resized['ymax'])

                    xml_list.append(value)
            xml_string = ET.tostring(xml_data, encoding='unicode')
            with open(xml_filename, 'w') as f:
                f.write(xml_string)
        img.close()

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


if __name__ == '__main__':
    os.makedirs(project_root, exist_ok=True)
    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(valid_data_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    if not os.path.exists(sys.argv[1]):
        print(f'{sys.argv[1]} 不存在!')
        exit(-1)

    # 遍历数据集目录下所有xml文件及其对应的图片
    dataset_path = pathlib.Path(sys.argv[1])
    found_data_list = []
    for xml_file in dataset_path.glob('**/*.xml'):
        possible_images = [xml_file.with_suffix(suffix) for suffix in supported_fmt]
        supported_images = list(filter(lambda p: p.is_file(), possible_images))
        if len(supported_images) == 0:
            print(f'找不到对应的图片文件：`{xml_file.as_posix()}`')
            continue
        found_data_list.append({'image': supported_images[0], 'label': xml_file})

    # 随机化数据集，将数据集拆分成训练集和验证集，并将其拷贝到/project/train/src_repo/dataset下
    random.shuffle(found_data_list)
    train_data_count = len(found_data_list) * 4 / 5
    train_data_list = []
    valid_data_list = []
    for i, data in enumerate(found_data_list):
        if i < train_data_count:  # 训练集
            dst = train_data_dir
            data_list = train_data_list
        else:  # 验证集
            dst = valid_data_dir
            data_list = valid_data_list
        image_dst = (pathlib.Path(dst) / data['image'].name).as_posix()
        label_dst = (pathlib.Path(dst) / data['label'].name).as_posix()
        shutil.copy(data['image'].as_posix(), image_dst)
        shutil.copy(data['label'].as_posix(), label_dst)
        data_list.append({'image': image_dst, 'label': label_dst})

    # 将XML转换成CSV格式
    train_xml_df = xml_to_csv(train_data_list, motor_image_output_dir=os.path.join(project_root, 'dataset/motor/train'))
    train_xml_df.to_csv(os.path.join(annotations_dir, 'train_labels.csv'), index=False)
    valid_xml_df = xml_to_csv(valid_data_list, motor_image_output_dir=os.path.join(project_root, 'dataset/motor/valid'))
    valid_xml_df.to_csv(os.path.join(annotations_dir, 'valid_labels.csv'), index=False)
    print('Successfully converted xml to csv.')

    # 将数据集转换成tf record格式
    csv_to_record(os.path.join(annotations_dir, 'train.record'), train_data_dir,
                  os.path.join(annotations_dir, 'train_labels.csv'))
    csv_to_record(os.path.join(annotations_dir, 'valid.record'), valid_data_dir,
                  os.path.join(annotations_dir, 'valid_labels.csv'))

    #
    with open(os.path.join(annotations_dir, 'label_map.pbtxt'), 'w') as f:
        label_map = """
item {
    id: 1
    name: "head"
}
"""
        f.write(label_map)
