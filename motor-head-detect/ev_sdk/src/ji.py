from __future__ import print_function

import logging as log
import os
import pathlib
import json
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

device = 'CPU'
motor_input_h, motor_input_w, motor_input_c, motor_input_n = (300, 300, 3, 1)
head_input_h, head_input_w, head_input_c, head_input_n = (200, 200, 3, 1)
log.basicConfig(level=log.DEBUG)

# For objection detection task, replace your target labels here.
motor_label_id_map = {
    1: "motor"
}
head_label_id_map = {
    1: "head"
}

motor_exec_net = None
head_exec_net = None

motor_net = None
head_net = None

motor_input_blob = None
head_input_blob = None

motor_output_dims = None
head_output_dims = None

def inside_rect(rect1, rect2, overlap_threshold=1.0):
    left = max(rect1.x, rect2.x)
    top = max(rect1.y, rect2.y)
    right = min(rect1.x, rect2.x)
    bottom = min(rect1.y, rect2.y)
    if left >= right or top >= bottom:
        return False
    overlap_ratio = (right - left) * (bottom - top) / (rect1.width * rect2.height)

    return overlap_threshold <= overlap_ratio <= 1.0


def init():
    """Initialize model

    Returns: model

    """
    # Load network
    motor_model_xml = "/usr/local/ev_sdk/model/openvino/ssd_inception_v2_motor.xml"
    head_model_xml = "/usr/local/ev_sdk/model/openvino/ssd_mobilenet_v2_head.xml"
    if not os.path.isfile(motor_model_xml):
        log.error(f'{motor_model_xml} does not exist')
        return None
    if not os.path.isfile(head_model_xml):
        log.error(f'{head_model_xml} does not exist')
        return None
    motor_model_bin = pathlib.Path(motor_model_xml).with_suffix('.bin').as_posix()
    head_model_bin = pathlib.Path(head_model_xml).with_suffix('.bin').as_posix()
    log.info(f"Loading network files: motor model:\n\t{motor_model_xml}\n\t{motor_model_bin}\n\thead model:\n\t{head_model_xml}\n\t{head_model_bin}")
    motor_net = IENetwork(model=motor_model_xml, weights=motor_model_bin)
    head_net = IENetwork(model=head_model_xml, weights=head_model_bin)

    # Setup output info
    global motor_output_blob
    global motor_output_dims
    motor_output_blob = next(iter(motor_net.outputs))
    motor_output_info = motor_net.outputs['DetectionOutput']
    motor_output_dims = motor_output_info.shape
    motor_output_info.precision = "FP32"

    global head_output_blob
    global head_output_dims
    head_output_blob = next(iter(head_net.outputs))
    head_output_info = head_net.outputs['DetectionOutput']
    head_output_dims = head_output_info.shape
    head_output_info.precision = "FP32"

    # Load Inference Engine
    log.info('Loading Inference Engine')
    ie = IECore()
    global motor_exec_net
    global head_exec_net
    motor_exec_net = ie.load_network(network=motor_net, device_name=device)
    head_exec_net = ie.load_network(network=head_net, device_name=device)

    log.info('Device info:')
    versions = ie.get_versions(device)
    print("{}".format(device))
    print("MKLDNNPlugin version ......... {}.{}".format(versions[device].major, versions[device].minor))
    print("Build ........... {}".format(versions[device].build_number))

    global motor_input_blob
    global motor_input_h, motor_input_w, motor_input_c, motor_input_n
    motor_input_blob = next(iter(motor_net.inputs))
    n, c, h, w = motor_net.inputs[motor_input_blob].shape
    motor_input_h, motor_input_w, motor_input_c, motor_input_n = h, w, c, n

    global head_input_blob
    global head_input_h, head_input_w, head_input_c, head_input_n
    head_input_blob = next(iter(head_net.inputs))
    n, c, h, w = head_net.inputs[head_input_blob].shape
    head_input_h, head_input_w, head_input_c, head_input_n = h, w, c, n

    return motor_net


def process_image(net, input_image):
    """Do inference to analysis input_image and get output

    Attributes:
        net: model handle
        input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR

    Returns: process result

    """
    if input_image is None:
        log.error('Invalid input args')
        return None
    log.info(f'process_image, ({input_image.shape}')
    ih, iw, _ = input_image.shape

    # --------------------------- Prepare input blobs -----------------------------------------------------
    motor_input_image = input_image.copy()
    if ih != motor_input_h or iw != motor_input_w:
        motor_input_image = cv2.resize(motor_input_image, (motor_input_w, motor_input_h))
    motor_input_image = motor_input_image.transpose((2, 0, 1))
    images = np.ndarray(shape=(motor_input_n, motor_input_c, motor_input_h, motor_input_w))
    images[0] = motor_input_image

    # --------------------------- Prepare output blobs ----------------------------------------------------
    log.info('Preparing output blobs')

    # --------------------------- Performing inference ----------------------------------------------------
    log.info("Creating infer request and starting inference")
    res = motor_exec_net.infer(inputs={motor_input_blob: images})

    # --------------------------- Read and postprocess output ---------------------------------------------
    log.info("Processing output blobs")
    res = res[motor_output_blob]
    data = res[0][0]

    motors = []
    for proposal in data:
        batch_id = np.int(proposal[0])
        if int(batch_id) >= 0:
            label = np.int(proposal[1])
            confidence = proposal[2]
            xmin = np.int(iw * proposal[3])
            ymin = np.int(ih * proposal[4])
            xmax = np.int(iw * proposal[5])
            ymax = np.int(ih * proposal[6])
            if label not in motor_label_id_map:
                log.warning(f'{label} does not in {motor_label_id_map}')
                continue
            if confidence > 0.2:
                obj = {
                    'name': motor_label_id_map[label],
                    'xmin': int(xmin),
                    'ymin': int(ymin),
                    'xmax': int(xmax),
                    'ymax': int(ymax),
                    'confidence': float(confidence)
                }
                if motor_label_id_map[label] == 'motor':
                    motors.append(obj)
    log.info(f'Found motors:{motors}')
    # -------------------------- Find all head inside motors ----------------------------
    # Filter all head inside motor
    heads = []
    for i, motor in enumerate(motors):
        # Prepare motor image
        # Expanded motor rect by 10 pixels
        expanded_xmin = max(0, motor['xmin'] - 10)
        expanded_ymin = max(0, motor['ymin'] - 10)
        expanded_xmax = min(iw, motor['xmax'] + 10)
        expanded_ymax = min(ih, motor['ymax'] + 10)
        expanded_img = input_image[expanded_ymin:expanded_ymax, expanded_xmin:expanded_xmax]
        cv2.imwrite(f'motor_{i}.jpg', expanded_img)
        motor_h, motor_w, _ = expanded_img.shape
        if motor_h != head_input_h or motor_w != head_input_w:
            expanded_img = cv2.resize(expanded_img, (head_input_w, head_input_h))
        expanded_img = expanded_img.transpose((2, 0, 1))
        input_motor_images = np.ndarray(shape=(head_input_n, head_input_c, head_input_h, head_input_w))
        input_motor_images[0] = expanded_img

        # Inference
        res = head_exec_net.infer(inputs={head_input_blob: input_motor_images})

        # Get output
        res = res[head_output_blob]
        data = res[0][0]
        for proposal in data:
            batch_id = np.int(proposal[0])
            if batch_id >= 0:
                label_id = np.int(proposal[1])
                confidence = proposal[2]
                xmin = np.int(motor_w * proposal[3])
                ymin = np.int(motor_h * proposal[4])
                xmax = np.int(motor_w * proposal[5])
                ymax = np.int(motor_h * proposal[6])
                if label_id not in head_label_id_map:
                    log.warning(f'{label_id} does not in {head_label_id_map}')
                    continue
                if confidence > 0.2:
                    obj = {
                        'name': head_label_id_map[label_id],
                        'xmin': int(xmin) + expanded_xmin,
                        'ymin': int(ymin) + expanded_ymin,
                        'xmax': int(xmax) + expanded_xmin,
                        'ymax': int(ymax) + expanded_ymin,
                        'confidence': float(confidence)
                    }
                    if head_label_id_map[label_id] == 'head':
                        heads.append(obj)


    return json.dumps({"objects": heads})


if __name__ == '__main__':
    # Test API
    img = cv2.imread('/home/data/20/MotorHTL20200610_216.jpg')
    predictor = init()
    result = process_image(predictor, img)
    log.info(result)
