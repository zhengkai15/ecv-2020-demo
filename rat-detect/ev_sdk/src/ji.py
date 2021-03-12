from __future__ import print_function

import logging as log
import os
import pathlib
import json
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

device = 'CPU'
input_h, input_w, input_c, input_n = (300, 300, 3, 1)
log.basicConfig(level=log.DEBUG)

# For objection detection task, replace your target labels here.
label_id_map = {
    1: "rat",
}
exec_net = None


def init():
    """Initialize model

    Returns: model

    """
    model_xml = "/usr/local/ev_sdk/model/openvino/ssd_mobilenet_v2.xml"
    if not os.path.isfile(model_xml):
        log.error(f'{model_xml} does not exist')
        return None
    model_bin = pathlib.Path(model_xml).with_suffix('.bin').as_posix()
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    # Load Inference Engine
    log.info('Loading Inference Engine')
    ie = IECore()
    global exec_net
    exec_net = ie.load_network(network=net, device_name=device)
    log.info('Device info:')
    versions = ie.get_versions(device)
    print("{}".format(device))
    print("MKLDNNPlugin version ......... {}.{}".format(versions[device].major, versions[device].minor))
    print("Build ........... {}".format(versions[device].build_number))

    input_blob = next(iter(net.inputs))
    n, c, h, w = net.inputs[input_blob].shape
    global input_h, input_w, input_c, input_n
    input_h, input_w, input_c, input_n = h, w, c, n

    return net


def process_image(net, input_image):
    """Do inference to analysis input_image and get output

    Attributes:
        net: model handle
        input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR

    Returns: process result

    """
    if not net or input_image is None:
        log.error('Invalid input args')
        return None
    log.info(f'process_image, ({input_image.shape}')
    ih, iw, _ = input_image.shape

    # --------------------------- Prepare input blobs -----------------------------------------------------
    if ih != input_h or iw != input_w:
        input_image = cv2.resize(input_image, (input_w, input_h))
    input_image = input_image.transpose((2, 0, 1))
    images = np.ndarray(shape=(input_n, input_c, input_h, input_w))
    images[0] = input_image

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    # --------------------------- Prepare output blobs ----------------------------------------------------
    log.info('Preparing output blobs')

    output_name = "DetectionOutput"
    try:
        output_info = net.outputs[output_name]
    except KeyError:
        log.error(f"Can't find a {output_name} layer in the topology")
        return None

    output_dims = output_info.shape
    if len(output_dims) != 4:
        log.error("Incorrect output dimensions for SSD model")
    max_proposal_count, object_size = output_dims[2], output_dims[3]

    if object_size != 7:
        log.error("Output item should have 7 as a last dimension")

    output_info.precision = "FP32"

    # --------------------------- Performing inference ----------------------------------------------------
    log.info("Creating infer request and starting inference")
    res = exec_net.infer(inputs={input_blob: images})

    # --------------------------- Read and postprocess output ---------------------------------------------
    log.info("Processing output blobs")
    res = res[out_blob]
    data = res[0][0]

    detect_objs = []
    for proposal in data:
        if proposal[2] > 0:
            batch_id = np.int(proposal[0])
            label = np.int(proposal[1])
            confidence = proposal[2]
            xmin = np.int(iw * proposal[3])
            ymin = np.int(ih * proposal[4])
            xmax = np.int(iw * proposal[5])
            ymax = np.int(ih * proposal[6])
            if label not in label_id_map:
                log.warning(f'{label} does not in {label_id_map}')
                continue
            if proposal[2] > 0.6:
                detect_objs.append({
                    'name': label_id_map[label],
                    'xmin': int(xmin),
                    'ymin': int(ymin),
                    'xmax': int(xmax),
                    'ymax': int(ymax),
                    'confidence': float(confidence)
                })
    return json.dumps({"objects": detect_objs})


if __name__ == '__main__':
    # Test API
    img = cv2.imread('/usr/local/ev_sdk/data/dog.jpg')
    predictor = init()
    result = process_image(predictor, img, 0.5)
    log.info(result)
