import argparse
import io
import json
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sys
from glob import glob

import cv2
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from pyson.utils import multi_thread, print_source, read_json, timeit
from simple_waymo_open_dataset_reader import (WaymoDataFileReader, dataset_pb2,
                                              label_pb2, utils)
from tqdm import tqdm

from utils import *
debug = os.environ.get('DEBUG', False) == '1'

cmap = matplotlib.cm.get_cmap("viridis")
tf.enable_eager_execution()


if debug:
    os
    output_dir = './tmp'
    os.makedirs("./cache", exist_ok=1)
    tf_paths = glob("/data/waymo/mini_tfrecord/*.tfrecord")
else:
    output_dir = '/ssd6/coco_style_1.2/'
    tf_paths = glob("/toyota/waymo/training_1.2/*tfrecord")

print("tf paths length:", len(tf_paths))
out_laser_dir = output_dir + '/laser_images'
out_image_dir = output_dir + '/images'
out_json_dir = output_dir + '/annotations/output_json'
coco_json_dir = output_dir + "annotations/output_json_coco/"
os.makedirs(coco_json_dir, exist_ok=True)
os.makedirs(out_laser_dir, exist_ok=1)
os.makedirs(out_image_dir, exist_ok=1)
os.makedirs(out_json_dir, exist_ok=1)

parser = argparse.ArgumentParser()
parser.add_argument("--start", "-s", type=int, default=0, )
parser.add_argument("--num_runner", "-n", default=1, type=int)
args = parser.parse_args()

sample = json.load(open('/ssd6/coco_style_1.2/annotations/val.json'))



anno = read_json('/ssd6/coco_style_1.2/annotations/test.json')


def f_frame_data(frame_data):
    frame, frame_id, frame_name = frame_data
    laser_name = dataset_pb2.LaserName.TOP
    laser = utils.get(frame.lasers, laser_name)
    laser_calibration = utils.get(frame.context.laser_calibrations, laser_name)
    ri, camera_projection, range_image_pose = utils.parse_range_image_and_camera_projection(
        laser)
    pcl, pcl_attr = utils.project_to_pointcloud(frame, ri, camera_projection,
                                                range_image_pose,
                                                laser_calibration)
    result = {}
    for camera_name in camera_names:
        # try:
        camera_calibration = utils.get(frame.context.camera_calibrations,
                                    camera_name)
        camera = utils.get(frame.images, camera_name)
        image_name = dataset_pb2.CameraName.Name.Name(camera_name)
        output_name = os.path.join(
            output_dir, 'images', f'{frame_name}_{frame_id}_{image_name}.jpg')
        try:
            img = cv2.imread(output_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            # Decode the image
            img = utils.decode_image(camera)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_name, img)
            # BGR to RGB

        # Get the transformation matrix for the camera.
        vehicle_to_image = utils.get_image_transform(camera_calibration)
        vehicle_to_labels = []
        for label in frame.laser_labels:
            vehicle_to_label = np.linalg.inv(
                utils.get_box_transformation_matrix(label.box))
            vehicle_to_labels.append(vehicle_to_label)

        vehicle_to_labels = np.stack(vehicle_to_labels)
        # Convert the pointcloud to homogeneous coordinates.
        pcl1 = np.concatenate((pcl, np.ones_like(pcl[:, 0:1])), axis=1)
        #device = torch.device(f'cuda:{frame_id%4}')
        #---------torch version
        device = 'cpu'
        torch_vehicle_to_labels = torch.from_numpy(
            vehicle_to_labels.astype(np.float32)).to(device)
        torch_pcl1 = torch.from_numpy(pcl1.astype(np.float32)).to(device)
        proj_pcl = torch.einsum('lij,bj->lbi', torch_vehicle_to_labels, torch_pcl1).cpu().numpy()
        # numpy version
        # proj_pcl = np.einsum('lij,bj->lbi', vehicle_to_labels, pcl1)
        mask = np.logical_and.reduce(np.logical_and(proj_pcl >= -1,
                                                    proj_pcl <= 1),
                                    axis=2)
        # Count the points inside each label's box.
        counts = mask.sum(1)
        # Keep boxes which contain at least 10 LIDAR points.
        visibility = counts > 10
        # Display the LIDAR points on the image.
        laser_as_img = np.zeros_like(img)
        display_laser_on_image(laser_as_img, pcl, vehicle_to_image, pcl_attr)
        output_laser_name = os.path.join(out_laser_dir,
                                        os.path.basename(output_name))
        box_3d_list = get_3d_points(camera_calibration, frame.laser_labels)
        box_2d_list = get_2d_bbox(frame, camera_name)
        cv2.imwrite(output_laser_name, laser_as_img)
        result[output_name] = dict(box_3d_list=box_3d_list,
                                box_2d_list=box_2d_list,
                                counts=counts.tolist())
        # except:
        #     #ignore this frame
        #     print("Error on a frame, ignore")
        #     continue
        # if debug:
        #     display_laser_on_image(img, pcl, vehicle_to_image, pcl_attr)
        #     display_3d_box_on_image(img, box_3d_list, visibility)
        #     display_2d_box_on_image(img, box_2d_list)
        #     debug_path = './cache/' + os.path.basename(output_laser_name)
        #     cv2.imwrite(debug_path, img)
        #     print(debug_path)
    return result





camera_names = [
    dataset_pb2.CameraName.FRONT, dataset_pb2.CameraName.FRONT_LEFT,
    dataset_pb2.CameraName.FRONT_RIGHT, dataset_pb2.CameraName.SIDE_LEFT,
    dataset_pb2.CameraName.SIDE_RIGHT
]

_tf_paths = []
for filename in tf_paths:
    out_json = out_json_dir + "/" + os.path.basename(filename) + '.json'
    output_path = os.path.join(coco_json_dir, os.path.basename(filename))+".json"

    if  os.path.exists(out_json) and os.path.exists(output_path):
        pass 
    else:
        _tf_paths.append(filename)

tf_paths = _tf_paths
# import ipdb; ipdb.set_trace()

print("Total filterd paths: ", len(tf_paths))

length = len(tf_paths) // args.num_runner
start = args.start * length
end = min((args.start + 1) * length, len(tf_paths))


tf_paths = tf_paths[start:end]

for p_i, filename in enumerate(tf_paths):
    print('-----------------------------------')
    # convert TF->JSON
    print("process", start + p_i, "/", end)
    out_json = out_json_dir + "/" + os.path.basename(filename) + '.json'
    READ_FRAMES=False
    if not os.path.exists(out_json):
        # continue # ignore for debug
        out = dict()
        frames_data = f_datapath(filename)
        READ_FRAMES = True
        print("TF->JSON not exists, MULTITHREAD:", out_json)
        outs = multi_thread(f_frame_data,
                            frames_data,
                            verbose=1,
                            max_workers=4)
        for _ in outs:
            out.update(_)
        with open(out_json, 'w') as f:
            json.dump(out, f)


    # Convert to coco_format
    output_path = os.path.join(coco_json_dir, os.path.basename(filename))+".json"
    if os.path.exists(output_path):
        print("JSON->COCO exists, CONTINUE", output_path)
        continue
    else:
        
        try:
            print("TF->JSON exists, READ:", out_json)
            out = read_json(out_json)
        except:
            print("Cannot read-> remove json", out_json)
            os.remove(out_json)
            continue
            
        if not READ_FRAMES:
            frames_data = f_datapath(filename)

    print('Converting JSON->COCO....')
    images = []
    annotations = []
    image_id = 0
    labels_3d = get_3d_label(frames_data)
    for _, (key, value) in enumerate(out.items()):
        value['cates'] = labels_3d[key]['type_id_list']
        camera_calibration = labels_3d[key]['camera_calibration']
        images.append(image_to_coco_dict(key, image_id, camera_calibration))
        annos = annotation_to_dict(value, image_id)
        for anno in annos:
            annotations.append(anno)
        image_id += 1

    sample['images'] = images
    sample['annotations'] = annotations
    with open(output_path, "w") as f:
        json.dump(sample, f)

