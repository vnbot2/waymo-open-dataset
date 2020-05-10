import torch

from pyson.utils import print_source, multi_thread, read_json
import mmcv

import numpy as np
import math
import cv2
import io
# from waymo_open_dataset import dataset_pb2 as open_dataset
import sys
from tqdm import tqdm
import json
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils
from utils import *
import matplotlib.cm
from glob import glob
import argparse
from pyson.utils import read_json
# os.environ['CUDA_VISIBLE_DEVICES']='0'

debug = os.environ.get('DEBUG', False) == '1'

cmap = matplotlib.cm.get_cmap("viridis")
tf.enable_eager_execution()

# tf.config.gpu.set_per_process_memory_fraction(0.75)
# tf.config.experimental.set_per_process_memory_growth(True)

data_paths_train =  glob('/ssd6/waymo/tfrecord_train/*.tfrecord')#[_ for _ in]# if not _ in data_paths_val]
if debug:
    os
    output_dir = 'outdir'
    os.makedirs("./cache")
else:
    output_dir = '/ssd6/coco_style_1.2/'
out_laser_dir = output_dir + '/laser_images'
out_image_dir = output_dir + '/images'
out_json_dir = output_dir + '/annotations/output_json'
os.makedirs(out_laser_dir, exist_ok=1)
os.makedirs(out_image_dir, exist_ok=1)
os.makedirs(out_json_dir, exist_ok=1)


parser = argparse.ArgumentParser()
parser.add_argument("--start","-s", type=int)
args = parser.parse_args()

def f_datapath(data_path):
    f_name = os.path.basename(data_path)
    frame_name = os.path.basename(data_path)
    frame_id = 0
    dataset = tf.data.TFRecordDataset(data_path, compression_type='')
    frames = []
    for data in dataset:
        frame_id += 1
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frames.append((frame, frame_id, f_name))
    return frames

anno = read_json('/ssd6/coco_style_1.2/annotations/test.json')

from pyson.utils import timeit


def f_frame_data(frame_data):
    frame, frame_id, frame_name = frame_data
    laser_name = dataset_pb2.LaserName.TOP
    laser = utils.get(frame.lasers, laser_name)
    laser_calibration = utils.get(frame.context.laser_calibrations, laser_name)
    ri, camera_projection, range_image_pose = utils.parse_range_image_and_camera_projection(laser)
    pcl, pcl_attr = utils.project_to_pointcloud(frame, ri, camera_projection, range_image_pose, laser_calibration)
    result = {}
    for camera_name in camera_names:
        camera_calibration = utils.get(frame.context.camera_calibrations, camera_name)
        camera = utils.get(frame.images, camera_name)
        image_name = dataset_pb2.CameraName.Name.Name(camera_name)
        output_name = os.path.join(output_dir, 'images', f'{frame_name}_{frame_id}_{image_name}.jpg')
        if os.path.exists(output_name):
            img = cv2.imread(output_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Decode the image
            img = utils.decode_image(camera)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_name, img)
            # BGR to RGB

        
         # Get the transformation matrix for the camera.
        vehicle_to_image = utils.get_image_transform(camera_calibration)
        vehicle_to_labels = []
        for label in frame.laser_labels:
            vehicle_to_label = np.linalg.inv(utils.get_box_transformation_matrix(label.box))
            vehicle_to_labels.append(vehicle_to_label)


        vehicle_to_labels = np.stack(vehicle_to_labels)
        # Convert the pointcloud to homogeneous coordinates.
        pcl1 = np.concatenate((pcl,np.ones_like(pcl[:,0:1])),axis=1)
        device = torch.device(f'cuda:{frame_id%4}')
        torch_vehicle_to_labels = torch.from_numpy(vehicle_to_labels.astype(np.float32)).to(device)
        torch_pcl1 = torch.from_numpy(pcl1.astype(np.float32)).to(device)
        proj_pcl = torch.einsum('lij,bj->lbi', torch_vehicle_to_labels, torch_pcl1).cpu().numpy()
        mask = np.logical_and.reduce(np.logical_and(proj_pcl >= -1, proj_pcl <= 1),axis=2)
        # Count the points inside each label's box.
        counts = mask.sum(1)
        # Keep boxes which contain at least 10 LIDAR points.
        visibility = counts > 10

        # Display the LIDAR points on the image.
        laser_as_img = np.zeros_like(img)
        display_laser_on_image(laser_as_img, pcl, vehicle_to_image, pcl_attr)
        output_laser_name = os.path.join(out_laser_dir, os.path.basename(output_name))
        box_3d_list = get_3d_points(camera_calibration, frame.laser_labels)
        box_2d_list = get_2d_bbox(frame, camera_name)
        # debug


        # if not os.path.exists(output_laser_name):
        cv2.imwrite(output_laser_name, laser_as_img)
        # print(output_laser_name)
        result[output_name] = dict(box_3d_list=box_3d_list, box_2d_list=box_2d_list,
                                   visibility=visibility.tolist())
        if debug:
            display_laser_on_image(img, pcl, vehicle_to_image, pcl_attr)
            display_3d_box_on_image(img, box_3d_list, visibility)
            display_2d_box_on_image(img, box_2d_list)
            debug_path = './cache/'+os.path.basename(output_laser_name)
            cv2.imwrite(debug_path, img)
            print(debug_path)
    return result

camera_names = [dataset_pb2.CameraName.FRONT, dataset_pb2.CameraName.FRONT_LEFT, dataset_pb2.CameraName.FRONT_RIGHT, dataset_pb2.CameraName.SIDE_LEFT, dataset_pb2.CameraName.SIDE_RIGHT]

paths = data_paths_train[args.start:args.start+100]
for i, filename in enumerate(paths):
    out_json = out_json_dir+"/"+os.path.basename(filename)+'.json'
    out = dict()
    print(i, filename, '/', len(paths))
    frames_data = f_datapath(filename)
    f_frame_data(frames_data[0])
    outs = multi_thread(f_frame_data, frames_data, verbose=1, max_workers=4)
    for _ in out:
        out.update(out)
    print('-----------------------')
    with open(out_json, 'w') as f:
        json.dump(out, f)
