import io
import math
import os
# from waymo_open_dataset import dataset_pb2 as open_dataset
import sys
from glob import glob

import cv2
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pyson.utils import multi_thread, print_source, read_json
from simple_waymo_open_dataset_reader import (WaymoDataFileReader, dataset_pb2,
											  label_pb2, utils)
from tqdm import tqdm

from utils import *

# import mmcv


tf.enable_eager_execution()

cmap = matplotlib.cm.get_cmap("viridis")


def get_2d_bbox(frame, camera_name):
	def get_2d_box(label):
		box = label.box
		x1 = int(box.center_x - box.length/2)
		x2 = int(box.center_x + box.length/2)
		y1 = int(box.center_y - box.width/2)
		y2 = int(box.center_y + box.width/2)
		return (x1, y1), (x2, y2), label.type, label.id
	assert camera_name >= 1
	boxes_2d = []

	for labels in frame.camera_labels:
		if labels.name == camera_name:
			break

	for label in labels.labels:
		boxes_2d.append(get_2d_box(label))
	return boxes_2d


def get_3d_points(camera_calibration, labels):
	vehicle_to_image = utils.get_image_transform(camera_calibration)
	vertices_list = []
	for label in labels:
		import ipdb; ipdb.set_trace()
		vertices = utils.get_3d_box_projected_corners(vehicle_to_image, label)
		if vertices is not None:
			vertices = vertices.tolist()
		vertices_list.append(vertices)
	return vertices_list


def my_draw_3d_box(img, vertices, colour=(255, 128, 128), draw_2d_bounding_box=False):
	"""Draw a 3D bounding from a given 3D label on a given "img". "vehicle_to_image" must be a projection matrix from the vehicle reference frame to the image space.

	draw_2d_bounding_box: If set a 2D bounding box encompassing the 3D box will be drawn
	"""
	if isinstance(vertices, list):
		vertices = np.array(vertices)
	import cv2
	if vertices is None:
		# The box is not visible in this image
		return

	if draw_2d_bounding_box:
		x1, y1, x2, y2 = compute_2d_bounding_box(img.shape, vertices)

		if (x1 != x2 and y1 != y2):
			cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness=1)
	else:
		# Draw the edges of the 3D bounding box
		for k in [0, 1]:
			for l in [0, 1]:
				for idx1, idx2 in [((0, k, l), (1, k, l)), ((k, 0, l), (k, 1, l)), ((k, l, 0), (k, l, 1))]:
					cv2.line(img, tuple(vertices[idx1]), tuple(
						vertices[idx2]), colour, thickness=1)
		# Draw a cross on the front face to identify front & back.
		for idx1, idx2 in [((1, 0, 0), (1, 1, 1)), ((1, 1, 0), (1, 0, 1))]:
			cv2.line(img, tuple(vertices[idx1]), tuple(
				vertices[idx2]), colour, thickness=1)


def display_3d_box_on_image(img, boxes_3d, visibility):
	for boxe_3d, vis in zip(boxes_3d, visibility):
		if vis:
			colour = (0, 0, 200)
		else:
			colour = (128, 0, 0)
		my_draw_3d_box(img, boxe_3d, colour)


def display_2d_box_on_image(img, boxes, color=(0, 255, 0), thick=2):
	for a, b, _, _ in boxes:
		cv2.rectangle(img, a, b, color, thick)


def display_labels_on_image(camera_calibration, img, labels, visibility):
	# Get the transformation matrix from the vehicle frame to image space.
	vehicle_to_image = utils.get_image_transform(camera_calibration)
	# Draw all the groundtruth labels
	for label, vis in zip(labels, visibility):
		if vis:
			colour = (0, 0, 200)
		else:
			colour = (128, 0, 0)
		utils.draw_3d_box(img, vehicle_to_image, label, colour=colour)


def display_laser_on_image(img, pcl, vehicle_to_image, pcl_attr):
	# Convert the pointcloud to homogeneous coordinates.
	pcl1 = np.concatenate((pcl, np.ones_like(pcl[:, 0:1])), axis=1)

	# Transform the point cloud to image space.
	proj_pcl = np.einsum('ij,bj->bi', vehicle_to_image, pcl1)

	# Filter LIDAR points which are behind the camera.
	mask = proj_pcl[:, 2] > 0
	proj_pcl = proj_pcl[mask]
	proj_pcl_attr = pcl_attr[mask]

	# Project the point cloud onto the image.
	proj_pcl = proj_pcl[:, :2]/proj_pcl[:, 2:3]

	# Filter points which are outside the image.
	mask = np.logical_and(
		np.logical_and(proj_pcl[:, 0] > 0, proj_pcl[:, 0] < img.shape[1]),
		np.logical_and(proj_pcl[:, 1] > 0, proj_pcl[:, 1] < img.shape[1]))

	proj_pcl = proj_pcl[mask]
	proj_pcl_attr = proj_pcl_attr[mask]

	# Colour code the points based on distance.
	coloured_intensity = 255*cmap(proj_pcl_attr[:, 0]/30)

	# Draw a circle for each point.
	for i in range(proj_pcl.shape[0]):
		cv2.circle(img, (int(proj_pcl[i, 0]), int(
			proj_pcl[i, 1])), 1, coloured_intensity[i])


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


def get_3d_label(frames_data):
	# get label 3d
	def f_frame_data(frame_data):
		output_dir = '/ssd6/coco_style_1.2/'
		frame, frame_id, frame_name = frame_data
		result = {}
		for camera_name in range(1, 6, 1):
			camera_calibration = utils.get(frame.context.camera_calibrations, camera_name)
			vehicle_to_image = utils.get_image_transform(camera_calibration)
			camera = utils.get(frame.images, camera_name)
			image_name = dataset_pb2.CameraName.Name.Name(camera_name)
			output_name = os.path.join(
				output_dir, 'images', f'{frame_name}_{frame_id}_{image_name}.jpg')
			result[os.path.basename(output_name)] = dict(type_id_list=[(l.type, l.id) for l in frame.laser_labels],
										vehicle_to_image=vehicle_to_image)

		return result

	label_3d_result = multi_thread(f_frame_data, frames_data)
	labels_3d_return = label_3d_result[0]
	for _ in label_3d_result:
		labels_3d_return.update(_)
	return labels_3d_return


def image_to_coco_dict(image_path, index, vehicle_to_image):
	h, w = cv2.imread(image_path).shape[:2]
	return {'license': 4,
			'file_name': os.path.basename(image_path),
			'height': h,
			'width': w,
			'id': index,
			'vehicle_to_image':vehicle_to_image,
			}


def annotation_to_dict(value, image_index):
	anno_id = None
	rt = []
	box_3d_list = value['box_3d_list']
	counts = value['counts']
	cates = value['cates']
	box_2d_list = value['box_2d_list']
	for count, box_3d, (cate,box_id ) in zip(counts, box_3d_list, cates):
		if box_3d is not None:
			ann = {'image_id': image_index,
				   'category_id': cate,
				   'box_3d': box_3d,
				   'iscrowd': -1,
				   'count': count,
				   'id': anno_id,
				   'box_id': box_id,
				   }
			rt.append(ann)
	for a, b, cate, box_id in box_2d_list:
		ann = {'image_id': image_index,
			   'category_id': cate,
			   'box': [*a, *b],
			   'iscrowd': -1,
			   'id': anno_id,
			   'box_id': box_id,
			   }
		rt.append(ann)
	return rt




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