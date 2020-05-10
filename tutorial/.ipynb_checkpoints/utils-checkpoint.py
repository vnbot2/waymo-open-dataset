from pyson.utils import print_source
# import mmcv

import numpy as np
import math
import cv2
import io
# from waymo_open_dataset import dataset_pb2 as open_dataset
import sys
from tqdm import tqdm
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils
from utils import *
import matplotlib.cm
from glob import glob
from pyson.utils import read_json

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
    vertices_list = [ ]
    for label in labels:
        vertices = utils.get_3d_box_projected_corners(vehicle_to_image, label) 
        if vertices is not None:
            vertices = vertices.tolist()
        vertices_list.append(vertices)
    return vertices_list

def my_draw_3d_box(img, vertices, colour=(255,128,128), draw_2d_bounding_box=False):
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
        x1,y1,x2,y2 = compute_2d_bounding_box(img.shape, vertices)

        if (x1 != x2 and y1 != y2):
            cv2.rectangle(img, (x1,y1), (x2,y2), colour, thickness = 1)
    else:
        # Draw the edges of the 3D bounding box
        for k in [0, 1]:
            for l in [0, 1]:
                for idx1,idx2 in [((0,k,l),(1,k,l)), ((k,0,l),(k,1,l)), ((k,l,0),(k,l,1))]:
                    cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), colour, thickness=1)
        # Draw a cross on the front face to identify front & back.
        for idx1,idx2 in [((1,0,0),(1,1,1)), ((1,1,0),(1,0,1))]:
            cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), colour, thickness=1)
            
            

def display_3d_box_on_image(img, boxes_3d, visibility):
#     import ipdb; ipdb.set_trace()
    # Get the transformation matrix from the vehicle frame to image space.
#     vehicle_to_image = utils.get_image_transform(camera_calibration)
    # Draw all the groundtruth labels
    for boxe_3d,vis in zip(boxes_3d, visibility):
        if vis:
            colour = (0,0,200)
        else:
            colour = (128,0,0)
        my_draw_3d_box(img, boxe_3d, colour)
            
def display_2d_box_on_image(img, boxes, color=(0,255,0), thick=2):
    for a,b, cls in boxes:
        cv2.rectangle(img, a,b,color, thick)
            
            

def display_labels_on_image(camera_calibration, img, labels, visibility):
#     import ipdb; ipdb.set_trace()
    # Get the transformation matrix from the vehicle frame to image space.
    vehicle_to_image = utils.get_image_transform(camera_calibration)
    # Draw all the groundtruth labels
    for label,vis in zip(labels, visibility):
        if vis:
            colour = (0,0,200)
        else:
            colour = (128,0,0)
        utils.draw_3d_box(img, vehicle_to_image, label, colour=colour)
    
def display_laser_on_image(img, pcl, vehicle_to_image, pcl_attr):
    # Convert the pointcloud to homogeneous coordinates.
    pcl1 = np.concatenate((pcl,np.ones_like(pcl[:,0:1])),axis=1)

    # Transform the point cloud to image space.
    proj_pcl = np.einsum('ij,bj->bi', vehicle_to_image, pcl1) 

    # Filter LIDAR points which are behind the camera.
    mask = proj_pcl[:,2] > 0
    proj_pcl = proj_pcl[mask]
    proj_pcl_attr = pcl_attr[mask]

    # Project the point cloud onto the image.
    proj_pcl = proj_pcl[:,:2]/proj_pcl[:,2:3]

    # Filter points which are outside the image.
    mask = np.logical_and(
        np.logical_and(proj_pcl[:,0] > 0, proj_pcl[:,0] < img.shape[1]),
        np.logical_and(proj_pcl[:,1] > 0, proj_pcl[:,1] < img.shape[1]))

    proj_pcl = proj_pcl[mask]
    proj_pcl_attr = proj_pcl_attr[mask]

    # Colour code the points based on distance.
    coloured_intensity = 255*cmap(proj_pcl_attr[:,0]/30)

    # Draw a circle for each point.
    for i in range(proj_pcl.shape[0]):
        cv2.circle(img, (int(proj_pcl[i,0]),int(proj_pcl[i,1])), 1, coloured_intensity[i])