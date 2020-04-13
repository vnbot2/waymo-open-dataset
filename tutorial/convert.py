import json
from tqdm import tqdm
import tensorflow as tf
import os
import cv2
import math
import numpy as np
import itertools

from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from pyson.utils import multi_thread
# tf.__version__
from glob import glob
################################################
# Step 1 tf->json
###################################################

data_paths_val = glob('/waymo/val*/*.tfrecord')
data_paths_train = [_ for _ in glob('/waymo/*/*.tfrecord') if not _ in data_paths_val]
data_paths_train_cam = [_ for _ in data_paths_train if 'camera_labels' in _]
data_paths_train_not_cam = [_ for _ in data_paths_train if not 'camera_labels' in _]

paths_records = data_paths_val
output_dir = '/waymo/coco_style/'
path_sample_coco_annotation = '/waymo/coco_style/intermidiate_json/instances_val2017.json'
path_output_annotation = "/waymo/coco_style/annotations/val_all.json"

def process_frame(frame):
    frame, frame_id, frame_name = frame
    images = frame.images
    images = frame.images
    labels = frame.camera_labels

    os.makedirs(output_dir, exist_ok=True)
    rt = dict()
    for im_id in range(len(images)):
        image = images[im_id]
        image_name = open_dataset.CameraName.Name.Name(image.name)
        image = tf.image.decode_jpeg(image.image).numpy()
        output_name = os.path.join(output_dir, 'images', f'{frame_name}_{frame_id}_{image_name}.jpg')
        if not os.path.exists(output_name):
            cv2.imwrite(output_name, image)
        bboxes_coco = []
        class_ids = []
        detection_difficulty_levels = []
        tracking_difficulty_levels = []
        if len(labels) == 0: # no prvided data for 2d
            with_camlabel=False
        else:
            with_camlabel=True
            bboxes = labels[im_id].labels # bboxes list for image 0 in this frame
            for box in bboxes:
                cx, cy, h, w = box.box.center_x, box.box.center_y, box.box.width, box.box.length
                x = cx - w/2
                y = cy - h/2
                class_id = box.type
                np_box = [x,y,w,h]
                np_box = np.clip(np_box, 0, 10000)
                bboxes_coco.append(np_box.tolist())
                class_ids.append(class_id)
                tracking_difficulty_levels.append(box.tracking_difficulty_level)
                detection_difficulty_levels.append(box.detection_difficulty_level)

        rt[output_name] = dict(with_camlabel=with_camlabel,
                               bboxes=bboxes_coco, 
                               labels=class_ids, 
                               tracking_difficulty_level=tracking_difficulty_levels, 
                               detection_difficulty_level = detection_difficulty_levels)
    return rt

def f_datapath(data_path):
    f_name = os.path.basename(data_path)
    frame_name = os.path.basename(data_path)
    frame_id = 0
    dataset = tf.data.TFRecordDataset(data_path, compression_type='')
    frames = []
    for data in dataset:
        frame_id += 1
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frames.append((frame, frame_id, f_name))
    return frames

data_dict = dict()

for data_path_cam in tqdm(paths_records):
    frames = f_datapath(data_path_cam)
    results = multi_thread(process_frame, frames, verbose=False, max_workers=None)
    name = os.path.basename(data_path_cam)
    for _ in results:
        data_dict.update(_)
    

sample = json.load(open(path_sample_coco_annotation))

sample['info'] = {'description': '2D waymo',
 'url': 'waymo.com',
 'version': '1.0',
 'year': 2020,
 'contributor': 'Hai Anh',
 'date_created': '2019/04/06'}

sample['licenses'] = []

rt_images = []
rt_annotations = []
rt_with_cam_labels = []
for image_id, (image_name, labels) in enumerate(data_dict.items()):
    filename = os.path.basename(image_name)
    image = {'license': 4,
         'file_name': filename,
         'height': 1280,
         'width': 1920,
         'id': image_id
    }
    rt_with_cam_labels.append(labels['with_camlabel'])
    rt_images.append(image)
    bboxes = labels['bboxes']
    labels = labels['labels']
    bboxes = np.array(bboxes).astype('int')
    for box, lbl in zip(bboxes, labels):
        annotation = dict(image_id=image_id, category_id=lbl, bbox=box.tolist(), iscrowd=0, id=len(rt_annotations))
        rt_annotations.append(annotation)
    

cates =  [
 {'supercategory': 'vehicle','id': 1, 'name': 'vehicle'},
 {'supercategory': 'perdestrian','id': 2, 'name': 'perdestrian'},
 {'supercategory': 'sign','id': 3, 'name': 'sign'},
 {'supercategory': 'cyclis','id': 3, 'name': 'cyclis'}
]

sample['images'] = rt_images
sample['with_camlabel'] = rt_with_cam_labels
sample['annotations'] = rt_annotations
sample['categories'] = cates

with open(path_output_annotation, 'w') as f:
    json.dump(sample, f)
print('done!', path_output_annotation)
