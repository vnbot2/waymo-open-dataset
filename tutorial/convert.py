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
from utils import get_3d_points
################################################
# Step 1 tf->json
###################################################

data_paths_val = glob('/waymo/validation/*.tfrecord')
data_paths_test = glob('/ssd6/waymo/test/*.tfrecord')
data_paths_train = [_ for _ in glob('/waymo/training_1.2/*.tfrecord') if not _ in data_paths_val]
data_paths_train_cam = [_ for _ in data_paths_train if 'camera_labels' in _]
data_paths_train_not_cam = [_ for _ in data_paths_train if not 'camera_labels' in _]

paths_records = data_paths_train
print("Len not cam:", len(paths_records))
output_dir = '/ssd6/coco_style_1.2/'
path_output_annotation = "/ssd6/coco_style_1.2/annotations/train.json"
path_sample_coco_annotation = '/ssd6/coco_style_1.2/annotations/val.json'
os.makedirs(output_dir+ "annotations/", exist_ok=True)
os.makedirs(output_dir +"images/", exist_ok=True)

def process_frame(frame):
    frame, frame_id, frame_name = frame
    images = frame.images
    images = frame.images
    labels = frame.camera_labels
    import ipdb; ipdb.set_trace()
    lidar_labels = frame.lidar_labels
    rt = dict()
    for im_id in range(len(images)):
        image = images[im_id]
        image_name = open_dataset.CameraName.Name.Name(image.name)
        output_name = os.path.join(output_dir, 'images', f'{frame_name}_{frame_id}_{image_name}.jpg')
        image = tf.image.decode_jpeg(image.image).numpy()
        #if not os.path.exists(output_name):
        #    cv2.imwrite(output_name, image)
        bboxes_coco = []
        bboxes_3d = []
        class_ids = []
        bboxes_id = []
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
                bboxes_id.append(box.id)
                bboxes_coco.append(np_box.tolist())
                class_ids.append(class_id)
                tracking_difficulty_levels.append(box.tracking_difficulty_level)
                detection_difficulty_levels.append(box.detection_difficulty_level)
        
        # process 3d box
        bboxes_3d = get_3d_points(camera_calibration, frame.laser_labels)
        
        
        rt[output_name] = dict(with_camlabel=with_camlabel,
                               bboxes=bboxes_coco, 
                               bboxes_3d=bboxes_3d,
                               bboxes_id=bboxes_id, 
                               labels=class_ids,
                               timestamp_micros=frame.timestamp_micros,
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

# debug
frames = f_datapath(paths_records[0])
process_frame(frames[0])
# ----------------------
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
         'id': image_id,
         'timestamp_micros':labels["timestamp_micros"]
    }
    rt_with_cam_labels.append(labels['with_camlabel'])
    rt_images.append(image)
    bboxes = labels['bboxes']
    labels = labels['labels']
    bboxes_id = labels['bboxes_id']
    bboxes = np.array(bboxes).astype('int')

    anno_indi_dir = os.path.join(output_dir, "annotations", "anns")
    os.makedirs(anno_indi_dir, exist_ok=True)
    for anno_id, (box, box_id, lbl) in enumerate(zip(bboxes,bboxes_id, labels)):
        annotation = dict(image_id=image_id, box_id=box_id,category_id=lbl, bbox=box.tolist(), iscrowd=0, id=len(rt_annotations))
        out_path = os.path.join(anno_indi_dir, f"{anno_id}.json" )
        with open(out_path, "w") as f:
            json.dump(annotation, f)
        rt_annotations.append(os.path.join("anns", f"{anno_id}.json"))

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
