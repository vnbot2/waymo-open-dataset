import json
from pyson.utils import read_json
from glob import glob
import os
from tqdm import tqdm
output_dir = '/ssd6/coco_style_1.2/'
coco_json_dir = output_dir + "annotations/output_json_coco/"
out_file = output_dir +"annotations/train_3d.json"

json_paths = glob(f"{coco_json_dir}/*")


img_id_new = 0

ann_id_new = 0
images = []
annotations = []
sample = read_json(json_paths[0])
cates = [{'supercategory': 'vehicle', 'id': 1, 'name': 'vehicle'},
         {'supercategory': 'perdestrian', 'id': 2, 'name': 'perdestrian'},
         {'supercategory': 'sign', 'id': 3, 'name': 'sign'},
         {'supercategory': 'cyclis', 'id': 3, 'name': 'cyclis'}]
del sample['with_camlabel']
sample['categories'] = cates

for path in tqdm(json_paths[:5]):
    image_id_old_new = dict()
    x = read_json(path)
    for image in x['images']:
        if not image['id'] in image_id_old_new:
            image_id_old_new[image['id']] = img_id_new
            img_id_new += 1
        image['id'] = image_id_old_new[image['id']]
        images.append(image)

    for anno in x['annotations']:
        anno['image_id'] = image_id_old_new[anno['image_id']]
        anno['id'] = ann_id_new
        ann_id_new += 1
        annotations.append(anno)

print("-----------Summary-------------")
print("Total images:", len(images))
print("Total anno:", len(annotations))
sample['images'] = images
sample['annotations'] = annotations
img_ids = set()
for anno in annotations:
    img_ids.add(anno['image_id'])
print("num of images with labels: ",len(img_ids))

with open(out_file, "w") as f:
    json.dump(sample, f)
print("out_file: ", out_file)