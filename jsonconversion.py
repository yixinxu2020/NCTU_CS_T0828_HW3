import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import skimage.draw


#TRAIN DATA
images = []
annotation = []
categories = []
coco = COCO("pascal_train.json")


# transform pascal_train.json data to coco format with 10 images as validation data
# if wanto get train data, just change 10 to all data
# for i in coco.imgs:
for i in range(1,11):
    imgIds = i
    annids = coco.getAnnIds(imgIds=imgIds)
    anns = coco.loadAnns(annids)
    img_info = coco.loadImgs(ids=imgIds)
    file_name = img_info[0]['file_name']
    height = img_info[0]['height']
    width = img_info[0]['width']
    images.append({
        "file_name": str(file_name), 
        "height": int(height),
        "width": int(width),
        "id": int(imgIds)
    })
    for j in range(len(anns)):
        ann_id = anns[j]['id']
        area = anns[j]['area']
        poly = anns[j]['segmentation']
        category = anns[j]['category_id']
        bbox = anns[j]['bbox']
        iscrowd = anns[j]['iscrowd']
        name = coco.cats[category]['name']
        supercategory = coco.cats[category]['supercategory']
        annotation.append({
            "segmentation": poly,
            "area": float(area),
            "iscrowd": iscrowd,
            "image_id": int(imgIds),
            "bbox": bbox,
            "category_id": int(category),
            "id": int(ann_id)
        })
for k in coco.cats:
    supercategory = coco.cats[k]['supercategory']
    category = coco.cats[k]['id']
    name = coco.cats[k]['name']
    categories.append({
        "supercategory": str(supercategory),
        "id": int(category),
        "name": str(name),
    })
dict = {'images':images, 'annotations':annotation, 'categories':categories}

#with open ('instance_train2017.json', 'w') as json_file:
with open('instances_val2017.json', 'w') as json_file:
        json.dump(dict, json_file)
