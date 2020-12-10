import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
print(ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

#%matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR + '/tranmodel', "mask_rcnn_pascal_0160.h5")
#85<95<90=100=105

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

import cv2
import json
import glob
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

cocoGt = COCO("coco/pascal/test.json")

from itertools import groupby
from pycocotools import mask as maskutil


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
    return compressed_rle

coco_dt = []
count = 0

for imgid in cocoGt.imgs:
    image = cv2.imread("coco/pascal/test2017/" + cocoGt.loadImgs(ids=imgid)[0]['file_name'])[:,:,::-1]
    results = model.detect([image], verbose=1)
    r = results[0]
    n_instances = len(r['scores'])
    count+=1
    print('picture: ', count)
    if len(r['class_ids']) > 0:
        for i in range(n_instances):
            pred = {}
            pred['image_id'] = imgid
            pred['category_id'] = int(r['class_ids'][i])
            pred['segmentation'] = binary_mask_to_rle(r['masks'][:,:,i])
            pred['score'] = float(r['scores'][i])
            coco_dt.append(pred)

with open("../logs/0856152.json", "w") as f:
    json.dump(coco_dt, f)
