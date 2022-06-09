import os.path
import json
import argparse
import selectivesearch
import numpy as np
import random
import datetime as dt
import copy

def split_dataset(dataset):

    anns = dataset['annotations']
    imgs = dataset['images']
    nr_images = len(imgs)

    nr_testing_images = int(nr_images*0.2)
    nr_nontraining_images = int(nr_images*(0.2+ 0.1))

    random.shuffle(imgs)

    # Add new datasets
    train_set = {
        'images': [],
        'annotations': [],
        'categories': [],
    }
    train_set['categories'] = dataset['categories']

    val_set = copy.deepcopy(train_set)
    test_set = copy.deepcopy(train_set)

    test_set['images'] = imgs[0:nr_testing_images]
    val_set['images'] = imgs[nr_testing_images:nr_nontraining_images]
    train_set['images'] = imgs[nr_nontraining_images:nr_images]

    # Aux Image Ids to split annotations
    test_img_ids, val_img_ids, train_img_ids = [],[],[]
    for img in test_set['images']:
        test_img_ids.append(img['id'])

    for img in val_set['images']:
        val_img_ids.append(img['id'])

    for img in train_set['images']:
        train_img_ids.append(img['id'])

    # Split instance annotations
    for ann in anns:
        if ann['image_id'] in test_img_ids:
            test_set['annotations'].append(ann)
        elif ann['image_id'] in val_img_ids:
            val_set['annotations'].append(ann)
        elif ann['image_id'] in train_img_ids:
            train_set['annotations'].append(ann)

    return train_set,val_set,test_set

def initialize_proposals(dataset):
    images = dataset['images']
    all_proposals = []
    for img in images:
        img_lbl, regions = selectivesearch.selective_search(img, scale=200, min_size=100)
        img_area = np.prod(img.shape[:2])
        candidates = []
        for r in regions:
            if r['rect'] in candidates: continue
            if r['size'] < (0.05*img_area): continue
            if r['size'] > (1*img_area): continue
            all_proposals.append(r['rect'])   #x, y, w, h = r['rect']
    return all_proposals

