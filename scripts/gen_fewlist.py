import argparse
import random
import os
import numpy as np
from os import path

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# few_nums = [1, 10]
few_nums = [1, 2, 3, 5, 10]
# few_nums = [20]
DROOT = '/home/bykang/voc'
root =  DROOT + '/voclist/'
rootfile =  DROOT + '/voc_train.txt'

def is_valid(imgpath, cls_name):
    imgpath = imgpath.strip()
    labpath = imgpath.replace('images', 'labels_1c/{}'.format(cls_name)) \
                         .replace('JPEGImages', 'labels_1c/{}'.format(cls_name)) \
                         .replace('.jpg', '.txt').replace('.png','.txt')
    if os.path.getsize(labpath):
        return True
    else:
        return False

def gen_image_fewlist():
    print('-----------------------------------------------------------')
    print('----------- Generating fewlist  (images) ------------------')
    for i, clsname in enumerate(classes):
        print('===> Processing class: {}'.format(clsname))
        with open(path.join(root, '{}_train.txt'.format(clsname))) as f:
            name_list = f.readlines()
        num = max(few_nums)
        random.seed(i)
        # selected_list = random.sample(name_list, num)
        selected_list = []
        while len(selected_list) < num:
            x = random.sample(name_list, num)[0]
            if not is_valid(x, clsname):
                continue
            selected_list.append(x)

        for n in few_nums:
            with open(path.join(root, '{}shot_{}_train.txt'.format(n, clsname)), 'w') as f:
                for i in range(n):
                    f.write(selected_list[i])

# -------------------------------------------------------------------------------------

def get_bbox_fewlist(rootfile, shot):
    with open(rootfile) as f:
        names = f.readlines()
    random.seed(2018)
    cls_lists = [[] for _ in range(len(classes))]
    cls_counts = [0] * len(classes)
    while min(cls_counts) < shot:
        imgpath = random.sample(names, 1)[0]
        labpath = imgpath.strip().replace('images', 'labels') \
                                 .replace('JPEGImages', 'labels') \
                                 .replace('.jpg', '.txt').replace('.png','.txt')
        # To avoid duplication
        names.remove(imgpath)

        if not os.path.getsize(labpath):
            continue
        # Load converted annotations
        bs = np.loadtxt(labpath)
        bs = np.reshape(bs, (-1, 5))
        if bs.shape[0] > 3:
            continue

        # Check total number of bbox per class so far
        overflow = False
        bcls = bs[:,0].astype(np.int).tolist()
        for ci in set(bcls):
            if cls_counts[ci] + bcls.count(ci) > shot:
                overflow = True
                break
        if overflow:
            continue

        # Add current imagepath to the file lists 
        for ci in set(bcls):
            cls_counts[ci] += bcls.count(ci)
            cls_lists[ci].append(imgpath)

    return cls_lists


def gen_bbox_fewlist():
    print('-----------------------------------------------------------')
    print('----------- Generating fewlist  (bboxes) ------------------')
    for n in few_nums:
        print('===> On {} shot ...'.format(n))
        filelists = get_bbox_fewlist(rootfile, n)
        for i, clsname in enumerate(classes):
            print('   | Processing class: {}'.format(clsname))
            with open(path.join(root, 'box_{}shot_{}_train.txt'.format(n, clsname)), 'w') as f:
                for name in filelists[i]:
                    f.write(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default=None, choices=['box', 'img', 'both'])
    args = parser.parse_args()

    if args.type is None or args.type == 'box':
        gen_bbox_fewlist()
    elif args.type == 'img':
        gen_image_fewlist()
    elif args.type == 'both':
        gen_image_fewlist()
        gen_bbox_fewlist()


if __name__ == '__main__':
    main()






