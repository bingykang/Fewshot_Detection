import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import argparse


# parser = argparse.ArgumentParser()
# parser.add_argument('--type', type=str, choices=['1c', 'all'], required=True)
# args = parser.parse_args()


sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id, class_name):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('VOCdevkit/VOC%s/labels_1c/%s/%s.txt'%(year, class_name, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls != class_name or int(difficult) == 1:
            continue
        # cls_id = classes.index(cls)
        cls_id = 0
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()

if not os.path.exists('voclist'):
    os.mkdir('voclist')

for class_name in classes:
    for year, image_set in sets:
        image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s_%s.txt'%(year, class_name, image_set)).read().strip().split()
        ids, flags = image_ids[::2], image_ids[1::2]
        image_ids = list(zip(ids, flags))

        # File to save the image path list
        list_file = open('voclist/%s_%s_%s.txt'%(year, class_name, image_set), 'w')

        # File to save the image labels
        label_dir = 'labels_1c/' + class_name
        if not os.path.exists('VOCdevkit/VOC%s/%s/'%(year, label_dir)):
            os.makedirs('VOCdevkit/VOC%s/%s/'%(year, label_dir))

        # Traverse all images
        for image_id, flag in image_ids:
            if int(flag) == -1:
                continue
            list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
            convert_annotation(year, image_id, class_name)
        list_file.close()

    files = [
        'voclist/2007_{}_train.txt'.format(class_name),
        'voclist/2007_{}_val.txt'.format(class_name),
        'voclist/2012_{}_*.txt'.format(class_name)
    ]
    files = ' '.join(files)
    cmd = 'cat ' + files + '> voclist/{}_train.txt'.format(class_name)
    os.system(cmd)