#!/usr/bin/python
# encoding: utf-8
import random
import os
import pdb
import numpy as np
from PIL import Image
from PIL import ImageFile
from cfg import cfg
ImageFile.LOAD_TRUNCATED_IMAGES = True


def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out

def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    #constrain_image(im)
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res

def data_augmentation(img, shape, jitter, hue, saturation, exposure, flag=True):
    oh = img.height  
    ow = img.width
    
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    if flag:
        pleft  = random.randint(-dw, dw)
        pright = random.randint(-dw, dw)
        ptop   = random.randint(-dh, dh)
        pbot   = random.randint(-dh, dh)
        flip = random.randint(1,10000)%2

        swidth =  ow - pleft - pright
        sheight = oh - ptop - pbot

        sx = float(swidth)  / ow
        sy = float(sheight) / oh
        
        cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

        dx = (float(pleft)/ow)/sx
        dy = (float(ptop) /oh)/sy

        sized = cropped.resize(shape)

        if flip: 
            sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
        img = random_distort_image(sized, hue, saturation, exposure)
    else:
        # pleft, pright, ptop, pbot, flip = 0, 0, 0, 0, 0
        flip, dx, dy, sx, sy  = 0, 0, 0, 1, 1
        img = img.resize(shape)

    return img, flip, dx,dy,sx,sy 


def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
    max_boxes = cfg.max_boxes
    label = np.zeros((max_boxes,5))

    if os.path.exists(labpath)  and os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))
        cc = 0
        for i in range(bs.shape[0]):
            # Filter out bboxes not in base classes
            imgid = labpath.split('/')[-1].split('.')[0]
            clsid = int(bs[i][0])
            # if clsid not in cfg.base_ids:
            #     continue
            if clsid in cfg.base_ids:
                keepit = True
            elif cfg.yolo_joint and imgid in cfg.metaids:
                keepit = True
            else:
                keepit = False
            if not keepit:
                continue

            x1 = bs[i][1] - bs[i][3]/2
            y1 = bs[i][2] - bs[i][4]/2
            x2 = bs[i][1] + bs[i][3]/2
            y2 = bs[i][2] + bs[i][4]/2
            
            x1 = min(0.999, max(0, x1 * sx - dx)) 
            y1 = min(0.999, max(0, y1 * sy - dy)) 
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))
            
            bs[i][1] = (x1 + x2)/2
            bs[i][2] = (y1 + y2)/2
            bs[i][3] = (x2 - x1)
            bs[i][4] = (y2 - y1)

            if flip:
                bs[i][1] =  0.999 - bs[i][1] 
            
            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break

    label = np.reshape(label, (-1))
    return label


def fill_truth_detection_meta(labpath, w, h, flip, dx, dy, sx, sy):
    max_boxes = cfg.max_boxes
    n_cls = len(cfg.base_classes)
    label = np.zeros((n_cls, max_boxes, 5))

    if os.path.exists(labpath) and os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))
        ccs = [0] * n_cls
        for i in range(bs.shape[0]):
            # Filter out bboxes not in base classes
            clsid = int(bs[i][0])
            if clsid not in cfg.base_ids:
                continue
            x1 = bs[i][1] - bs[i][3]/2
            y1 = bs[i][2] - bs[i][4]/2
            x2 = bs[i][1] + bs[i][3]/2
            y2 = bs[i][2] + bs[i][4]/2
            
            x1 = min(0.999, max(0, x1 * sx - dx)) 
            y1 = min(0.999, max(0, y1 * sy - dy)) 
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))
            
            bs[i][1] = (x1 + x2)/2
            bs[i][2] = (y1 + y2)/2
            bs[i][3] = (x2 - x1)
            bs[i][4] = (y2 - y1)

            if flip:
                bs[i][1] =  0.999 - bs[i][1] 
            
            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue

            # Copy bbox info for building target
            ind = cfg.base_ids.index(clsid)
            if ind >= n_cls or ccs[ind]>= cfg.max_boxes:
                pdb.set_trace()
            label[ind][ccs[ind]] = bs[i]
            label[ind][ccs[ind]][0] = ind
            ccs[ind] += 1
            if sum(ccs) >= 50:
                break

    label = np.reshape(label, (n_cls, -1))
    return label


def load_label(labpath, w, h, flip, dx, dy, sx, sy):
    label = []
    # if os.path.exists(labpath) and os.path.getsize(labpath):
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))
        cc = 0
        for i in range(bs.shape[0]):
            x1 = bs[i][1] - bs[i][3]/2
            y1 = bs[i][2] - bs[i][4]/2
            x2 = bs[i][1] + bs[i][3]/2
            y2 = bs[i][2] + bs[i][4]/2
            
            x1 = min(0.999, max(0, x1 * sx - dx)) 
            y1 = min(0.999, max(0, y1 * sy - dy)) 
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))
            
            bs[i][1] = (x1 + x2)/2
            bs[i][2] = (y1 + y2)/2
            bs[i][3] = (x2 - x1)
            bs[i][4] = (y2 - y1)

            if flip:
                bs[i][1] =  0.999 - bs[i][1] 
            
            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue
            # label[cc] = bs[i]
            label.append(bs[i, 1:])
            cc += 1
            if cc >= 50:
                break

    return label


def load_data_detection(imgpath, labpath, shape, jitter, hue, saturation, exposure, data_aug=True):
    # labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
    # labpath = imgpath.replace('images', 'labels_1c/aeroplane').replace('JPEGImages', 'labels_1c/aeroplane').replace('.jpg', '.txt').replace('.png','.txt')

    ## data augmentation
    img = Image.open(imgpath).convert('RGB')
    img,flip,dx,dy,sx,sy = data_augmentation(img, shape, jitter, hue, saturation, exposure, flag=data_aug)
    if cfg.metayolo:
        label = fill_truth_detection_meta(labpath, img.width, img.height, flip, dx, dy, 1./sx, 1./sy)
    else:
        label = fill_truth_detection(labpath, img.width, img.height, flip, dx, dy, 1./sx, 1./sy)
    return img,label

def load_data_with_label(imgpath, labpath, shape, jitter, hue, saturation, exposure, data_aug=True):
    ## data augmentation
    img = Image.open(imgpath).convert('RGB')
    img,flip,dx,dy,sx,sy = data_augmentation(img, shape, jitter, hue, saturation, exposure, flag=data_aug)
    label = load_label(labpath, img.width, img.height, flip, dx, dy, 1./sx, 1./sy)
    return img, label
