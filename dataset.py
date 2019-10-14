#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils import read_truths_args, read_truths, is_dict
from image import *
from cfg import cfg
from collections import defaultdict
import pdb

def topath(p):
    return p.replace('scratch', 'tmp_scratch/basilisk')

def loadlines(root, checkvalid=True):
    if is_dict(root):
        lines = []
        with open(root, 'r') as f:
            # files = [line.rstrip().split()[-1] for line in f.readlines()]
            files = [line.rstrip().split() for line in f.readlines()]
            if checkvalid:
                files = [topath(line[-1]) for line in files if line[0] in cfg.base_classes]
            else:
                files = [topath(line[-1]) for line in files if line[0] in cfg.classes]
        for file in files:
            with open(file, 'r') as f:
                lines.extend(f.readlines())
        lines = sorted(list(set(lines)))
    else:
        with open(root, 'r') as file:
            lines = file.readlines()
    if checkvalid:
        lines = [topath(l) for l in lines if listDataset.is_valid(topath(l))]
    return lines


def is_valid(imgpath, withnovel=True):
    labpath = listDataset.get_labpath(imgpath.rstrip())
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is not None:
            bs = np.reshape(bs, (-1, 5))
            clsset = set(bs[:,0].astype(np.int).tolist())
            if withnovel:
                # Check whether an image contains base objects
                if not clsset.isdisjoint(set(cfg.base_ids)):
                    return True
            else:
                # Check whether an image contains base objects only
                if clsset.isdisjoint(set(cfg.novel_ids)):
                    return True 

    return False


def build_dataset(dataopt):
    # Base training dataset
    if not cfg.tuning:
        return loadlines(dataopt['train'])

    # Meta tuning dataset
    if cfg.repeat == 1:
        return loadlines(dataopt['meta'])
    else:
        if 'dynamic' not in dataopt or int(dataopt['dynamic']) == 0:
            return loadlines(dataopt['meta']) * cfg.repeat
        else:
            metalist, metacnt = load_metadict(dataopt['meta'], cfg.repeat)
            return build_fewset(dataopt['train'], metalist, metacnt, cfg.shot*cfg.repeat) 


def load_metadict(metapath, repeat=1):
    with open(metapath, 'r') as f:
        files = []
        for line in f.readlines():
            pair = line.rstrip().split()
            if len(pair) == 2:
                pass
            elif len(pair) == 4:
                pair = [pair[0]+' '+pair[1], pair[2]+' '+pair[3]]
            else:
                raise NotImplementedError('{} not recognized'.format(pair))
            files.append(pair)
        # files = [line.rstrip().split() for line in f.readlines()]

        metadict = {line[0]: loadlines(line[1]) for line in files}

    pdb.set_trace()
    # Remove base-class images
    for k in metadict.keys():
        if k not in cfg.novel_classes:
            metadict[k] = []
    metalist = set(sum(metadict.values(), []))

    # Count bboxes
    metacnt = {c:0 for c in metadict.keys()}
    for imgpath in metalist:
        labpath = listDataset.get_labpath(imgpath.strip())
        # Load converted annotations
        bs = np.loadtxt(labpath)
        bs = np.reshape(bs, (-1, 5))
        bcls = bs[:,0].astype(np.int).tolist()
        for ci in set(bcls):
            metacnt[cfg.classes[ci]] += bcls.count(ci)

    for c in metacnt.keys():
        metacnt[c] *= repeat

    metalist =  list(metalist) * repeat
    return metalist, metacnt


def build_fewset(imglist, metalist, metacnt, shot, replace=True):
    # Random sample bboxes for base classes
    if isinstance(imglist, str):
        with open(imglist) as f:
            names = f.readlines()
    elif isinstance(imglist, list):
        names = imglist.copy()
    else:
        raise NotImplementedError('imglist type not recognized')

    while min(metacnt.values()) < shot:
        imgpath = random.sample(names, 1)[0]
        labpath = listDataset.get_labpath(imgpath.strip())
        # Remove empty annotation
        if not os.path.getsize(labpath):
            names.remove(imgpath)
            continue

        # Load converted annotations
        bs = np.loadtxt(labpath)
        bs = np.reshape(bs, (-1, 5))
        bcls = bs[:,0].astype(np.int).tolist()

        if bs.shape[0] > 3:
            continue

        # Remove images contatining novel objects
        if not set(bcls).isdisjoint(set(cfg.novel_ids)):
            names.remove(imgpath)
            continue

        # Check total number of bbox per class so far
        overflow = False
        for ci in set(bcls):
            if metacnt[cfg.classes[ci]] + bcls.count(ci) > shot:
                overflow = True
                break
        if overflow:
            names.remove(imgpath)
            continue

        # Add current imagepath to the file lists 
        for ci in set(bcls):
            metacnt[cfg.classes[ci]] += bcls.count(ci)
        metalist.append(imgpath)

        # To avoid duplication
        if not replace:
            names.remove(imgpath)

    random.shuffle(metalist)
    return metalist


class listDataset(Dataset):

    def __init__(self, root,
            shape=None,
            shuffle=True,
            transform=None,
            target_transform=None,
            train=False, seen=0,
            batch_size=64,
            num_workers=4):
        self.train = train

        if isinstance(root, list):
            self.lines = root
        elif is_dict(root):
            lines = []
            with open(root, 'r') as f:
                files = [line.rstrip().split()[-1] for line in f.readlines()]
            for file in files:
                with open(file, 'r') as f:
                    lines.extend(f.readlines())
            self.lines = sorted(list(set(lines)))
        else:
            with open(root, 'r') as file:
                self.lines = [topath(l) for l in file.readlines()]

        # Filter out images not in base classes
        print("===> Number of samples (before filtring): %d" % len(self.lines))
        if self.train and not isinstance(root, list):
            self.lines = [l for l in self.lines if self.is_valid(l)]
        print("===> Number of samples (after filtring): %d" % len(self.lines))

        if shuffle:
            random.shuffle(self.lines)

        self.nSamples  = len(self.lines)
        self.transform = transform
        self.target_transform = target_transform
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.first_batch = False

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        bs = 64
        batchs = 4000
        if self.train and index % bs== 0 and cfg.data != 'coco' and cfg.multiscale:
            if self.first_batch:
                width = 19 * 32
                self.shape = (width, width)
                self.first_batch = False
            elif self.seen < batchs*bs:
                width = 13*32
                self.shape = (width, width)
            elif self.seen < 2*batchs*bs:
                width = (random.randint(0,3) + 13)*32
                self.shape = (width, width)
            elif self.seen < 3*batchs*bs:
                width = (random.randint(0,5) + 12)*32
                self.shape = (width, width)
            elif self.seen < 4*batchs*bs:
                width = (random.randint(0,7) + 11)*32
                self.shape = (width, width)
            else: # self.seen < 20000*64:
                # width = (random.randint(1,7) + 10)*32
                width = (random.randint(0,9) + 10)*32
                self.shape = (width, width)

        jitter = 0.2
        hue = 0.1
        saturation = 1.5 
        exposure = 1.5

        labpath = listDataset.get_labpath(imgpath)
        img, label = load_data_detection(imgpath, labpath, self.shape, jitter, hue, saturation, exposure, data_aug=self.train)
        label = torch.from_numpy(label)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        return (img, label)

    @staticmethod
    def get_labpath(imgpath):
        subdir = 'labels'
        labpath = imgpath.replace('images', subdir) \
                         .replace('JPEGImages', subdir) \
                         .replace('.jpg', '.txt').replace('.png','.txt')
        return labpath

    @staticmethod
    def is_valid(imgpath):
        labpath = listDataset.get_labpath(imgpath.rstrip())
        if os.path.getsize(labpath):
            bs = np.loadtxt(labpath)
            if bs is not None:
                bs = np.reshape(bs, (-1, 5))
                clsset = set(bs[:,0].astype(np.int).tolist())
                if not clsset.isdisjoint(set(cfg.base_ids)):
                    return True
        return False


class MetaDataset(Dataset):
    def __init__(self,
            metafiles,
            transform=None,
            target_transform=None,
            train=False,
            num_workers=4,
            ensemble=False,
            with_ids=False):

        # Backup labeled image paths (for meta-model)
        if train:
            self.classes = cfg.base_classes
            factor = 1
            if cfg.data == 'coco':
                factor = 4
        else:
            # self.classes = cfg.base_classes
            if cfg.data == 'coco':
                self.classes = cfg.base_classes
            else:
                self.classes = cfg.classes
            factor = 10
        print('num classes: ', len(self.classes))

        nbatch = factor * 500 * 64 * cfg.num_gpus // cfg.batch_size

        metainds = [[]] * len(self.classes)
        with open(metafiles, 'r') as f:
            metafiles = []
            for line in f.readlines():
                pair = line.rstrip().split()
                if len(pair) == 2:
                    pass
                elif len(pair) == 4:
                    pair = [pair[0]+' '+pair[1], pair[2]+' '+pair[3]]
                else:
                    raise NotImplementedError('{} not recognized'.format(pair))
                metafiles.append(pair)
            # metafiles = [tuple(line.rstrip().split()) for line in f.readlines()]
            metafiles = {k: topath(v) for k, v in metafiles}

            self.metalines = [[]] * len(self.classes)
            for i, clsname in enumerate(self.classes):
                with open(metafiles[clsname], 'r') as imgf:
                    lines = [topath(l) for l in imgf.readlines()]
                    self.metalines[i] = lines
                    if ensemble:
                        metainds[i] = list(zip([i]*len(lines), list(range(len(lines)))))
                    else:
                        inds = np.random.choice(range(len(lines)), nbatch).tolist()
                        metainds[i] = list(zip([i] * nbatch, inds))

        self.inds = sum(metainds, []) if ensemble else sum(list(zip(*metainds)), ())
        self.meta_cnts = [len(ls) for ls in self.metalines]
        if cfg.randmeta:
            self.inds = list(self.inds)
            random.shuffle(self.inds)
            self.inds = tuple(self.inds)

        self.with_ids = with_ids
        self.ensemble = ensemble
        self.batch_size = len(self.classes) * cfg.num_gpus
        self.meta_shape = (cfg.meta_width, cfg.meta_height)
        self.mask_shape = (cfg.mask_width, cfg.mask_height)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.num_workers = num_workers
        # self.meta_shape = (384, 384)
        self.meta_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        if ensemble:
            # import pickle
            # if os.path.exists('inds.pkl'):
            #     with open('inds.pkl', 'rb') as f:
            #         self.inds = pickle.load(f)
            # else:
            #     self.inds = self.filter(self.inds)
            #     with open('inds.pkl', 'wb') as f:
            #         pickle.dump(self.inds, f)
            self.inds = self.filter(self.inds)
            # with open('inds.pkl', 'rb') as f:
            #     self.inds = pickle.load(f)

        self.nSamples = len(self.inds)


    def __len__(self):
        return self.nSamples

    def get_img_mask(self, img, box, merge=True):
        w, h = self.mask_shape

        x1 = int(max(0, round((box[0] - box[2]/2) * w)))
        y1 = int(max(0, round((box[1] - box[3]/2) * h)))
        x2 = int(min(w, round((box[0] + box[2]/2) * w)))
        y2 = int(min(h, round((box[1] + box[3]/2) * h)))

        if cfg.metain_type in [3, 4]:
            croped = img.crop((x1, y1, x2, y2)).resize(img.size)
            croped = self.meta_transform(croped)
            img = self.meta_transform(img)
            img = torch.cat([img, croped])
        else:
            img = self.meta_transform(img)

        if x1 == x2 or y1 == y2:
            mask = None
        else:
            mask = torch.zeros((1, h, w))
            mask[:, y1:y2, x1:x2] = 1

        if merge:
            return torch.cat([img, mask]) 
        else:
            return img, mask

    def get_metaimg(self, clsid, imgpath):
        jitter = 0.2
        hue = 0.1
        saturation = 1.5 
        exposure = 1.5

        if isinstance(imgpath, int):
            imgpath = self.metalines[clsid][imgpath].rstrip()
        elif isinstance(imgpath, str):
            pass
        else:
            raise NotImplementedError("{}: img path not recognized")

        labpath = self.get_labpath(imgpath, self.classes[clsid])
        img, lab = load_data_with_label(
            imgpath, labpath, self.meta_shape, jitter, hue, saturation, exposure, data_aug=self.train)
        return img, lab

    def get_metain(self, clsid, metaind):
        meta_img, meta_lab = self.get_metaimg(clsid, metaind)
        if meta_lab:
            for lab in meta_lab:
                # print(lab)
                img, mask = self.get_img_mask(meta_img, lab, merge=False)
                if mask is None:
                    continue
                return (img, mask)

        # In case the selected meta image has only difficult objects
        while True and not self.ensemble:
        # while True:
            meta_imgpath = random.sample(self.metalines[clsid], 1)[0].rstrip()
            meta_img, meta_lab = self.get_metaimg(clsid, meta_imgpath)
            if not meta_lab:
                continue
            for lab in meta_lab:
                img, mask = self.get_img_mask(meta_img, lab, merge=False)
                if mask is None:
                    continue
                return (img, mask)
        return (None, None)

    def filter(self, inds):
        newinds = []
        print('===> filtering...')
        _cnt = 0
        for clsid, metaind in inds:
            print('|{}/{}'.format(_cnt, len(inds)))
            _cnt += 1
            img, mask = self.get_metain(clsid, metaind)
            if img is not None:
                newinds.append((clsid, metaind))
        return newinds

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        clsid, metaind = self.inds[index]

        img, mask = self.get_metain(clsid, metaind)
        # ipath = self.metalines[clsid][metaind]

        if self.with_ids:
            return (img, mask, clsid)
        else:
            return (img, mask)
   
    @staticmethod
    def get_labpath(imgpath, cls_name):
        if cfg.data == 'voc':
            labpath = imgpath.replace('images', 'labels_1c/{}'.format(cls_name)) \
                             .replace('JPEGImages', 'labels_1c/{}'.format(cls_name)) \
                             .replace('.jpg', '.txt').replace('.png','.txt')
        else:
            if 'train2014' in imgpath:
                labpath = imgpath.replace('images/train2014', 'labels_1c/train2014/{}'.format(cls_name)) \
                                 .replace('.jpg', '.txt').replace('.png','.txt')
            elif 'val2014' in imgpath:
                labpath = imgpath.replace('images/val2014', 'labels_1c/val2014/{}'.format(cls_name)) \
                                 .replace('.jpg', '.txt').replace('.png','.txt')
            else:
                raise NotImplementedError("Image path note recognized!")

        return labpath


if __name__ == '__main__':
    from utils import read_data_cfg
    from cfg import parse_cfg
    from torchvision import transforms

    datacfg = 'cfg/metayolo.data'
    netcfg = 'cfg/dynamic_darknet_last.cfg'
    metacfg = 'cfg/learnet_last.cfg'

    data_options  = read_data_cfg(datacfg)
    net_options   = parse_cfg(netcfg)[0]
    meta_options  = parse_cfg(metacfg)[0]

    cfg.config_data(data_options)
    cfg.config_meta(meta_options)
    cfg.config_net(net_options)
    cfg.num_gpus = 4

    metafiles = 'data/voc_metadict1_full.txt'
    trainlist = '/scratch/bykang/datasets/voc_train.txt'

    metaset = MetaDataset(metafiles=metafiles, train=True)
    metaloader = torch.utils.data.DataLoader(
        metaset,
        batch_size=metaset.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
        )

    batch_size = 64
    kwargs = {'num_workers': 0, 'pin_memory': True}
    # train_loader = torch.utils.data.DataLoader(
    #     listDataset(trainlist, shape=(416, 416),
    #                    shuffle=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                    ]), 
    #                    train=True, 
    #                    seen=0,
    #                    batch_size=batch_size,
    #                    num_workers=0),
    #     batch_size=batch_size, shuffle=False, **kwargs)

    # for img, label, nums in train_loader:
    #     print(img.shape, label.shape, torch.sum(nums))

    for img, mask in metaloader:
        # pdb.set_trace()
        print(img.shape, mask.shape)
    # _metaloader = iter(metaloader)
    # for i in range(10):
    # i = 0
    # while True:
    #     _metaloader = iter(metaloader)
    #     for _ in range(8):
    #         img, mask = _metaloader.next()
    #         # print(img.shape, mask.shape)
    #         print(i)
    #         i += 1
