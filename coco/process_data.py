import argparse
import random
import os
import numpy as np
import pdb
from os import path

root = '/scratch/bykang/coco/cocolist'
validfile = '/scratch/bykang/coco/5k.txt'
trainfile = '/scratch/bykang/coco/trainvalno5k.txt'
validdir = '/scratch/bykang/coco/images/val2014'
traindir = '/scratch/bykang/coco/images/train2014'
few_nums = [50]

def load_classes(data='voc'):
    fname = path.dirname(path.dirname(path.abspath(__file__)))
    fname = path.join(fname, 'data/{}.names'.format(data))
    print(fname)
    with open(fname) as f:
        classes = [l.strip() for l in f.readlines()]
    return classes


classes = load_classes('coco')
voc_classes = load_classes('voc')


def get_labelpath(imgpath):
    return imgpath.strip().replace('images', 'labels') \
                          .replace('JPEGImages', 'labels') \
                          .replace('.jpg', '.txt').replace('.png','.txt')


def is_valid(name, validids, clean=True):
    labpath = get_labelpath(name)
    if path.exists(labpath) and os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is not None:
            bs = np.reshape(bs, (-1, 5))
            clsset = set(bs[:,0].astype(np.int).tolist())
            if not clean and clsset & set(validids):
                return True
            if clean and clsset < set(validids):
                return True
    return False


def get_valid_list(novels, validdir):
    names = sorted(os.listdir(validdir))
    clsids = [classes.index(n) for n in novels]
    novel_list = []
    for name in names:
        if is_valid(path.join(validdir, name), clsids):
            novel_list.append(path.join(validdir, name) + '\n')
        if len(novel_list) >= 3000:
            break
    return novel_list


def load_coconovels():
    fname = path.dirname(path.dirname(path.abspath(__file__)))
    fname = path.join(fname, 'data/coco_novels.txt')
    with open(fname) as f:
        novels = [l.strip().split(',') for l in f.readlines()]
    return novels


def gen_valid_lists(dir=root, imgdir=validdir):
    if not path.exists(dir):
        os.makedirs(dir)
    novels = load_coconovels()

    for novel in novels:
        print('==> Generating validation image list for: ', novel)
        novel_list = get_valid_list(voc_classes + novel, imgdir)
        fname = path.join(dir, 'valid{}.txt'.format(len(novel)))
        print('==> Writing image names to :', fname)
        with open(fname, 'w') as f:
            for name in novel_list:
                f.write(name)
        print('Done..')


def get_bbox_fewlist(rootdir, shot, tgtclasses):
    if os.path.isdir(rootdir):
        names = sorted(os.listdir(rootdir))
    else:
        with open(rootdir) as f:
            names = f.readlines()

    random.seed(2018 + len(tgtclasses) + shot)
    # random.seed(999)
    clsids = [classes.index(n) for n in tgtclasses]
    cls_lists = [[] for _ in range(len(tgtclasses))]
    cls_counts = [0] * len(tgtclasses)
    while min(cls_counts) < shot:
        if len(names) == 0:
            if min(cls_counts) < 0.8 * shot:
                print(cls_counts)
                assert False, "No enough data"
            else:
                print(cls_counts)
                break
            # names = sorted(os.listdir(rootdir))
        name = random.sample(names, 1)[0]
        if os.path.isdir(rootdir):
            imgpath = path.join(rootdir, name) + '\n'
        else:
            imgpath = name
        labpath = get_labelpath(imgpath)

        # To avoid duplication
        names.remove(name)

        if not path.exists(labpath) or not path.getsize(labpath):
             continue
        # Load converted annotations
        bs = np.loadtxt(labpath)
        bs = np.reshape(bs, (-1, 5))

        # Remove images containing objs from other classes
        clsset = set(bs[:,0].astype(np.int).tolist())
        if not clsset <= set(clsids):
            continue
        if bs.shape[0] > 10:
            continue

        # Check total number of bbox per class so far
        overflow = False
        bcls = bs[:,0].astype(np.int).tolist()
        for ci in set(bcls):
            ind = tgtclasses.index(classes[ci])
            if cls_counts[ind] + bcls.count(ci) > shot:
                overflow = True
                break
        if overflow:
            continue

        # Add current imagepath to the file lists 
        for ci in set(bcls):
            ind = tgtclasses.index(classes[ci])
            cls_counts[ind] += bcls.count(ci)
            cls_lists[ind].append(imgpath)
        # print(len(names), cls_counts, min(cls_counts), max(cls_counts), bcls)

    return cls_lists

def get_bbox_fewlistv2(rootdir, shot, tgtclasses):
    if os.path.isdir(rootdir):
        names = sorted(os.listdir(rootdir))
    else:
        with open(rootdir) as f:
            names = f.readlines()

    # random.seed(2018 + len(tgtclasses) + shot)
    random.seed(999)
    if shot == 10:
        priority_cs = [[78]]
    elif shot == 30:
        priority_cs = [[24, 26, 30, 31, 34, 35, 38, 42, 44, 70, 78]]
    elif shot == 50:
        priority_cs = [
            [34, 35],
            [38],
            [29, 32, 70, 78],
            [24, 26, 30, 31, 40, 42, 44, 57]
        ]
        # priority_cs = [
        #     [35, 34, 38],
        #     [29, 32, 70, 78],
        #     [24, 26, 30, 31, 40, 42, 44, 57]
        # ]
        # [29, 32, 36, 57]
    else:
        priority_cs = []
    clsids = [classes.index(n) for n in tgtclasses]
    cls_lists = [[] for _ in range(len(tgtclasses))]
    cls_counts = [0] * len(tgtclasses)

    def get_bcls(name):
        if os.path.isdir(rootdir):
            imgpath = path.join(rootdir, name) + '\n'
        else:
            imgpath = name
        labpath = get_labelpath(imgpath)
        if not path.exists(labpath) or not path.getsize(labpath):
             return None
        # Load converted annotations
        bs = np.loadtxt(labpath)
        bs = np.reshape(bs, (-1, 5))

        # Remove images containing objs from other classes
        bcls = bs[:,0].astype(np.int).tolist()
        return bcls, imgpath

    def is_valid(bcls):
        if bcls is None:
            return False
        if not set(bcls) <= set(clsids):
            return False
        threshold = 10 if shot == 50 else 5
        if len(bcls) > threshold:
            return False

        overflow = False
        for ci in set(bcls):
            ind = tgtclasses.index(classes[ci])
            if cls_counts[ind] + bcls.count(ci) > shot:
                overflow = True
                break
        if overflow:
            return False

        return True

    def add_it(imgpath, bcls):
        # Add current imagepath to the file lists 
        for ci in set(bcls):
            ind = tgtclasses.index(classes[ci])
            cls_counts[ind] += bcls.count(ci)
            cls_lists[ind].append(imgpath)

    def check_subset(cs):
        cnts = [cls_counts[c] for c in cs]
        print(cs)
        print(cnts)
        if min(cnts) < 0.9 * shot:
            import pdb; pdb.set_trace()

    for pcs in priority_cs:
        # Generate a subset containing images of prioritized classes
        selected_names, selected_cnts = [], []
        for name in names:
            bcls, imgpath = get_bcls(name)
            if bcls is None: continue
            if not set(bcls).isdisjoint(set(pcs)):
                selected_names.append(name)
                selected_cnts.append(len(bcls))

        # Select samples for prioritized classes
        sorted_inds = np.argsort(selected_cnts)
        for i in sorted_inds:
            name = selected_names[i]
            bcls, imgpath = get_bcls(name)
            if is_valid(bcls):
                add_it(imgpath, bcls)
            names.remove(name)
        check_subset(pcs)

    # Select samples for the rest classes
    while min(cls_counts) < shot:
        if len(names) == 0:
            # pdb.set_trace()
            if min(cls_counts) < 0.8 * shot:
                print(cls_counts)
                lessids = np.argwhere(np.asarray(cls_counts) < shot).squeeze()
                print(lessids)
                print([cls_counts[li] for li in lessids])
                tobreak = False
                import pdb; pdb.set_trace() 
                if tobreak:
                    break
                assert False, "No enough data"
            else:
                print(cls_counts)
                break
            # names = sorted(os.listdir(rootdir))
        name = random.sample(names, 1)[0]
        bcls, imgpath = get_bcls(name)

        if is_valid(bcls):
            add_it(imgpath, bcls)

        # To avoid duplication
        names.remove(name)

    return cls_lists


def gen_bbox_fewlist():
    print('-----------------------------------------------------------')
    print('----------- Generating fewlist  (bboxes) ------------------')
    novels = load_coconovels()
    for novel in novels:
        print('===> For novel classes: ', novel )
        tgtclasses = voc_classes + novel
        for n in few_nums:
            print('===> On {} shot ...'.format(n))
            filelists = get_bbox_fewlist(traindir, n, tgtclasses)
            # pdb.set_trace()
            for i, clsname in enumerate(tgtclasses):
                print('   | Processing class: {}'.format(clsname))
                with open(path.join(root, 'nov{}_box_{}shot_{}_train.txt'.format(len(novel),n, clsname)), 'w') as f:
                    for name in filelists[i]:
                        f.write(name)
        print('-----------------------------------------------------------')


def gen_bbox_fewlist_fullcoco():
    print('-----------------------------------------------------------')
    print('--------- Generating fewlist (bboxes, fullcoco) -----------')

    tgtclasses = classes
    for n in few_nums:
        print('===> On {} shot ...'.format(n))
        filelists = get_bbox_fewlistv2(trainfile, n, tgtclasses)
        # pdb.set_trace()
        for i, clsname in enumerate(tgtclasses):
            # print('   | Processing class: {}'.format(clsname))
            print('{} {}'.format(clsname, path.join(root, 'full_box_{}shot_{}_trainval.txt'.format(n, clsname))))
            with open(path.join(root, 'full_box_{}shot_{}_trainval.txt'.format(n, clsname)), 'w') as f:
                for name in filelists[i]:
                    f.write(name)
    print('-----------------------------------------------------------')


def gen_label1c(fullset=False, traindir=traindir):
    if fullset:
        names = sorted(os.listdir(traindir))
        all_imgs = [path.join(traindir, name) for name in names]
    else: 
        novels = load_coconovels()
        all_imgs = []
        for novel in novels:
            tgtclasses = voc_classes + novel
            for n in few_nums:
                for clsname in tgtclasses:
                    fname = 'nov{}_box_{}shot_{}_train.txt'.format(len(novel), n, clsname)
                    with open(path.join(root, fname)) as f:
                        all_imgs.extend(f.readlines())
        all_imgs = list(set(all_imgs))

    print('We have in total {} images'.format(len(all_imgs)))
    for i, img in enumerate(all_imgs):
        labpath = get_labelpath(img)
        if not path.exists(labpath) or not path.getsize(labpath):
             continue

        print('{:04}/{} | '.format(i, len(all_imgs)) + img.strip())
        bs = np.loadtxt(labpath)
        bs = np.reshape(bs, (-1, 5))

        clsset = set(bs[:,0].astype(np.int).tolist())
        for clsid in clsset:
            ind = bs[:,0] == clsid
            subbs = bs[ind]

            folder = path.join(path.dirname(labpath), classes[clsid])
            folder = folder.replace('labels', 'labels_1c')
            if not path.exists(folder):
                os.makedirs(folder)

            with open(path.join(folder, path.basename(labpath)), 'w') as f:
                for b in subbs:
                    f.write(" ".join([str(a) for a in b]) + '\n')


def gen_traindict():
    rootdir = traindir
    cls_lists = [[] for _ in classes]
    names = sorted(os.listdir(rootdir))

    for name in names:
        imgpath = path.join(rootdir, name) + '\n'
        labpath = get_labelpath(imgpath)

        if not path.exists(labpath) or not path.getsize(labpath):
             continue

        # Load converted annotations
        bs = np.loadtxt(labpath)
        bs = np.reshape(bs, (-1, 5))

        clsset = set(bs[:,0].astype(np.int).tolist())
        for c in clsset:
            cls_lists[c].append(imgpath)

    pdb.set_trace()
    for ci, c in enumerate(classes):
        print('{} {}'.format(c, path.join(root, 'full_{}_train.txt'.format(c))))
        with open(path.join(root, 'full_{}_train.txt'.format(c)), 'w') as f:
            for ipath in cls_lists[ci]:
                f.write(ipath)


def gen_trainvaldict(trainfile=trainfile):
    rootdir = traindir
    cls_lists = [[] for _ in classes]
    with open(trainfile) as f:
        names = f.readlines()

    for name in names:
        imgpath = name
        labpath = get_labelpath(imgpath)

        if not path.exists(labpath) or not path.getsize(labpath):
             continue

        # Load converted annotations
        bs = np.loadtxt(labpath)
        bs = np.reshape(bs, (-1, 5))

        clsset = set(bs[:,0].astype(np.int).tolist())
        for c in clsset:
            cls_lists[c].append(imgpath)

    for ci, c in enumerate(classes):
        print('{} {}'.format(c, path.join(root, 'full_{}_trainval.txt'.format(c))))
        with open(path.join(root, 'full_{}_trainval.txt'.format(c)), 'w') as f:
            for ipath in cls_lists[ci]:
                f.write(ipath)


if __name__ == '__main__':
    # gen_trainvaldict()
    # gen_traindict()
    # gen_label1c(fullset=True, traindir=traindir)
    # gen_label1c(fullset=True, traindir=validdir)
    # gen_bbox_fewlist()
    gen_bbox_fewlist_fullcoco()
    # novels = load_coconovels()
    # l = get_bbox_fewlist(traindir, 10, voc_classes + novels[1])
    # print([len(ll) for ll in l])





