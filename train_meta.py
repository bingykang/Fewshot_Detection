from __future__ import print_function
import sys
if len(sys.argv) != 5:
    print('Usage:')
    print('python train.py datacfg darknetcfg learnetcfg weightfile')
    exit()

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable

import dataset
import random
import math
import os
from utils import *
from cfg import parse_cfg, cfg
from darknet_meta import Darknet
from models.tiny_yolo import TinyYoloNet
import pdb

# Training settings
datacfg       = sys.argv[1]
darknetcfg    = parse_cfg(sys.argv[2])
learnetcfg    = parse_cfg(sys.argv[3])
weightfile    = sys.argv[4]

data_options  = read_data_cfg(datacfg)
net_options   = darknetcfg[0]
meta_options  = learnetcfg[0]

# Configure options
cfg.config_data(data_options)
cfg.config_meta(meta_options)
cfg.config_net(net_options)

# Parameters 
metadict      = data_options['meta']
trainlist     = data_options['train']

testlist      = data_options['valid']
backupdir     = data_options['backup']
gpus          = data_options['gpus']  # e.g. 0,1,2,3
ngpus         = len(gpus.split(','))
num_workers   = int(data_options['num_workers'])

batch_size    = int(net_options['batch'])
max_batches   = int(net_options['max_batches'])
learning_rate = float(net_options['learning_rate'])
momentum      = float(net_options['momentum'])
decay         = float(net_options['decay'])
steps         = [float(step) for step in net_options['steps'].split(',')]
scales        = [float(scale) for scale in net_options['scales'].split(',')]

#Train parameters
use_cuda      = True
seed          = int(time.time())
eps           = 1e-5
dot_interval  = 70  # batches
# save_interval = 10  # epoches

# Test parameters
conf_thresh   = 0.25
nms_thresh    = 0.4
iou_thresh    = 0.5

## --------------------------------------------------------------------------
## MAIN
backupdir = cfg.backup
print('logging to ' + backupdir)
if not os.path.exists(backupdir):
    os.mkdir(backupdir)

torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

model       = Darknet(darknetcfg, learnetcfg)
region_loss = model.loss

model.load_weights(weightfile)
model.print_network()


###################################################
### Meta-model parameters
region_loss.seen  = model.seen
processed_batches = 0 if cfg.tuning else model.seen/batch_size
trainlist         = dataset.build_dataset(data_options)
nsamples          = len(trainlist)
init_width        = model.width
init_height       = model.height
init_epoch        = 0 if cfg.tuning else model.seen/nsamples
max_epochs        = max_batches*batch_size/nsamples+1
max_epochs        = int(math.ceil(cfg.max_epoch*1./cfg.repeat)) if cfg.tuning else max_epochs 
print(cfg.repeat, nsamples, max_batches, batch_size)
print(num_workers)

kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}
test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(testlist, shape=(init_width, init_height),
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]), train=False),
    batch_size=batch_size, shuffle=False, **kwargs)

test_metaset = dataset.MetaDataset(metafiles=metadict, train=True)
test_metaloader = torch.utils.data.DataLoader(
    test_metaset,
    batch_size=test_metaset.batch_size,
    shuffle=False,
    num_workers=num_workers//2,
    pin_memory=True
)

# Adjust learning rate
factor = len(test_metaset.classes)
if cfg.neg_ratio == 'full':
    factor = 15.
elif cfg.neg_ratio == 1:
    factor = 3.0
elif cfg.neg_ratio == 0:
    factor = 1.5
elif cfg.neg_ratio == 5:
    factor = 8.0

print('factor:', factor)
learning_rate /= factor

if use_cuda:
    if ngpus > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

optimizer = optim.SGD(model.parameters(),
                      lr=learning_rate/batch_size,
                      momentum=momentum,
                      dampening=0,
                      weight_decay=decay*batch_size*factor)


def adjust_learning_rate(optimizer, batch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr/batch_size
    return lr

def train(epoch):
    global processed_batches
    t0 = time.time()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(trainlist, shape=(init_width, init_height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]), 
                       train=True, 
                       seen=cur_model.seen,
                       batch_size=batch_size,
                       num_workers=num_workers),
        batch_size=batch_size, shuffle=False, **kwargs)

    metaset = dataset.MetaDataset(metafiles=metadict, train=True)
    metaloader = torch.utils.data.DataLoader(
        metaset,
        batch_size=metaset.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    metaloader = iter(metaloader)

    lr = adjust_learning_rate(optimizer, processed_batches)
    logging('epoch %d/%d, processed %d samples, lr %f' % (epoch, max_epochs, epoch * len(train_loader.dataset), lr))

    model.train()
    t1 = time.time()
    avg_time = torch.zeros(9)
    for batch_idx, (data, target) in enumerate(train_loader):
        metax, mask = metaloader.next()
        t2 = time.time()
        adjust_learning_rate(optimizer, processed_batches)
        processed_batches = processed_batches + 1

        if use_cuda:
            data = data.cuda()
            metax = metax.cuda()
            mask = mask.cuda()
            #target= target.cuda()
        t3 = time.time()
        data, target = Variable(data), Variable(target)
        metax, mask = Variable(metax), Variable(mask)
        t4 = time.time()
        optimizer.zero_grad()
        t5 = time.time()
        output = model(data, metax, mask)
        t6 = time.time()
        region_loss.seen = region_loss.seen + data.data.size(0)
        loss = region_loss(output, target)
        t7 = time.time()
        loss.backward()
        t8 = time.time()
        optimizer.step()
        t9 = time.time()
        if False and batch_idx > 1:
            avg_time[0] = avg_time[0] + (t2-t1)
            avg_time[1] = avg_time[1] + (t3-t2)
            avg_time[2] = avg_time[2] + (t4-t3)
            avg_time[3] = avg_time[3] + (t5-t4)
            avg_time[4] = avg_time[4] + (t6-t5)
            avg_time[5] = avg_time[5] + (t7-t6)
            avg_time[6] = avg_time[6] + (t8-t7)
            avg_time[7] = avg_time[7] + (t9-t8)
            avg_time[8] = avg_time[8] + (t9-t1)
            print('-------------------------------')
            print('       load data : %f' % (avg_time[0]/(batch_idx)))
            print('     cpu to cuda : %f' % (avg_time[1]/(batch_idx)))
            print('cuda to variable : %f' % (avg_time[2]/(batch_idx)))
            print('       zero_grad : %f' % (avg_time[3]/(batch_idx)))
            print(' forward feature : %f' % (avg_time[4]/(batch_idx)))
            print('    forward loss : %f' % (avg_time[5]/(batch_idx)))
            print('        backward : %f' % (avg_time[6]/(batch_idx)))
            print('            step : %f' % (avg_time[7]/(batch_idx)))
            print('           total : %f' % (avg_time[8]/(batch_idx)))
        t1 = time.time()
    print('')
    t1 = time.time()
    logging('training with %f samples/s' % (len(train_loader.dataset)/(t1-t0)))

    if (epoch+1) % cfg.save_interval == 0:
        logging('save weights to %s/%06d.weights' % (backupdir, epoch+1))
        cur_model.seen = (epoch + 1) * len(train_loader.dataset)
        cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch+1))


def test(epoch):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    model.eval()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    num_classes = cur_model.num_classes
    anchors     = cur_model.anchors
    num_anchors = cur_model.num_anchors
    total       = 0.0
    proposals   = 0.0
    correct     = 0.0

    _test_metaloader = iter(test_metaloader)
    for batch_idx, (data, target) in enumerate(test_loader):
        metax, mask = _test_metaloader.next()
        if use_cuda:
            data = data.cuda()
            metax = metax.cuda()
            mask = mask.cuda()
        data = Variable(data, volatile=True)
        metax = Variable(metax, volatile=True)
        mask = Variable(mask, volatile=True)
        output = model(data, metax, mask).data
        all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
        for i in range(output.size(0)):
            boxes = all_boxes[i]
            boxes = nms(boxes, nms_thresh)
            truths = target[i].view(-1, 5)
            num_gts = truths_length(truths)
     
            total = total + num_gts
    
            for i in range(len(boxes)):
                if boxes[i][4] > conf_thresh:
                    proposals = proposals+1

            for i in range(num_gts):
                box_gt = [truths[i][1], truths[i][2], truths[i][3], truths[i][4], 1.0, 1.0, truths[i][0]]
                best_iou = 0
                best_j = -1
                for j in range(len(boxes)):
                    iou = bbox_iou(box_gt, boxes[j], x1y1x2y2=False)
                    if iou > best_iou:
                        best_j = j
                        best_iou = iou
                if best_iou > iou_thresh and boxes[best_j][6] == box_gt[6]:
                    correct = correct+1

    precision = 1.0*correct/(proposals+eps)
    recall = 1.0*correct/(total+eps)
    fscore = 2.0*precision*recall/(precision+recall+eps)
    logging("precision: %f, recall: %f, fscore: %f" % (precision, recall, fscore))



evaluate = False
if evaluate:
    logging('evaluating ...')
    test(0)
else:
    for epoch in range(init_epoch, max_epochs):
        train(epoch)
        # test(epoch)
