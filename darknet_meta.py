import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from region_loss import RegionLossV2
from cfg import *
# from dynamic_conv import DynamicConv2d
from dynamic_conv import dynamic_conv2d
from pooling import GlobalMaxPool2d
from pooling import GlobalAvgPool2d
from pooling import Split
import pdb
#from layers.batchnorm.bn import BN2d


def maybe_repeat(x1, x2):
    n1 = x1.size(0)
    n2 = x2.size(0)
    if n1 == n2:
        pass
    elif n1 < n2:
        assert n2 % n1 == 0
        shape = x1.shape[1:]
        nc = n2 // n1
        x1 = x1.repeat(nc, *[1]*x1.dim())
        x1 = x1.transpose(0,1).contiguous()
        x1 = x1.view(-1, *shape)
    else:
        assert n1 % n2 == 0
        shape = x2.shape[1:]
        nc = n1 // n2
        x2 = x2.repeat(nc, *[1]*x2.dim())
        x2 = x2.transpose(0,1).contiguous()
        x2 = x2.view(-1, *shape)
    return x1, x2


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)
        return x

class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H/hs, hs, W/ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H/hs*W/ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H/hs, W/ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H/hs, W/ws)
        return x


# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

# support route shortcut and reorg
class Darknet(nn.Module):
    def __init__(self, darknet_file, learnet_file):
        super(Darknet, self).__init__()
        self.blocks = darknet_file if isinstance(darknet_file, list) else parse_cfg(darknet_file)
        self.learnet_blocks = learnet_file if isinstance(learnet_file, list) else parse_cfg(learnet_file)
        self.models = self.create_network(self.blocks) # merge conv, bn,leaky
        self.learnet_models = self.create_network(self.learnet_blocks)
        self.loss = self.models[len(self.models)-1]

        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

        if self.blocks[(len(self.blocks)-1)]['type'] == 'region':
            self.anchors = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            self.anchor_step = self.loss.anchor_step
            self.num_classes = self.loss.num_classes

        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0

    def meta_forward(self, metax, mask):
        # Get weights from learnet
        done_split = False
        for i in range(int(self.learnet_blocks[0]['feat_layer'])):
            if i == 0 and metax.size(1) == 6:
                done_split = True
                metax = torch.cat(torch.split(metax, 3, dim=1))
            metax = self.models[i](metax)
        if done_split:
            metax = torch.cat(torch.split(metax, int(metax.size(0)/2)), dim=1)
        if cfg.metain_type in [2, 3]:
            metax = torch.cat([metax, mask], dim=1)

        dynamic_weights = []
        for model in self.learnet_models:
            metax = model(metax)
            if isinstance(metax, list):
                dynamic_weights.append(metax[0])
                metax = metax[-1]
        dynamic_weights.append(metax)

        return dynamic_weights

    def detect_forward(self, x, dynamic_weights):
        # Perform detection
        ind = -2
        dynamic_cnt = 0
        self.loss = None
        outputs = dict()
        for block in self.blocks:
            ind = ind + 1
            #if ind > 0:
            #    return x

            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional' or \
                 block['type'] == 'maxpool' or \
                 block['type'] == 'reorg' or \
                 block['type'] == 'avgpool' or \
                 block['type'] == 'softmax' or \
                 block['type'] == 'connected' or \
                 block['type'] == 'globalavg' or \
                 block['type'] == 'globalmax':
                if self.is_dynamic(block):
                    x = self.models[ind]((x, dynamic_weights[dynamic_cnt]))
                    dynamic_cnt += 1
                else:
                    x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    if 'concat' in block and int(block['concat']) == 0:
                        x = (x1, x2)
                    else:
                        x1, x2 = maybe_repeat(x1, x2)
                        x = torch.cat((x1,x2),1)
                    outputs[ind] = x
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind-1]
                x  = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'region':
                continue
                if self.loss:
                    self.loss = self.loss + self.models[ind](x)
                else:
                    self.loss = self.models[ind](x)
                outputs[ind] = None
            elif block['type'] == 'cost':
                continue
            else:
                print('unknown type %s' % (block['type']))
        return x
        
    def forward(self, x, metax, mask, ids=None):
        # pdb.set_trace()
        dynamic_weights = self.meta_forward(metax, mask)
        x = self.detect_forward(x, dynamic_weights)
        return x

    def print_network(self):
        print_cfg(self.blocks)
        print('---------------------------------------------------------------------')
        print_cfg(self.learnet_blocks)

    def create_network(self, blocks):
        models = nn.ModuleList()

        prev_filters = 3
        out_filters =[]
        conv_id = 0
        dynamic_count = 0
        for block in blocks:
            if block['type'] == 'net' or block['type'] == 'learnet':
                prev_filters = int(block['channels'])
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size-1)/2 if is_pad else 0
                activation = block['activation']
                groups = 1
                bias = bool(int(block['bias'])) if 'bias' in block else True

                if self.is_dynamic(block):
                    partial = int(block['partial']) if 'partial' in block else None
                    Conv2d = dynamic_conv2d(dynamic_count == 0, partial=partial)
                    dynamic_count += 1
                else:
                    Conv2d = nn.Conv2d
                if 'groups' in block:
                    groups = int(block['groups'])

                model = nn.Sequential()
                if batch_normalize:
                    model.add_module(
                        'conv{0}'.format(conv_id),
                        Conv2d(prev_filters, filters, kernel_size, stride, pad, groups=groups, bias=False))
                    model.add_module(
                        'bn{0}'.format(conv_id),
                        nn.BatchNorm2d(filters))
                    #model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    model.add_module(
                        'conv{0}'.format(conv_id),
                        Conv2d(prev_filters, filters, kernel_size, stride, pad, groups=groups, bias=bias))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride > 1:
                    model = nn.MaxPool2d(pool_size, stride)
                else:
                    model = MaxPoolStride1()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(size_average=True)
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(size_average=True)
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(size_average=True)
                out_filters.append(1)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                models.append(Reorg(stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
                if len(layers) == 1:
                    prev_filters = out_filters[layers[0]]
                elif len(layers) == 2:
                    assert(layers[0] == ind - 1)
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind-1]
                out_filters.append(prev_filters)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                               nn.Linear(prev_filters, filters),
                               nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'region':
                loss = RegionLossV2()
                anchors = block['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(block['classes'])
                loss.num_anchors = int(block['num'])
                loss.anchor_step = len(loss.anchors)/loss.num_anchors
                loss.object_scale = float(block['object_scale'])
                loss.noobject_scale = float(block['noobject_scale'])
                loss.class_scale = float(block['class_scale'])
                loss.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                models.append(loss)
            elif block['type'] == 'globalmax':
                model = GlobalMaxPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'globalavg':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'split':
                splits = [int(sz) for sz in block['splits'].split(',')]
                model = Split(splits)
                prev_filters = splits[-1]
                out_filters.append(prev_filters)
                models.append(model)
            else:
                print('unknown type %s' % (block['type']))
    
        # pdb.set_trace()
        return models

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype = np.float32)
        fp.close()

        start = 0
        for blocks, models in [(self.blocks, self.models), (self.learnet_blocks, self.learnet_models)]:
            ind = -2
            for block in blocks:
                if start >= buf.size:
                    break
                ind = ind + 1
                if block['type'] == 'net' or block['type'] == 'learnet':
                    continue
                elif block['type'] == 'convolutional':
                    model = models[ind]
                    if self.is_dynamic(block) and model[0].weight is None:
                        continue    
                    batch_normalize = int(block['batch_normalize'])
                    if batch_normalize:
                        start = load_conv_bn(buf, start, model[0], model[1])
                    else:
                        
                        start = load_conv(buf, start, model[0])
                elif block['type'] == 'connected':
                    model = models[ind]
                    if block['activation'] != 'linear':
                        start = load_fc(buf, start, model[0])
                    else:
                        start = load_fc(buf, start, model)
                elif block['type'] == 'maxpool':
                    pass
                elif block['type'] == 'reorg':
                    pass
                elif block['type'] == 'route':
                    pass
                elif block['type'] == 'shortcut':
                    pass
                elif block['type'] == 'region':
                    pass
                elif block['type'] == 'avgpool':
                    pass
                elif block['type'] == 'softmax':
                    pass
                elif block['type'] == 'cost':
                    pass
                elif block['type'] == 'globalmax':
                    pass
                elif block['type'] == 'globalavg':
                    pass
                elif block['type'] == 'split':
                    pass
                else:
                    print('unknown type %s' % (block['type']))

    def save_weights(self, outfile, cutoff=0):
        # pdb.set_trace()
        if cutoff <= 0:
            cutoff = len(self.blocks) - 1 + len(self.learnet_blocks)

        fp = open(outfile, 'wb')
        self.header[3] = self.seen
        header = self.header
        header.numpy().tofile(fp)

        ind = -1
        for blockId in range(1, cutoff+1):
            # pdb.set_trace()
            if blockId >= len(self.blocks):
                if blockId == len(self.blocks):
                    ind = -2
                blockId = blockId - len(self.blocks)
                blocks = self.learnet_blocks
                models = self.learnet_models
            else:
                blocks = self.blocks
                models = self.models
            ind = ind + 1

            block = blocks[blockId]
            if block['type'] == 'convolutional':
                model = models[ind]
                if self.is_dynamic(block) and model[0].weight is None:
                    continue
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    save_conv_bn(fp, model[0], model[1])
                else:
                    save_conv(fp, model[0])
            elif block['type'] == 'connected':
                model = models[ind]
                if block['activation'] == 'linear':
                    save_fc(fp, model)
                else:
                    save_fc(fp, model[0])
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            elif block['type'] == 'globalmax':
                pass
            elif block['type'] == 'learnet':
                pass
            elif block['type'] == 'globalavg':
                pass
            elif block['type'] == 'split':
                pass
            else:
                print('unknown type %s' % (block['type']))
        fp.close()

    def is_dynamic(self, block):
        return 'dynamic' in block and int(block['dynamic']) == 1


if __name__ == '__main__':
    import argparse
    from torch.autograd import Variable

    parser = argparse.ArgumentParser()
    parser.add_argument('--darknet', type=str, required=True)
    parser.add_argument('--learnet', type=str, required=True)
    args = parser.parse_args()

    net = Darknet(args.darknet, args.learnet)
    net = net.cuda()

    x = Variable(torch.randn(8, 3, 416, 416))
    metax = Variable(torch.randn(8, 3, 384, 384))
    mask = Variable(torch.randn(8, 1, 96, 96))
    x = x.cuda()
    metax = metax.cuda()
    mask = mask.cuda()

    y = net(x, metax, mask)
    pdb.set_trace()
    net.save_weights('/tmp/dynamic.weights')
    print('hello')