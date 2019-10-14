import torch
from utils import convert2cpu
import numpy as np
from easydict import EasyDict as edict
from os import path

__C = edict()
cfg = __C


def load_classes(data='voc'):
    fname = path.dirname(path.abspath(__file__))
    fname = path.join(fname, 'data/{}.names'.format(data))
    print(fname)
    with open(fname) as f:
        classes = [l.strip() for l in f.readlines()]
    return classes

__C.voc_classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"]

__C.coco_classes = load_classes(data='coco')
__C.vocids_in_coco = [__C.coco_classes.index(c) for c in __C.voc_classes]
__C.cocoonly_ids = [i for i in range(len(__C.coco_classes)) if i not in __C.vocids_in_coco]

# Maximum number of bboxes per category
__C.max_boxes = 50

__C.neg_ratio = 'full'
__C.tuning = False
__C.metayolo = True
__C.repeat = 1
__C.save_interval = 10
__C.multiscale = True

# '1' for image only, '2' for image + mask, '3' for image + mask + croped obj
__C.metain_type = 2

def get_ids(root):
    lines = []
    with open(root, 'r') as f:
        # files = [line.rstrip().split()[-1] for line in f.readlines()]
        files = [line.rstrip().split() for line in f.readlines()]
        files = [line[-1] for line in files if line[0] in cfg.base_classes]
    for file in files:
        with open(file, 'r') as f:
            lines.extend(f.readlines())
    lines = sorted(list(set(lines)))
    ids = [l.split('/')[-1].split('.')[0] for l in lines]
    # print(ids)
    return ids

def get_novels(root, id=None):
    if root.endswith('txt'):
        if id == 'None':
            return []
        with open(root, 'r') as f:
            novels = f.readlines()
        return novels[int(id)].strip().split(',')
    else:
        return root.split(',')

def add_backup(backup, addon):
    strs = backup.split('_')
    strs[0] += addon
    return '_'.join(strs)

def __configure_data(dataopt):
    __C.data = dataopt['data']
    if dataopt['data'] == 'voc':
        __C.classes = __C.voc_classes
    elif dataopt['data'] == 'coco':
        __C.classes = __C.coco_classes
        __C.save_interval = 2

    if 'scale' in dataopt:
        __C.multiscale = int(dataopt['scale'])

    if 'metain_type' in dataopt:
        __C.metain_type = int(dataopt['metain_type'])

    if 'tuning' in dataopt:
        __C.tuning = bool(int(dataopt['tuning']))
        __C.max_epoch = int(dataopt['max_epoch']) if 'max_epoch' in dataopt else 500
        __C.repeat = int(dataopt['repeat']) if 'repeat' in dataopt else 100
        if __C.max_epoch / __C.repeat <= 20:
            __C.save_interval = 1
        elif __C.max_epoch / __C.repeat <= 50:
            __C.save_interval = 2
        elif __C.max_epoch / __C.repeat <= 100:
            __C.save_interval = 5
        else:
            __C.save_interval = 10
        if __C.data == 'coco':
            __C.save_interval = 2

        __C.shot = int(dataopt['meta'].split('.')[0].split('_')[-1].replace('shot', ''))

    print('save_interval', __C.save_interval)

    __C.novelid = novelid = dataopt['novelid'] if 'novelid' in dataopt else 'None'
    __C.novel_classes = get_novels(dataopt['novel'], novelid)
    print(__C.novel_classes)
    if __C.tuning:
        if dataopt['data'] == 'coco':
            __C.base_classes = __C.voc_classes + __C.novel_classes
            __C.base_classes = __C.classes
        elif dataopt['data'] == 'voc':
            __C.base_classes = __C.classes
        else:
            raise NotImplementedError('Data type {} not found'.format(dataopt['data']))
    else:
        __C.base_classes = [c for c in __C.classes if c not in __C.novel_classes]
    __C.base_ids = [__C.classes.index(c) for c in __C.base_classes]
    __C.novel_ids = [__C.classes.index(c) for c in __C.novel_classes]
    __C._real_base_ids = [i for i in range(len(__C.classes)) if i not in __C.novel_ids]
    print('base_ids', __C.base_ids)
    __C.num_gpus = len(dataopt['gpus'].split(','))
    __C.neg_ratio = dataopt['neg'] if 'neg' in dataopt else __C.neg_ratio
    __C.randmeta = bool(int(dataopt['rand'])) if 'rand' in dataopt else False
    __C.metayolo = bool(int(dataopt['metayolo']))

    if __C.neg_ratio.isdigit():
        __C.neg_ratio = float(__C.neg_ratio)
        if __C.neg_ratio.is_integer():
            __C.neg_ratio = int(__C.neg_ratio)

    # Set up backup dir
    __C.backup = dataopt['backup']
    if not __C.multiscale:
        __C.backup += 'fix'
    if __C.metain_type != 2:
        __C.backup = add_backup(__C.backup, 'in{}'.format(__C.metain_type))
    __C.backup += '_novel{}'.format(novelid)
    if cfg.metayolo:
        __C.backup = __C.backup + '_neg{}'.format(cfg.neg_ratio)
    if cfg.randmeta:
        __C.backup += '_rand'

    # Get few-shot image ids
    cfg.yolo_joint = int(dataopt['joint']) if 'joint' in dataopt else False
    if cfg.yolo_joint:
        cfg.metaids = get_ids(dataopt['meta'])
        shot = int(dataopt['meta'].split('.')[0].split('_')[-1].replace('shot', ''))
        __C.backup += '_joint{}'.format(shot)

def __configure_net(netopt):
    __C.height = int(netopt['height'])
    __C.width = int(netopt['width'])
    __C.batch_size = int(netopt['batch'])


def __configure_meta(metaopt):
    __C.meta_height = int(metaopt['height'])
    __C.meta_width = int(metaopt['width'])
    factor = int(metaopt['feat_layer'])
    if factor == 0:
        __C.mask_height = __C.meta_height
        __C.mask_width = __C.meta_width
    else:
        __C.mask_height = __C.meta_height // factor
        __C.mask_width = __C.meta_width // factor

    # meta input type
    if factor == 0:
        if __C.metain_type == 1:
            metaopt['channels'] = 3
        elif __C.metain_type == 2:
            metaopt['channels'] = 4
        elif __C.metain_type == 3:
            metaopt['channels'] = 7
        elif __C.metain_type == 4:
            metaopt['channels'] = 6
        else:
            raise NotImplementedError('Meta input type not found: {}'.format(__C.metain_type))
    elif factor == 4:
        if __C.metain_type == 1:
            metaopt['channels'] = 64
        elif __C.metain_type == 2:
            metaopt['channels'] = 65
        elif __C.metain_type == 3:
            metaopt['channels'] = 64*2 + 1
        elif __C.metain_type == 4:
            metaopt['channels'] = 64*2
        else:
            raise NotImplementedError('Meta input type not found: {}'.format(__C.metain_type))
    else:
        raise NotImplementedError('Feat layer not found{}'.format(factor))


__C.config_data = __configure_data
__C.config_meta = __configure_meta
__C.config_net = __configure_net


def parse_cfg(cfgfile):
    blocks = []
    fp = open(cfgfile, 'r')
    block =  None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue        
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            # set default value
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            key,value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()

    if block:
        blocks.append(block)
    fp.close()
    return blocks

def print_cfg(blocks):
    print('layer     filters    size              input                output');
    prev_width = 416
    prev_height = 416
    prev_filters = 3
    out_filters =[]
    out_widths =[]
    out_heights =[]
    ind = -2
    for block in blocks:
        ind = ind + 1
        if block['type'] == 'net' or block['type'] == 'learnet':
            prev_width = int(block['width'])
            prev_height = int(block['height'])
            prev_filters = int(block['channels'])
            if block['type'] == 'learnet':
                factor = int(block['feat_layer'])
                if factor != 0:
                    prev_width = prev_width // factor
                    prev_height = prev_height // factor
            continue
        elif block['type'] == 'convolutional':
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            pad = (kernel_size-1)/2 if is_pad else 0
            width = (prev_width + 2*pad - kernel_size)/stride + 1
            height = (prev_height + 2*pad - kernel_size)/stride + 1
            if 'dynamic' in block and int(block['dynamic']) == 1:
                name = 'dconv'
            else:
                name = 'conv'
            print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, name, filters, kernel_size, kernel_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'maxpool':
            pool_size = int(block['size'])
            stride = int(block['stride'])
            width = prev_width/stride
            height = prev_height/stride
            print('%5d %-6s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'max', pool_size, pool_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'globalmax':
            pool_size = prev_width
            stride = 1
            width = prev_width/pool_size
            height = prev_height/pool_size
            filters = prev_filters
            print('%5d %-6s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'glomax', pool_size, pool_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'globalavg':
            pool_size = prev_width
            stride = 1
            width = prev_width/pool_size
            height = prev_height/pool_size
            filters = prev_filters
            print('%5d %-6s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'gloavg', pool_size, pool_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'avgpool':
            width = 1
            height = 1
            print('%5d %-6s                   %3d x %3d x%4d   ->  %3d' % (ind, 'avg', prev_width, prev_height, prev_filters,  prev_filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'split':
            splits = [int(sz) for sz in block['splits'].split(',')]
            filters = splits[-1]            
            print(('%5d %-6s %3d -> {}' % (ind, 'split', prev_filters)).format(splits))
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'softmax':
            print('%5d %-6s                                    ->  %3d' % (ind, 'softmax', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'cost':
            print('%5d %-6s                                     ->  %3d' % (ind, 'cost', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'reorg':
            stride = int(block['stride'])
            filters = stride * stride * prev_filters
            width = prev_width/stride
            height = prev_height/stride
            print('%5d %-6s             / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'reorg', stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers]
            if len(layers) == 1:
                print('%5d %-6s %d' % (ind, 'route', layers[0]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                print('%5d %-6s %d %d' % (ind, 'route', layers[0], layers[1]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert(prev_width == out_widths[layers[1]])
                assert(prev_height == out_heights[layers[1]])
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'region':
            print('%5d %-6s' % (ind, 'detection'))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'shortcut':
            from_id = int(block['from'])
            from_id = from_id if from_id > 0 else from_id+ind
            print('%5d %-6s %d' % (ind, 'shortcut', from_id))
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'connected':
            filters = int(block['output'])
            print('%5d %-6s                            %d  ->  %3d' % (ind, 'connected', prev_filters,  filters))
            prev_filters = filters
            out_widths.append(1)
            out_heights.append(1)
            out_filters.append(prev_filters)
        elif block['type'] == 'reshape':
            inshape = [int(i) for i in block['inshape'].split(',')]
            outshape = [int(i) for i in block['outshape'].split(',')]
            if outshape[0] == -1:
                assert np.prod(inshape) % np.prod(outshape[1:]) == 0
                prev_filters = int(np.prod(inshape) / np.prod(outshape[1:]))
            else:
                prev_filters = outshape[0]
            outshape[0] = prev_filters

            print('%5d %-6s  %s  ->  %s' % (ind, 'reshape', inshape,  outshape))
            if len(outshape) == 1:
                out_widths.append(1)
                out_heights.append(1)
            elif len(outshape) == 3:
                out_widths.append(outshape[1])
                out_heights.append(outshape[2])
            else:
                raise NotImplementedError()
            out_filters.append(prev_filters)
        else:
            print('unknown type %s' % (block['type']))

def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    if conv_model.bias is not None:
        num_b = conv_model.bias.numel()
        conv_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w
    return start

def load_convfromcoco(buf, start, conv_model):
    print('------ loading coco to voc ----------')
    tmpb = torch.zeros(425)
    tmpw = torch.zeros(425, 1024, 1, 1)
    num_w = tmpw.numel()
    inds = np.concatenate([
        np.asarray([i for i in range(5)]),
        np.asarray(__C.vocids_in_coco) + 5]
    ) 
    allinds = np.concatenate([inds + i * 85 for i in range(5)])
    if conv_model.bias is not None:
        num_b = tmpb.numel()
        tmpb.copy_(torch.from_numpy(buf[start:start+num_b])); start = start + num_b
        conv_model.bias.data.copy_(tmpb[allinds]);   
    tmpw.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w
    conv_model.weight.data.copy_(tmpw[allinds]);
    return start
    

def save_conv(fp, conv_model):
    if conv_model.weight.is_cuda:
        if conv_model.bias is not None:
            convert2cpu(conv_model.bias.data).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        if conv_model.bias is not None:
            conv_model.bias.data.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)

def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));     start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b]));  start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w 
    return start

def save_conv_bn(fp, conv_model, bn_model):
    if bn_model.bias.is_cuda:
        convert2cpu(bn_model.bias.data).numpy().tofile(fp)
        convert2cpu(bn_model.weight.data).numpy().tofile(fp)
        convert2cpu(bn_model.running_mean).numpy().tofile(fp)
        convert2cpu(bn_model.running_var).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        bn_model.bias.data.numpy().tofile(fp)
        bn_model.weight.data.numpy().tofile(fp)
        bn_model.running_mean.numpy().tofile(fp)
        bn_model.running_var.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)

def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));     start = start + num_b
    fc_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]));   start = start + num_w 
    return start

def save_fc(fp, fc_model):
    fc_model.bias.data.numpy().tofile(fp)
    fc_model.weight.data.numpy().tofile(fp)

if __name__ == '__main__':
    import sys
    blocks = parse_cfg('cfg/yolo.cfg')
    if len(sys.argv) == 2:
        blocks = parse_cfg(sys.argv[1])
    print_cfg(blocks)
