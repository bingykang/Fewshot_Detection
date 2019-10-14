import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image, ImageDraw
import sys
from collections import OrderedDict
from utils import do_detect, plot_boxes, load_class_names
sys.path.append('/home/xiaohang/caffe/python')
sys.path.append('.')
import caffe
from region_loss import RegionLoss
class Scale(nn.Module):
    def __init__(self):
        super(Scale, self).__init__()
    def forward(self, x):
        return x


class Eltwise(nn.Module):
    def __init__(self, operation='+'):
        super(Eltwise, self).__init__()
        self.operation = operation

    def forward(self, input_feats):
        if isinstance(input_feats, tuple):
            print "error : The input of Eltwise layer must be a tuple"
        for i, feat in enumerate(input_feats):
            if x is None:
                x = feat
                continue
            if self.operation == '+':
                x += feat
            if self.operation == '*':
                x *= feat
            if self.operation == '/':
                x /= feat
        return x

class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, input_feats):
        if not isinstance(input_feats, tuple):
            print 'The input of Concat layer must be a tuple'
        self.length = len(input_feats)
        x = torch.cat(input_feats, 1)
        return x



def parse_prototxt(protofile):
    def line_type(line):
        if line.find(':') >= 0:
            return 0
        elif line.find('{') >= 0:
            return 1
        return -1

    def parse_param_block(fp):
        block = dict()
        line = fp.readline().strip()
        while line != '}':
            ltype = line_type(line)
            if ltype == 0: # key: value
                key, value = line.split(':')
                key = key.strip()
                value = value.strip().strip('"')
                block[key] = value
            elif ltype == 1: # blockname {
                key = line.split('{')[0].strip()
                sub_block = parse_param_block(fp)
                block[key] = sub_block
            line = fp.readline().strip()
        return block

    def parse_layer_block(fp):
        block = dict()
        block['top'] = []
        block['bottom'] = []
        line = fp.readline().strip()
        while line != '}':
            ltype = line_type(line)
            if ltype == 0: # key: value
                key, value = line.split(':')
                key = key.strip()
                value = value.strip().strip('"')
                if key == 'top' or key == 'bottom':
                    block[key].append(value)
                else:
                    block[key] = value
            elif ltype == 1: # blockname {
                key = line.split('{')[0].strip()
                sub_block = parse_param_block(fp)
                block[key] = sub_block
            line = fp.readline().strip()
        return block

    fp = open(protofile, 'r')
    props = dict()
    layers = []
    line = fp.readline()
    while line != '':
        ltype = line_type(line)
        if ltype == 0: # key: value
            key, value = line.split(':')
            key = key.strip()
            value = value.strip().strip('"')
            props[key] = value
        elif ltype == 1: # blockname {
            key = line.split('{')[0].strip()
            assert(key == 'layer' or key == 'input_shape')
            layer = parse_layer_block(fp)
            layers.append(layer)
            #print layer
        line = fp.readline()
    net_info = dict()
    net_info['props'] = props
    net_info['layers'] = layers
    #print net_info

    return net_info


class CaffeNet(nn.Module):
    def __init__(self, protofile, caffemodel):
        super(CaffeNet, self).__init__()
        self.seen = 0
        self.num_classes = 1
        self.is_pretrained = True
        if not caffemodel is None:
            self.is_pretrained = True
        self.anchors = [0.625,0.750,   0.625,0.750,   0.625,0.750, \
                0.625,0.750,   0.625,0.750,   1.000,1.200,  \
                1.000,1.200,   1.000,1.200,   1.000,1.200,   \
                1.600,1.920,   2.560,3.072,   4.096,4.915,  \
                6.554,7.864,   10.486,12.583]
        #self.anchors = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
        self.num_anchors = len(self.anchors)/2
        self.width = 480
        self.height = 320

        self.loss = RegionLoss(self.num_classes, self.anchors, self.num_anchors)

        self.net_info = parse_prototxt(protofile)
        self.models = self.create_network(self.net_info)
        self.modelList = nn.ModuleList()
        if self.is_pretrained:
            self.load_weigths_from_caffe(protofile, caffemodel)
        for name,model in self.models.items():
            self.modelList.append(model)


    def load_weigths_from_caffe(self, protofile, caffemodel):
        caffe.set_mode_cpu()
        net = caffe.Net(protofile, caffemodel, caffe.TEST)
        for name, layer in self.models.items():
            if isinstance(layer, nn.Conv2d):
                caffe_weight = net.params[name][0].data
                layer.weight.data = torch.from_numpy(caffe_weight)
                if len(net.params[name]) > 1:
                    caffe_bias = net.params[name][1].data
                    layer.bias.data = torch.from_numpy(caffe_bias)
                continue
            if isinstance(layer, nn.BatchNorm2d):
                caffe_means = net.params[name][0].data
                caffe_var = net.params[name][1].data
                layer.running_mean = torch.from_numpy(caffe_means)
                layer.running_var = torch.from_numpy(caffe_var)
                # find the scale layer
                top_name_of_bn = self.layer_map_to_top[name][0]
                scale_name = ''
                for caffe_layer in self.net_info['layers']:
                    if caffe_layer['type'] == 'Scale' and caffe_layer['bottom'][0] == top_name_of_bn:
                        scale_name = caffe_layer['name']
                        break
                if scale_name != '':
                    caffe_weight = net.params[scale_name][0].data
                    layer.weight.data = torch.from_numpy(caffe_weight)
                    if len(net.params[name]) > 1:
                        caffe_bias = net.params[scale_name][1].data
                        layer.bias.data = torch.from_numpy(caffe_bias)



    def print_network(self):
        print(self.net_info)

    def create_network(self, net_info):
        #print net_info
        models = OrderedDict()
        top_dim = {'data': 3}
        self.layer_map_to_bottom = dict()
        self.layer_map_to_top = dict()

        for layer in net_info['layers']:
            name = layer['name']
            ltype = layer['type']

            if ltype == 'Data':
                continue
            if ltype == 'ImageData':
                continue
            if layer.has_key('top'):
                tops = layer['top']
                self.layer_map_to_top[name] = tops
            if layer.has_key('bottom'):
                bottoms = layer['bottom']
                self.layer_map_to_bottom[name] = bottoms
            if ltype == 'Convolution':
                filters = int(layer['convolution_param']['num_output'])
                kernel_size = int(layer['convolution_param']['kernel_size'])
                stride = 1
                group = 1
                pad = 0
                bias = True
                dilation = 1
                if layer['convolution_param'].has_key('stride'):
                    stride = int(layer['convolution_param']['stride'])
                if layer['convolution_param'].has_key('pad'):
                    pad = int(layer['convolution_param']['pad'])
                if layer['convolution_param'].has_key('group'):
                    group = int(layer['convolution_param']['group'])
                if layer['convolution_param'].has_key('bias_term'):
                    bias = True if layer['convolution_param']\
                            ['bias_term'].lower() == 'false' else False 
                if layer['convolution_param'].has_key('dilation'):
                    dilation = int(layer['convolution_param']['dilation'])
                num_output = int(layer['convolution_param']['num_output'])
                top_dim[tops[0]]=num_output
                num_input = top_dim[bottoms[0]]
                models[name] =  nn.Conv2d(num_input, num_output, kernel_size,
                    stride,pad,groups=group, bias=bias, dilation=dilation)
            elif ltype == 'ReLU':
                inplace = (bottoms == tops)
                top_dim[tops[0]] = top_dim[bottoms[0]]
                models[name] = nn.ReLU(inplace=False)
            elif ltype == 'Pooling':
                kernel_size = int(layer['pooling_param']['kernel_size'])
                stride = 1
                if layer['pooling_param'].has_key('stride'):
                    stride = int(layer['pooling_param']['stride'])
                top_dim[tops[0]] = top_dim[bottoms[0]]
                models[name] = nn.MaxPool2d(kernel_size, stride)
            elif ltype == 'BatchNorm':
                if layer['batch_norm_param'].has_key('use_global_stats'):
                    use_global_stats = True if layer['batch_norm_param']\
                            ['use_global_stats'].lower() == 'true' else False
                top_dim[tops[0]] = top_dim[bottoms[0]]
                
                models[name] = nn.BatchNorm2d(top_dim[bottoms[0]])

            elif ltype == 'Scale':
                top_dim[tops[0]] = top_dim[bottoms[0]]
                models[name] = Scale()
            elif ltype == 'Eltwise':
                top_dim[tops[0]] = top_dim[bottoms[0]]
                models[name] = Eltwise('+')
            elif ltype == 'Concat':
                top_dim[tops[0]] = 0
                for i, x in enumerate(bottoms):
                    top_dim[tops[0]] += top_dim[x]
                models[name] = Concat()
            elif ltype == 'Dropout':
                if layer['top'][0] == layer['bottom'][0]:
                    inplace = True
                else: 
                    inplace = False
                top_dim[tops[0]] = top_dim[bottoms[0]]
                models[name] = nn.Dropout2d(inplace=inplace)
            else:
                print '%s is not NotImplemented'%ltype

        return models

    def forward(self, x, target=None):
        blobs = OrderedDict()
        for name, layer in self.models.items():
            output_names = self.layer_map_to_top[name]
            input_names = self.layer_map_to_bottom[name]
            print "-----------------------------------------"
            print 'input_names: ',input_names
            print 'output_names:',output_names
            print layer
            # frist layer
            if input_names[0] == 'data':
                top_blobs = layer(x)
            else:
                input_blobs = [blobs[i] for i in input_names ]
                if isinstance(layer, Concat) or isinstance(layer, Eltwise):
                    top_blobs = layer(input_blobs)
                else:
                    top_blobs = layer(input_blobs[0])
            if not isinstance(top_blobs, tuple):
                top_blobs = (top_blobs,)

            for k, v in zip(output_names, top_blobs):
                blobs[k] = v
        output_name = blobs.keys()[-1]
        print 'output_name',output_name
        return blobs[output_name]



if __name__ == '__main__':
    prototxt = 'tiny_yolo_nbn_reluface.prototxt'
    caffemodel = '/nfs/xiaohang/for_chenchao/tiny_yolo_nbn_reluface.caffemodel'
    imgfile = 'data/face.jpg'
    
    m = CaffeNet(prototxt, caffemodel)
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((m.width, m.height))
    #if m.num_classes == 20:
    #    namesfile = '../data/voc.names'
    #class_names = load_class_names(namesfile)
    class_names = ['face']
    for i in range(1):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    plot_boxes(img, boxes, 'predictions.jpg', class_names)
