import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from cfg import *
from darknet import MaxPoolStride1
from region_loss import RegionLoss

class TinyYoloNet(nn.Module):
    def __init__(self):
        super(TinyYoloNet, self).__init__()
        self.seen = 0
        self.num_classes = 20
        self.anchors = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]
        self.num_anchors = len(self.anchors)/2
        num_output = (5+self.num_classes)*self.num_anchors
        self.width = 160
        self.height = 160

        self.loss = RegionLoss(self.num_classes, self.anchors, self.num_anchors)
        self.cnn = nn.Sequential(OrderedDict([
            # conv1
            ('conv1', nn.Conv2d( 3, 16, 3, 1, 1, bias=False)),
            ('bn1', nn.BatchNorm2d(16)),
            ('leaky1', nn.LeakyReLU(0.1, inplace=True)),
            ('pool1', nn.MaxPool2d(2, 2)),

            # conv2
            ('conv2', nn.Conv2d(16, 32, 3, 1, 1, bias=False)),
            ('bn2', nn.BatchNorm2d(32)),
            ('leaky2', nn.LeakyReLU(0.1, inplace=True)),
            ('pool2', nn.MaxPool2d(2, 2)),

            # conv3
            ('conv3', nn.Conv2d(32, 64, 3, 1, 1, bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('leaky3', nn.LeakyReLU(0.1, inplace=True)),
            ('pool3', nn.MaxPool2d(2, 2)),

            # conv4
            ('conv4', nn.Conv2d(64, 128, 3, 1, 1, bias=False)),
            ('bn4', nn.BatchNorm2d(128)),
            ('leaky4', nn.LeakyReLU(0.1, inplace=True)),
            ('pool4', nn.MaxPool2d(2, 2)),

            # conv5
            ('conv5', nn.Conv2d(128, 256, 3, 1, 1, bias=False)),
            ('bn5', nn.BatchNorm2d(256)),
            ('leaky5', nn.LeakyReLU(0.1, inplace=True)),
            ('pool5', nn.MaxPool2d(2, 2)),

            # conv6
            ('conv6', nn.Conv2d(256, 512, 3, 1, 1, bias=False)),
            ('bn6', nn.BatchNorm2d(512)),
            ('leaky6', nn.LeakyReLU(0.1, inplace=True)),
            ('pool6', MaxPoolStride1()),

            # conv7
            ('conv7', nn.Conv2d(512, 1024, 3, 1, 1, bias=False)),
            ('bn7', nn.BatchNorm2d(1024)),
            ('leaky7', nn.LeakyReLU(0.1, inplace=True)),

            # conv8
            ('conv8', nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)),
            ('bn8', nn.BatchNorm2d(1024)),
            ('leaky8', nn.LeakyReLU(0.1, inplace=True)),

            # output
            ('output', nn.Conv2d(1024, num_output, 1, 1, 0)),
        ]))

    def forward(self, x):
        x = self.cnn(x)
        return x

    def print_network(self):
        print(self)

    def load_weights(self, path):
        #buf = np.fromfile('tiny-yolo-voc.weights', dtype = np.float32)
        buf = np.fromfile(path, dtype = np.float32)
        start = 4
        
        start = load_conv_bn(buf, start, self.cnn[0], self.cnn[1])
        start = load_conv_bn(buf, start, self.cnn[4], self.cnn[5])
        start = load_conv_bn(buf, start, self.cnn[8], self.cnn[9])
        start = load_conv_bn(buf, start, self.cnn[12], self.cnn[13])
        start = load_conv_bn(buf, start, self.cnn[16], self.cnn[17])
        start = load_conv_bn(buf, start, self.cnn[20], self.cnn[21])
        
        start = load_conv_bn(buf, start, self.cnn[24], self.cnn[25])
        start = load_conv_bn(buf, start, self.cnn[27], self.cnn[28])
        start = load_conv(buf, start, self.cnn[30])

if __name__ == '__main__':
    from PIL import Image
    from utils import *
    m = TinyYoloNet() 
    m.float()
    m.eval()
    m.load_darknet_weights('tiny-yolo-voc.weights')
    print(m)
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = Image.open('data/person.jpg').convert('RGB')
    sized = img.resize((416,416))
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)

    class_names = load_class_names('data/voc.names')
    plot_boxes(img, boxes, 'predict1.jpg', class_names)  

