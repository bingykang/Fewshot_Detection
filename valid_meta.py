from darknet_meta import Darknet
import dataset
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import *
from cfg import cfg
from cfg import parse_cfg
import os
import pdb


def valid(datacfg, darknetcfg, learnetcfg, weightfile, outfile):
    options = read_data_cfg(datacfg)
    valid_images = options['valid']
    metadict = options['meta']
    # name_list = options['names']
    # backup = cfg.backup
    ckpt = weightfile.split('/')[-1].split('.')[0]
    backup = weightfile.split('/')[-2]
    prefix = 'results/' + backup.split('/')[-1] + '/e' + ckpt
    print('saving to: ' + prefix)
    # prefix = 'results/' + weightfile.split('/')[1]
    # names = load_class_names(name_list)

    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]
    
    m = Darknet(darknetcfg, learnetcfg)
    m.print_network()
    m.load_weights(weightfile)
    m.cuda()
    m.eval()

    valid_dataset = dataset.listDataset(valid_images, shape=(m.width, m.height),
                       shuffle=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]))
    valid_batchsize = 2
    assert(valid_batchsize > 1)

    kwargs = {'num_workers': 4, 'pin_memory': True}
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs) 

    metaset = dataset.MetaDataset(metafiles=metadict, train=False)
    metaloader = torch.utils.data.DataLoader(
        metaset,
        batch_size=metaset.batch_size,
        shuffle=False,
        **kwargs
    )
    metaloader = iter(metaloader)
    n_cls = len(metaset.classes)

    if not os.path.exists(prefix):
        # os.mkdir(prefix)
        os.makedirs(prefix)

    fps = [0]*n_cls
    for i, cls_name in enumerate(metaset.classes):
        buf = '%s/%s%s.txt' % (prefix, outfile, cls_name)
        fps[i] = open(buf, 'w')
   
    lineId = -1
    
    conf_thresh = 0.005
    nms_thresh = 0.45
    for batch_idx, (data, target) in enumerate(valid_loader):
        metax, mask = metaloader.next()
        # print(ids)
        data = data.cuda()
        mask = mask.cuda()
        metax = metax.cuda()
        data = Variable(data, volatile = True)
        mask = Variable(mask, volatile = True)
        metax = Variable(metax, volatile = True)
        output = m(data, metax, mask)

        if isinstance(output, tuple):
            output = (output[0].data, output[1].data)
        else:
            output = output.data
 
        batch_boxes = get_region_boxes_v2(output, n_cls, conf_thresh, m.num_classes, m.anchors, m.num_anchors, 0, 1)

        if isinstance(output, tuple):
            bs = output[0].size(0)
        else:
            assert output.size(0) % n_cls == 0
            bs = output.size(0) // n_cls

        for b in range(bs):
            lineId = lineId + 1
            imgpath = valid_dataset.lines[lineId].rstrip()
            print(imgpath)
            imgid = os.path.basename(imgpath).split('.')[0]
            width, height = get_image_size(imgpath)
            for i in range(n_cls):
                # oi = i * bs + b
                oi = b * n_cls + i
                boxes = batch_boxes[oi]
                boxes = nms(boxes, nms_thresh)
                for box in boxes:
                    x1 = (box[0] - box[2]/2.0) * width
                    y1 = (box[1] - box[3]/2.0) * height
                    x2 = (box[0] + box[2]/2.0) * width
                    y2 = (box[1] + box[3]/2.0) * height

                    det_conf = box[4]
                    for j in range((len(box)-5)/2):
                        cls_conf = box[5+2*j]
                        cls_id = box[6+2*j]
                        prob =det_conf * cls_conf
                        fps[i].write('%s %f %f %f %f %f\n' % (imgid, prob, x1, y1, x2, y2))

    for i in range(n_cls):
        fps[i].close()

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 5 or len(sys.argv) == 6:
        datacfg = sys.argv[1]
        darknet = parse_cfg(sys.argv[2])
        learnet = parse_cfg(sys.argv[3])
        weightfile = sys.argv[4]
        if len(sys.argv) == 6:
            gpu = sys.argv[5]
        else:
            gpu = '0'

        data_options  = read_data_cfg(datacfg)
        net_options   = darknet[0]
        meta_options  = learnet[0]
        data_options['gpus'] = gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

        # Configure options
        cfg.config_data(data_options)
        cfg.config_meta(meta_options)
        cfg.config_net(net_options)

        outfile = 'comp4_det_test_'
        valid(datacfg, darknet, learnet, weightfile, outfile)
    else:
        print('Usage:')
        print(' python valid.py datacfg cfgfile weightfile')
