from darknet import Darknet
import dataset
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import *
from cfg import cfg
from cfg import parse_cfg
import os


def valid(datacfg, cfgfile, weightfile, outfile):
    options = read_data_cfg(datacfg)
    valid_images = options['valid']
    # backup = cfg.backup
    backup = weightfile.split('/')[-2]
    ckpt = weightfile.split('/')[-1].split('.')[0]
    prefix = 'results/' + backup.split('/')[-1] + '/e' + ckpt
    print('saving to: ' + prefix)
    names = cfg.classes

    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]
    
    m = Darknet(cfgfile)
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

    fps = [0]*m.num_classes
    if not os.path.exists(prefix):
        # os.mkdir(prefix)
        os.makedirs(prefix)
    for i in range(m.num_classes):
        buf = '%s/%s%s.txt' % (prefix, outfile, names[i])
        fps[i] = open(buf, 'w')
   
    lineId = -1
    
    conf_thresh = 0.005
    nms_thresh = 0.45
    for batch_idx, (data, target) in enumerate(valid_loader):
        data = data.cuda()
        data = Variable(data, volatile = True)
        output = m(data).data
        batch_boxes = get_region_boxes(output, conf_thresh, m.num_classes, m.anchors, m.num_anchors, 0, 1)
        for i in range(output.size(0)):
            lineId = lineId + 1
            fileId = os.path.basename(valid_files[lineId]).split('.')[0]
            width, height = get_image_size(valid_files[lineId])
            print(valid_files[lineId])
            boxes = batch_boxes[i]
            boxes = nms(boxes, nms_thresh)
            for box in boxes:
                x1 = (box[0] - box[2]/2.0) * width
                y1 = (box[1] - box[3]/2.0) * height
                x2 = (box[0] + box[2]/2.0) * width
                y2 = (box[1] + box[3]/2.0) * height

                det_conf = box[4]
                # import pdb
                # pdb.set_trace()
                for j in range((len(box)-5)/2):
                    cls_conf = box[5+2*j]
                    cls_id = box[6+2*j]
                    prob =det_conf * cls_conf
                    fps[cls_id].write('%s %f %f %f %f %f\n' % (fileId, prob, x1, y1, x2, y2))
                    # fps[cls_id].write('%s %f %f %f %f %f %f\n' % (fileId, det_conf, cls_conf, x1, y1, x2, y2))

    for i in range(m.num_classes):
        fps[i].close()

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 4 or len(sys.argv) == 5:
        datacfg = sys.argv[1]
        cfgfile = sys.argv[2]
        weightfile = sys.argv[3]
        if len(sys.argv) == 5:
            gpu = sys.argv[4]
        else:
            gpu = '0'

        data_options  = read_data_cfg(datacfg)
        net_options   = parse_cfg(cfgfile)[0]
        data_options['gpus'] = gpu
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        # Configure options
        cfg.config_data(data_options)
        cfg.config_net(net_options)

        outfile = 'comp4_det_test_'
        valid(datacfg, cfgfile, weightfile, outfile)
    else:
        print('Usage:')
        print(' python valid.py datacfg cfgfile weightfile')
