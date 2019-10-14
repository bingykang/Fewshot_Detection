import os
import os.path
from PIL import Image
import sys
sys.path.append('.')
from darknet import Darknet
from utils import do_detect, plot_boxes, load_class_names

def save_boxes(img, boxes, savename):
    fp = open(savename, 'w')
    filename = os.path.basename(savename)
    filename = os.path.splitext(filename)[0]
    fp.write('%s\n' % filename)
    fp.write('%d\n' % len(boxes))
    width = img.width
    height = img.height
    for box in boxes:
        x1 = round((box[0] - box[2]/2.0) * width)
        y1 = round((box[1] - box[3]/2.0) * height)
        x2 = round((box[0] + box[2]/2.0) * width)
        y2 = round((box[1] + box[3]/2.0) * height)
        w = x2 - x1
        h = y2 - y1
        conf = box[4]
        fp.write('%d %d %d %d %f\n' % (x1, y1, w, h, conf))
    fp.close()

def eval_widerface(cfgfile, weightfile, valdir, savedir):
    m = Darknet(cfgfile)
    m.load_weights(weightfile)
    use_cuda = 1
    if use_cuda:
        m.cuda()

    scale_size = 16
    class_names = load_class_names('data/names')
    for parent,dirnames,filenames in os.walk(valdir):
        if parent != valdir:
            targetdir = os.path.join(savedir, os.path.basename(parent))
            if not os.path.isdir(targetdir):
                os.mkdir(targetdir)
            for filename in filenames:
                imgfile = os.path.join(parent,filename)
                img = Image.open(imgfile).convert('RGB')
                sized_width = int(round(img.width*1.0/scale_size) * 16)
                sized_height = int(round(img.height*1.0/scale_size) * 16)
                sized = img.resize((sized_width, sized_height))
                print(filename, img.width, img.height, sized_width, sized_height)
                if sized_width * sized_height > 1024 * 2560:
                    print('omit %s' % filename)
                    continue
                boxes = do_detect(m, sized, 0.05, 0.4, use_cuda)
                if True:
                    savename = os.path.join(targetdir, filename)
                    print('save to %s' % savename)
                    plot_boxes(img, boxes, savename, class_names)
                if True:
                    savename = os.path.join(targetdir, os.path.splitext(filename)[0]+".txt")
                    print('save to %s' % savename)
                    save_boxes(img, boxes, savename)

if __name__ == '__main__':
    #eval_widerface('resnet50_test.cfg', 'resnet50_98000.weights', 'widerface/WIDER_val/images/', 'widerface/wider_val_pred/')
    #eval_widerface('resnet50_test.cfg', 'resnet50_148000.weights', 'widerface/WIDER_val/images/', 'widerface/wider_val_pred/')
    eval_widerface('resnet50_x32_test.cfg', 'resnet50_x32_288000.weights', 'widerface/WIDER_val/images/', 'widerface/wider_val_pred/')

