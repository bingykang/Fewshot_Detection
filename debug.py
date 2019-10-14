from __future__ import print_function
import torch.optim as optim
import os
import torch
import numpy as np
from darknet import Darknet
from PIL import Image
from utils import image2torch, convert2cpu
from torch.autograd import Variable

cfgfile = 'face4.1re_95.91.cfg'
weightfile = 'face4.1re_95.91.conv.15'
imgpath = 'data/train/images/10002.png'
labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
label = torch.zeros(50*5)
if os.path.getsize(labpath):
    tmp = torch.from_numpy(np.loadtxt(labpath))
    #tmp = torch.from_numpy(read_truths_args(labpath, 8.0/img.width))
    #tmp = torch.from_numpy(read_truths(labpath))
    tmp = tmp.view(-1)
    tsz = tmp.numel()
    #print('labpath = %s , tsz = %d' % (labpath, tsz))
    if tsz > 50*5:
        label = tmp[0:50*5]
    elif tsz > 0:
        label[0:tsz] = tmp
label = label.view(1, 50*5)

m = Darknet(cfgfile)
region_loss = m.loss
m.load_weights(weightfile)

print('--- bn weight ---')
print(m.models[0][1].weight)
print('--- bn bias ---')
print(m.models[0][1].bias)
print('--- bn running_mean ---')
print(m.models[0][1].running_mean)
print('--- bn running_var ---')
print(m.models[0][1].running_var)

m.train()
m = m.cuda()

optimizer = optim.SGD(m.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.1)

img = Image.open(imgpath)
img = image2torch(img)
img = Variable(img.cuda())

target = Variable(label)

print('----- img ---------------------')
print(img.data.storage()[0:100])
print('----- target  -----------------')
print(target.data.storage()[0:100])

optimizer.zero_grad()
output = m(img)
print('----- output ------------------')
print(output.data.storage()[0:100])
exit()

loss = region_loss(output, target)
print('----- loss --------------------')
print(loss)

save_grad = None
def extract(grad):
    global saved_grad
    saved_grad = convert2cpu(grad.data)

output.register_hook(extract)
loss.backward()

saved_grad = saved_grad.view(-1)
for i in xrange(saved_grad.size(0)):
    if abs(saved_grad[i]) >= 0.001:
        print('%d : %f' % (i, saved_grad[i]))

print(m.state_dict().keys())
#print(m.models[0][0].weight.grad.data.storage()[0:100])
#print(m.models[14][0].weight.data.storage()[0:100])
weight = m.models[13][0].weight.data
grad = m.models[13][0].weight.grad.data
mask = torch.abs(grad) >= 0.1
print(weight[mask])
print(grad[mask])

optimizer.step()
weight2 = m.models[13][0].weight.data
print(weight2[mask])
