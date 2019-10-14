import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.autograd import Function
import bn_lib

class BN2dFunc(Function):
    def __init__(self, running_mean, running_var, training, momentum, eps):
        self.running_mean = running_mean
        self.running_var = running_var
        self.training = training
        self.momentum = momentum
        self.eps = eps

    def forward(self, input, weight, bias):
        nB = input.size(0)
        nC = input.size(1)
        nH = input.size(2)
        nW = input.size(3)

        output = input.new(nB, nC, nH, nW) 
        self.input = input
        self.weight = weight
        self.bias = bias
        self.x = input.new(nB, nC, nH, nW) 
        self.x_norm = input.new(nB, nC, nH, nW) 
        self.mean = input.new(nB, nC) 
        self.var = input.new(nB, nC) 

        if input.is_cuda:
            bn_lib.bn_forward_gpu(input, self.x, self.x_norm, self.mean, self.running_mean, self.var, self.running_var, weight, bias, self.training, output)
        else:
            bn_lib.bn_forward(input, self.x, self.x_norm, self.mean, self.running_mean, self.var, self.running_var, weight, bias, self.training, output)
        return output

    def backward(self, grad_output):
        nB = grad_output.size(0)
        nC = grad_output.size(1)
        nH = grad_output.size(2)
        nW = grad_output.size(3)
        grad_input = grad_output.new(nB, nC, nH, nW) 
        grad_mean = grad_output.new(nC) 
        grad_var = grad_output.new(nC) 
        grad_weight = grad_output.new(nC) 
        grad_bias = grad_output.new(nC) 
        
        if grad_output.is_cuda:
            bn_lib.bn_backward_gpu(grad_output, self.input, self.x_norm, self.mean, grad_mean, self.var, grad_var, self.weight, grad_weight, self.bias, grad_bias, self.training, grad_input)
        else:
            bn_lib.bn_backward(grad_output, self.input, self.x_norm, self.mean, grad_mean, self.var, grad_var, self.weight, grad_weight, self.bias, grad_bias, self.training, grad_input)
        
        return grad_input, grad_weight, grad_bias  

class BN2d(nn.Module):
    def __init__(self, num_features, momentum=0.01, eps=1e-5):
        super(BN2d, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.momentum = momentum
        self.eps = eps

        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, input):
        #print('------------ BN2d input -------------')
        #print(input.data.storage()[0:10])
        return BN2dFunc(self.running_mean, self.running_var, self.training, self.momentum, self.eps)(input, self.weight, self.bias)

class BN2d_slow(nn.Module):
    def __init__(self, num_features, momentum=0.01):
        super(BN2d_slow, self).__init__()
        self.num_features = num_features
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.eps = 1e-5
        self.momentum = momentum

        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.weight.data.uniform_()
        self.bias.data.zero_()
    def forward(self, x): 
        nB = x.data.size(0)
        nC = x.data.size(1)
        nH = x.data.size(2)
        nW = x.data.size(3)
        samples = nB*nH*nW
        y = x.view(nB, nC, nH*nW).transpose(1,2).contiguous().view(-1,nC)
        if self.training:
            print('forward in training mode on autograd')
            m = Variable(y.mean(0).data, requires_grad=False)
            v = Variable(y.var(0).data, requires_grad=False)
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum * m.data.view(-1)
            self.running_var = (1-self.momentum)*self.running_var + self.momentum * v.data.view(-1)
            m = m.repeat(samples, 1)
            v = v.repeat(samples, 1)*(samples-1.0)/samples
        else:
            m = Variable(self.running_mean.repeat(samples, 1), requires_grad=False)
            v = Variable(self.running_var.repeat(samples, 1), requires_grad=False)
        w = self.weight.repeat(samples, 1)
        b = self.bias.repeat(samples, 1)
        y = (y - m)/(v+self.eps).sqrt() * w + b 
        y = y.view(nB, nH*nW, nC).transpose(1,2).contiguous().view(nB, nC, nH, nW) 
        return y


if __name__ == '__main__':
    nB = 64
    nC = 3
    nH = 4
    nW = 4
    samples = nB*nH*nW
    a = torch.rand(nB,nC,nH,nW)
    a = Variable(a)
    nn_model  = nn.BatchNorm2d(nC)
    dkn_model = BN2d(nC)
    atg_model = BN2d_slow(nC)

    nn_model.weight.data.fill_(1.0)
    nn_model.bias.data.zero_()
    dkn_model.weight.data.fill_(1.0)
    dkn_model.bias.data.zero_()
    atg_model.weight.data.fill_(1.0)
    atg_model.bias.data.zero_()
    nn_out_cpu = nn_model(a)
    dkn_out_cpu = dkn_model(a)
    atg_out_cpu = atg_model(a)



    a = a.cuda()
    nn_model.cuda()
    dkn_model.cuda()
    atg_model.cuda()

    nn_out_gpu = nn_model(a)
    dkn_out_gpu = dkn_model(a)
    atg_out_gpu = atg_model(a)

    print('--- nn cpu out ---')
    print(nn_out_cpu.data.storage()[0:10])
    print('--- dkn cpu out ---')
    print(dkn_out_cpu.data.storage()[0:10])
    print('--- atg cpu out ---')
    print(atg_out_cpu.data.storage()[0:10])


    print('--- nn gpu out ---')
    print(nn_out_gpu.data.storage()[0:10])
    print('--- dkn gpu out ---')
    print(dkn_out_gpu.data.storage()[0:10])
    print('--- atg gpu out ---')
    print(atg_out_gpu.data.storage()[0:10])
