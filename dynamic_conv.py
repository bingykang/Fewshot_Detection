import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
import pdb


class _ConvNd(nn.Module):
    """https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d"""

    partial = None

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        '''
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        '''
        if self.partial is not None:
            assert self.partial <= self.out_channels
            self.weight = Parameter(torch.Tensor(
                self.partial, *kernel_size))
        else:
            self.register_parameter('weight', None)
        # if bias:
        #     self.bias = Parameter(torch.Tensor(out_channels))
        # else:
        self.register_parameter('bias', None)
        self.reset_parameters()
        
    
    def reset_parameters(self):
        if self.partial is not None:
            n = self.partial
            for k in self.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n)
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


# class DynamicConv2d(_ConvNd):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=False):
#         # assert(in_channels == out_channels)
#         kernel_size = _pair(kernel_size)
#         stride = _pair(stride)
#         padding = _pair(padding)
#         dilation = _pair(dilation)
#         super(DynamicConv2d, self).__init__(
#                 in_channels, out_channels, kernel_size, stride, padding, dilation,
#                 False, _pair(0), groups, bias)

#     def forward(self, inputs):
#         input, dynamic_weight = inputs
#         assert tuple(dynamic_weight.size())[-2:] == self.kernel_size
#         # Get batch size
#         batch_size = input.size(0)
#         n_channels = input.size(1)
#         groups = batch_size * n_channels

#         # Reshape input tensor from size (N, C, H, W) to (1, N*C, H, W)
#         input = input.view(1, -1, input.size(2), input.size(3))
#         # Reshape dynamic_weight tensor from size (N, C, H, W) to (1, N*C, H, W)
#         dynamic_weight = dynamic_weight.view(-1, 1, dynamic_weight.size(2), dynamic_weight.size(3))
#         # Do convolution
#         conv_rlt = F.conv2d(input, dynamic_weight, self.bias, self.stride,
#                         self.padding, self.dilation, groups)
#         # Reshape conv_rlt tensor from (1, N*C, H, W) to (N, C, H, W)
#         conv_rlt = conv_rlt.view(batch_size, -1, conv_rlt.size(2), conv_rlt.size(3))
#         return conv_rlt

def dynamic_conv2d(is_first, partial=None):

    class DynamicConv2d(_ConvNd):
        is_first = None
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, groups=1, bias=False):
            # assert(in_channels == out_channels)nami
            kernel_size = _pair(kernel_size)
            stride = _pair(stride)
            padding = _pair(padding)
            dilation = _pair(dilation)
            super(DynamicConv2d, self).__init__(
                    in_channels, out_channels, kernel_size, stride, padding, dilation,
                    False, _pair(0), groups, bias)

        def forward(self, inputs):
            assert self.is_first is not None, 'Please set the state of DynamicConv2d first.'
            # pdb.set_trace()
            input, dynamic_weight = inputs
            assert tuple(dynamic_weight.size())[-2:] == self.kernel_size
            assert dynamic_weight.size(1) % input.size(1) == 0
            n_cls = dynamic_weight.size(0)

            # Take care of partial prediction
            if self.partial is not None:
                shared_weight = self.weight.repeat(n_cls, 1, 1, 1)
                dynamic_weight = torch.cat([shared_weight, dynamic_weight], dim=1)

            if self.is_first:
                # Get batch size
                batch_size = input.size(0)
                n_channels = input.size(1)
                # input tensor (N, C, H, W) -> (N, C*n_cls, H, W)
                input = input.repeat(1, n_cls, 1, 1)
            else:
                assert input.size(0) % n_cls == 0, "Input batch size does not match with n_cls"
                batch_size = input.size(0) // n_cls
                n_channels = input.size(1)
                in_size = (input.size(-2), input.size(-1))
                input = input.view(batch_size, n_cls*n_channels, *in_size)

            # Get group size
            group_size = dynamic_weight.size(1) // n_channels
            # Calculate the number of channels 
            groups = n_cls * n_channels // group_size
            # Reshape dynamic_weight tensor from size (N, C, H, W) to (N*C, 1, H, W)
            dynamic_weight = dynamic_weight.view(-1, group_size, dynamic_weight.size(2), dynamic_weight.size(3))

            conv_rlt = F.conv2d(input, dynamic_weight, self.bias, self.stride,
                            self.padding, self.dilation, groups)

            feat_size = (conv_rlt.size(-2), conv_rlt.size(-1))
            conv_rlt = conv_rlt.view(-1, n_channels, *feat_size)

            return conv_rlt

    DynamicConv2d.is_first = is_first
    DynamicConv2d.partial = partial
    return DynamicConv2d
