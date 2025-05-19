import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import math

import src.params as params

# conv2d with tensorflow "same" padding
class Conv2dSame(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        padding = kwargs.pop("padding", "SAME")
        super().__init__(*args, **kwargs)
        self.padding = padding
        self.num_kernel_dims = 2
        self.forward_func = lambda input, padding: F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.groups
        )
    
    def calc_same_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.kernel_size[dim]
        
        dilate = self.dilation if isinstance(self.dilation, int) else self.dilation[dim]
        stride = self.stride if isinstance(self.stride, int) else self.stride[dim]
        
        kernel_size = (filter_size - 1) * dilate + 1
        out_size = (input_size + stride - 1) // stride
        padding = max(0, (out_size - 1) * stride + kernel_size - input_size)
        num_odd = int(padding % 2 != 0)
        return num_odd, padding
    
    def forward(self, input):
        if self.padding == "VALID":
            return self.forward_func(input, padding=0)
        odd_1, padding_1 = self.calc_same_padding(input, dim=0)
        odd_2, padding_2 = self.calc_same_padding(input, dim=1)
        if odd_1 or odd_2:
            input = F.pad(input, [0, odd_2, 0, odd_1])
        return self.forward_func(input, padding=[padding_1 // 2, padding_2 // 2])


class ConvBatchNormRelu(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv
        self.bn = nn.BatchNorm2d(conv.out_channels, eps=params.BATCHNORM_EPSILON)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv(nn.Module):
    def __init__(self, kernel, stride, input_dim, output_dim):
        super().__init__()
        self.fused = ConvBatchNormRelu(
            Conv2dSame(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=kernel,
                stride=stride,
                bias=False
            )
        )
    
    def forward(self, x):
        return self.fused(x)


class SeparableConv(nn.Module):
    def __init__(self, kernel, stride, input_dim, output_dim):
        super().__init__()
        self.depthwise_conv = ConvBatchNormRelu(
            Conv2dSame(
                in_channels=input_dim,
                out_channels=input_dim,
                groups=input_dim,
                kernel_size=kernel,
                stride=stride,
                padding="SAME",
                bias=False
            )
        )
        self.pointwise_conv = ConvBatchNormRelu(
            Conv2dSame(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=1,
                stride=1,
                padding="SAME",
                bias=False
            )
        )
    
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class YAMNet(nn.Module):
    def __init__(self, v3=False):
        super().__init__()
        self.v3 = v3
        
        if self.v3:
            self.backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
            self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.ToTensor()
            ])
            self.classifier = nn.Linear(1000, params.NUM_CLASSES, bias=True)
        else:
            yamnet_layer_configs = [
                # (layer_function, kernel, stride, num_filters)
                (Conv,          [3, 3], 2,   32),
                (SeparableConv, [3, 3], 1,   64),
                (SeparableConv, [3, 3], 2,  128),
                (SeparableConv, [3, 3], 1,  128),
                (SeparableConv, [3, 3], 2,  256),
                (SeparableConv, [3, 3], 1,  256),
                (SeparableConv, [3, 3], 2,  512),
                (SeparableConv, [3, 3], 1,  512),
                (SeparableConv, [3, 3], 1,  512),
                (SeparableConv, [3, 3], 1,  512),
                (SeparableConv, [3, 3], 1,  512),
                (SeparableConv, [3, 3], 1,  512),
                (SeparableConv, [3, 3], 2, 1024),
                (SeparableConv, [3, 3], 1, 1024)
            ]
            
            input_dim = 1
            self.layer_names = []
            for (i, (layer_func, kernel, stride, output_dim)) in enumerate(yamnet_layer_configs):
                name = f"layer_{i + 1}"
                self.add_module(name, layer_func(kernel, stride, input_dim, output_dim))
                input_dim = output_dim
                self.layer_names.append(name)
            
            self.classifier = nn.Linear(input_dim, params.NUM_CLASSES, bias=True)
    
    def forward(self, x, to_prob=False):
        if self.v3:
            x = x.repeat(1, 3, 1, 1)
            x = self.backbone(x)
            x = self.classifier(x)
        else:
            for name in self.layer_names:
                mod = getattr(self, name)
                x = mod(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.reshape(x.shape[0], -1)
            x = self.classifier(x)
            
        if to_prob: x = torch.sigmoid(x)
        return x
        