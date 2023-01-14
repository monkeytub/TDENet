import torch

from torch import nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.dropout import Dropout, Dropout2d


class ResConvBlock(nn.Module):  # resnet
    """
    Basic Block for ResConvDeconv
    """
    expansion = 1   #对应主分支中卷积核的个数有没有发生变化

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * ResConvBlock.expansion, kernel_size=(3, 3), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(out_channels * ResConvBlock.expansion),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * ResConvBlock.expansion, kernel_size=(3, 3), padding=(1, 1),
                      bias=False),
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # if the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if in_channels != ResConvBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * ResConvBlock.expansion, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels * ResConvBlock.expansion)#减小的是通道数
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResidualConvDeconv(nn.Module):
    """
    activation: 'relu', 'leaky', 'elu'
    normalization: 'batch', 'instance', 'group{group_size}'
    conv_mode: 'same', 'valid'
    dim: 2, 3

    Input: The XYImage captured by sonar

    Output: Height Image reconstructed by Network:
    """

    def __init__(self,
        in_channels: int = 1,
        out_channels: int = 1,
        activation: str = 'relu',
        normalization: str = 'batch',
    ):
        super().__init__()

        # init varibles
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.activation = activation

        # network structure
        # in conv, without residual function
        self.in_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.max_pool_argmax1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # 下采样

        self.res_conv1 = ResConvBlock(64, 128)
        self.max_pool_argmax2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.res_conv2 = ResConvBlock(128, 256)
        self.max_pool_argmax3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.res_conv3 = ResConvBlock(256, 512)
        self.max_pool_argmax4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.res_deconv3 = ResConvBlock(512, 256)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.res_deconv2 = ResConvBlock(256, 128)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)         # MaxUnpool2d->上采样
        self.res_deconv1 = ResConvBlock(128, 64)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.out_conv = nn.Sequential(
            nn.Conv2d(64*2 , 1, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self.init_parameters()
    

    def forward(self, x):
        # [batch_size, 1, 512, 512]
        x = self.in_conv(x)
        first_conv_output = x
        unpooled_shape1 = x.size()
        down1, indices_pool1 = self.max_pool_argmax1(x)# 两个参数？

        # [batch_size, 64, 256, 256]
        x = self.res_conv1(down1)
        unpooled_shape2 = x.size()
        down2, indices_pool2 = self.max_pool_argmax2(x)   # 两个参数
        # [batch_size, 128, 128, 128]
        x = self.res_conv2(down2)
        unpooled_shape3 = x.size()
        down3, indices_pool3 = self.max_pool_argmax3(x)
        # [batch_size, 256, 64, 64]
        x = self.res_conv3(down3)
        unpooled_shape4 = x.size()
        down4, indices_pool4 = self.max_pool_argmax4(x)
        # [batch_size, 512, 32, 32]

        up4 = self.unpool4(down4, indices=indices_pool4,output_size=unpooled_shape4)
        x = self.res_deconv3(up4)
        # [batch_size, 256, 64, 64]
        up3 = self.unpool3(x, indices=indices_pool3,output_size=unpooled_shape3)
        x = self.res_deconv2(up3)
        # [batch_size, 128, 128, 128]

        up2 = self.unpool2(x, indices=indices_pool2,output_size=unpooled_shape2)
        x = self.res_deconv1(up2)



        # [batch_size, 64, 256, 256]
        up1 = self.unpool1(x, indices=indices_pool1,output_size=unpooled_shape1)

        # [batch_size, 64, 512, 512]
        x = torch.cat([first_conv_output, up1], dim=1)
        output = self.out_conv(x)

        return output
    
    
    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(
            module, (nn.Conv2d, nn.Linear)
        ):
            method(module.weight, **kwargs)
    
    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(
            module, (nn.Conv2d, nn.Linear)
        ):
            method(module.bias, **kwargs)
    
    def init_parameters(
        self,
        method_weights=nn.init.xavier_uniform_,
        method_bias=nn.init.zeros_,
        kwargs_weights={},
        kwargs_bias={}
    ):
        for module in self.modules():
            self.weight_init(module, method_weights, **kwargs_weights)
            # self.bias_init(module, method_bias, **kwargs_bias)


if __name__ == "__main__":
    model=ResidualConvDeconv(1,1)
    print(model.parameters)
    print(model.max_pool_argmax1)