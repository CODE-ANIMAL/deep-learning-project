import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    '''
    DoubleConv
    作用：该模块用于特征提取，通过两个连续的卷积操作，每个卷积后接一个批量归一化（Batch Normalization）层和 ReLU 激活函数。
    原因：批量归一化有助于加速收敛并保持梯度稳定；ReLU 激活函数引入非线性，帮助模型学习更复杂的特征。
    '''

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    '''
    Down
    作用：该模块包含一个最大池化层，用于减少空间维度，提取更高层次的特征，然后是一个 DoubleConv 模块。
    原因：最大池化可以减少特征图的尺寸，从而减少计算量，同时帮助模型捕获更高层次的特征。DoubleConv 提取局部特征。
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    '''
    Up
    作用：该模块负责逐步恢复空间细节，通过上采样操作（使用双线性插值或转置卷积）来增加特征图的尺寸，然后通过跳跃连接与相应的编码器层的特征图拼接，最后是一个 DoubleConv 模块。
    原因：上采样有助于恢复丢失的空间细节，跳跃连接则允许解码器直接访问编码器的早期特征图，这对于保留位置信息和低级特征非常重要。
    '''
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # 上采样，放大比例为2倍，使用双线性插值法，对齐输入输出的角点
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)  # 最后一个参数为什么？
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):  # x1: 上一层feature map, x2: skip connection from the encoder feature map
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]  # 高的差异
        diffX = x2.size()[3] - x1.size()[3]  # 宽的差异

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,  # x1填充至与x2相同宽高
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)  # concatenation
        return self.conv(x)


class OutConv(nn.Module):
    '''
    OutConv
    作用：该模块用于将解码器的最后一层特征图转换为与输入图像相同大小的分割图。
    原因：通过一个 1x1 的卷积层，将特征图转换为分割结果，每个像素点对应一个类别概率或类别标签。
    '''
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    '''
    UNet
    作用：该类组合了前面提到的所有组件，形成了一个完整的 U-Net 模型。
    原因：通过组合编码器和解码器结构，U-Net 能够有效地学习到图像的局部和全局特征，并通过跳跃连接将这些信息传递给输出层，从而提高分割精度。
    '''
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        '''
        DoubleConv作用是两个卷积，                              改变Chanel
        Down作用是通过最大池化层    减少图片尺寸    ，然后DoubleConv卷积2下     改变Chanel
        up：双线性插值法        扩大图片尺寸。DoubleConv卷积2次     改变Chanel。跳跃连接
        '''
        self.inc = DoubleConv(n_channels, 64)  # 输入通道数3，64个核   (2,512,512,3)---->（512,512,64）
        self.down1 = Down(64, 128)  # 512-->256。64核-->128核                           (256,256,128)
        self.down2 = Down(128, 256)   # 256-->128。128核-->256核                        (128,128,256)
        self.down3 = Down(256, 512)  # 128-->64。256核-->512核                          (64,64,512)
        factor = 2 if bilinear else 1  # factor = 2
        self.down4 = Down(512, 1024 // factor)  # 64-->32。512核-->512核                (32,32,1024)
        self.up1 = Up(1024, 512 // factor, bilinear)  #                                (64,64,512)
        self.up2 = Up(512, 256 // factor, bilinear)  #                                 (128,128,256)
        self.up3 = Up(256, 128 // factor, bilinear)  #                                 (256,256,128)
        self.up4 = Up(128, 64, bilinear)  #                                            (512,512,64)
        self.outc = OutConv(64, n_classes)  #                                          (512,512,2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
