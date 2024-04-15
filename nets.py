import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm


def upsample(x, h):
    return F.interpolate(input=x, size=(h, h), mode='bilinear', align_corners=False)


def add_avgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class MedianPooling(torch.nn.Module):
    def __init__(self, kernel_size=3):
        super(MedianPooling, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        padding = self.kernel_size // 2
        x_padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        pooled = torch.zeros(batch_size, channels, height, width, device=x.device)
        for h in range(height):
            for w in range(width):
                h_start = h
                h_end = h + self.kernel_size
                w_start = w
                w_end = w + self.kernel_size
                window = x_padded[:, :, h_start:h_end, w_start:w_end].contiguous().view(batch_size, channels, -1)
                median_values, _ = torch.median(window, dim=-1)
                pooled[:, :, h, w] = median_values
        return pooled


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=True),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=True))
        self.sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, in_planes // 4, kernel_size=1, bias=True)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes // 4, in_planes // 4, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(in_planes // 4),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(in_planes // 4, in_planes // 4, kernel_size=5, padding=2, bias=True),
                                   nn.BatchNorm2d(in_planes // 4),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(in_planes // 4, in_planes // 4, kernel_size=7, padding=3, bias=True),
                                   nn.BatchNorm2d(in_planes // 4),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.medianpool = MedianPooling()

    def forward(self, x):
        y = self.conv1(x)
        h = y.shape[3]
        x1 = self.conv2(y)

        x2 = self.conv3(y)

        x3 = self.conv4(y)

        x4 = self.conv5(y)
        x4 = upsample(x4, h)
        x4 = self.relu(x4)

        cat = torch.concat([x1, x2, x3, x4], 1)
        mult_result = self.medianpool(cat)
        max_result = self.max_pool(x)
        avg_result = self.avg_pool(x)

        max_out = self.fc(max_result)
        avg_out = self.fc(avg_result)
        mult_result = self.fc(mult_result)

        output = self.sigmoid(max_out + avg_out + mult_result)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(4, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(channel, channel // 4, kernel_size=1, bias=True)
        self.conv2 = nn.Sequential(nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=1, bias=True),
                                   nn.BatchNorm2d(channel // 4),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(channel // 4, channel // 4, kernel_size=5, padding=2, bias=True),
                                   nn.BatchNorm2d(channel // 4),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(channel // 4, channel // 4, kernel_size=7, padding=3, bias=True),
                                   nn.BatchNorm2d(channel // 4),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv1(x)
        h = y.shape[3]
        x1 = self.conv2(y)
        x2 = self.conv3(y)
        x3 = self.conv4(y)

        x4 = self.conv5(y)
        x4 = upsample(x4, h)
        x4 = self.relu(x4)

        cat = torch.concat([x1, x2, x3, x4], 1)

        max_scale, _ = torch.max(cat, dim=1, keepdim=True)
        avg_scale = torch.mean(cat, dim=1, keepdim=True)

        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result, max_scale, avg_scale], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class Conv2d(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, conv='conv', pad='mirror', norm='in', activ='relu',
                 sn=False):
        super(Conv2d, self).__init__()
        # Define padding
        if pad == 'mirror':
            self.padding = nn.ReflectionPad2d(kernel_size // 2)
        elif pad == 'none':
            self.padding = None
        else:
            self.padding = nn.ReflectionPad2d(pad)
        # Define conv layer
        if conv == 'conv':
            self.conv = nn.Conv2d(input_size, output_size, kernel_size=kernel_size, stride=stride)
        # Define norm layer
        if norm == 'in':
            self.norm = nn.InstanceNorm2d(output_size, affine=True)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(output_size)
        elif norm == 'none':
            self.norm = None
        # Define activation layer
        if activ == 'relu':
            self.relu = nn.ReLU()
        elif activ == 'leakyrelu':
            self.relu = nn.LeakyReLU(0.2)
        elif activ == 'none':
            self.relu = None
        # Use spectral norm
        if sn == True:
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        if self.padding:
            out = self.padding(x)
        else:
            out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        if self.relu:
            out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, input_size, kernel_size, stride, conv='conv', pad='mirror', norm='in', activ='relu', sn=False):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            Conv2d(input_size, input_size, kernel_size=kernel_size, stride=stride, conv=conv, pad=pad, norm=norm,
                   activ=activ, sn=sn),
            Conv2d(input_size, input_size, kernel_size=kernel_size, stride=stride, conv=conv, pad=pad, norm=norm,
                   activ=activ, sn=sn)
        )

        self.channel = ChannelAttention(in_planes=input_size)
        self.spacial = SpatialAttention(channel=input_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out1 = self.channel(out)
        out = out1 * out

        out2 = self.spacial(out)
        out = out2 * out

        out += residual
        out = self.relu(out)  # relu可以尝试不加
        return out


class Encoder(nn.Module):
    def __init__(self, input_size=3, activ='leakyrelu'):
        super(Encoder, self).__init__()
        self.conv_1 = Conv2d(input_size, 32, kernel_size=9, stride=1, activ=activ, sn=True)
        self.conv_2 = Conv2d(32, 64, kernel_size=3, stride=2, activ=activ, sn=True)
        self.conv_3 = Conv2d(64, 128, kernel_size=3, stride=2, activ=activ, sn=True)
        self.res_block = nn.Sequential(
            ResBlock(128, kernel_size=3, stride=1, activ=activ, sn=True),
            ResBlock(128, kernel_size=3, stride=1, activ=activ, sn=True),
            ResBlock(128, kernel_size=3, stride=1, activ=activ, sn=True),
            ResBlock(128, kernel_size=3, stride=1, activ=activ, sn=True)
        )

    def forward(self, x):
        out_1 = self.conv_1(x)
        out_2 = self.conv_2(out_1)
        out_3 = self.conv_3(out_2)
        out = self.res_block(out_3)
        return out, out_3, out_2


class Decoder(nn.Module):
    def __init__(self, output_size=3, activ='leakyrelu'):
        super(Decoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2d(256, 64, kernel_size=3, stride=1, activ=activ, sn=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2d(128, 32, kernel_size=3, stride=1, activ=activ, sn=True)
        )
        self.conv_3 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, output_size, kernel_size=9, stride=1)
        )

    def forward(self, x, age_vec, skip_1, skip_2):
        b, c = age_vec.size()
        age_vec = age_vec.view(b, c, 1, 1)
        out = age_vec * x
        out = torch.cat((out, skip_1), 1)
        out = self.conv_1(out)
        out = torch.cat((out, skip_2), 1)
        out = self.conv_2(out)
        out = self.conv_3(out)
        return out


class Mod_Age(nn.Module):
    def __init__(self):
        super(Mod_Age, self).__init__()
        self.fc_mix = nn.Sequential(
            nn.Linear(101, 128, bias=False),
            nn.Linear(128, 64, bias=False),
            nn.Linear(64, 128, bias=False)
        )

    def forward(self, x):
        b_s = x.size(0)
        z = torch.zeros(b_s, 101).type_as(x).float()
        for i in range(b_s):
            z[i, x[i]] = 1
        y = self.fc_mix(z)
        y = F.sigmoid(y)
        return y


class Mod_Race(nn.Module):
    def __init__(self):
        super(Mod_Race, self).__init__()
        self.fc_mix = nn.Sequential(
            nn.Linear(4, 128, bias=False),
            nn.Linear(128, 64, bias=False),
            nn.Linear(64, 128, bias=False)
        )

    def forward(self, x):
        b_s = x.size(0)
        z = torch.zeros(b_s, 4).type_as(x).float()
        for i in range(b_s):
            z[i, x[i]] = 1
        y = self.fc_mix(z)
        y = F.sigmoid(y)
        return y


class Mod_Gender(nn.Module):
    def __init__(self):
        super(Mod_Gender, self).__init__()
        self.fc_mix = nn.Sequential(
            nn.Linear(2, 128, bias=False),
            nn.Linear(128, 64, bias=False),
            nn.Linear(64, 128, bias=False)
        )

    def forward(self, x):
        b_s = x.size(0)
        z = torch.zeros(b_s, 2).type_as(x).float()
        for i in range(b_s):
            z[i, x[i]] = 1
        y = self.fc_mix(z)
        y = F.sigmoid(y)
        return y


class Dis_PatchGAN(nn.Module):
    def __init__(self, input_size=3):
        super(Dis_PatchGAN, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(input_size, 32, kernel_size=4, stride=2, norm='none', activ='leakyrelu', sn=True),
            Conv2d(32, 64, kernel_size=4, stride=2, norm='batch', activ='leakyrelu', sn=True),
            Conv2d(64, 128, kernel_size=4, stride=2, norm='batch', activ='leakyrelu', sn=True),
            Conv2d(128, 256, kernel_size=4, stride=2, norm='batch', activ='leakyrelu', sn=True),
            Conv2d(256, 512, kernel_size=4, stride=1, norm='batch', activ='leakyrelu', sn=True),
            Conv2d(512, 1, kernel_size=4, stride=1, norm='none', activ='none', sn=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.fc6 = nn.Linear(25088, 4096, bias=True)
        self.fc7 = nn.Linear(4096, 4096, bias=True)
        self.fc8_101 = nn.Linear(4096, 101, bias=True)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['p3'] = self.pool3(out['r33'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['p4'] = self.pool4(out['r43'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['p5'] = self.pool5(out['r53'])
        out['p5'] = out['p5'].view(out['p5'].size(0), -1)
        out['fc6'] = F.relu(self.fc6(out['p5']))
        out['fc7'] = F.relu(self.fc7(out['fc6']))
        out['fc8'] = self.fc8_101(out['fc7'])
        return out


if __name__ == "__main__":
    input1 = torch.rand(1, 3, 1024, 1024)
    model = Encoder()
    a, b, c = model(input1)
    print(a, b, c)
