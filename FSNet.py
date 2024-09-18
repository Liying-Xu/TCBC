import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


# import common

###### Layer
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(Bottleneck, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # m['relu1'] = nn.ReLU(True)
        m['conv2'] = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        # m['relu2'] = nn.ReLU(True)
        m['conv3'] = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.group1 = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(True))

    def forward(self, x):
        out = self.group1(x)
        return out


class irnn_layer(nn.Module):
    def __init__(self, in_channels):
        super(irnn_layer, self).__init__()
        self.left_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.right_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.up_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.down_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)

    def forward(self, x):
        _, _, H, W = x.shape
        top_left = x.clone()
        top_right = x.clone()
        top_up = x.clone()
        top_down = x.clone()
        top_left[:, :, :, 1:] = F.relu(self.left_weight(x)[:, :, :, :W - 1] + x[:, :, :, 1:], inplace=False)
        top_right[:, :, :, :-1] = F.relu(self.right_weight(x)[:, :, :, 1:] + x[:, :, :, :W - 1], inplace=False)
        top_up[:, :, 1:, :] = F.relu(self.up_weight(x)[:, :, :H - 1, :] + x[:, :, 1:, :], inplace=False)
        top_down[:, :, :-1, :] = F.relu(self.down_weight(x)[:, :, 1:, :] + x[:, :, :H - 1, :], inplace=False)
        return (top_up, top_right, top_down, top_left)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.out_channels = int(in_channels / 2)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1, padding_mode='reflect', stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, padding_mode='reflect', stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels, 4, kernel_size=1, padding=0, stride=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out


class SAM(nn.Module):
    def __init__(self, in_channels, out_channels, attention=1):
        super(SAM, self).__init__()
        self.out_channels = out_channels
        self.irnn1 = irnn_layer(self.out_channels)
        self.irnn2 = irnn_layer(self.out_channels)
        self.conv_in = conv3x3(in_channels, self.out_channels)
        self.relu1 = nn.ReLU(True)

        self.conv1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.out_channels * 4, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(self.out_channels * 4, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        self.conv_out = conv1x1(self.out_channels, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv1(x)
        top_up, top_right, top_down, top_left = self.irnn1(out)

        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])
        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv2(out)
        top_up, top_right, top_down, top_left = self.irnn2(out)

        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])

        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv3(out)
        out = self.relu2(out)
        mask = self.conv_out(out)
        Att = self.sigmod(mask)
        return Att


class YTMTRelu(nn.Module):
    def __init__(self):
        super(YTMTRelu, self).__init__()
        self.relu = nn.ReLU()
        self.res_block_c = Bottleneck(32, 32)
        self.res_block_g = Bottleneck(32, 32)

    def forward(self, input_c, input_g):
        out_c = self.res_block_c(input_c) + input_c
        out_g = self.res_block_g(input_g) + input_g

        out_cp, out_cn = self.relu(out_c), out_c - self.relu(out_c)
        out_gp, out_gn = self.relu(out_g), out_g - self.relu(out_g)
        out_c = out_cp + out_gn
        out_g = out_gp + out_cn

        return out_c, out_g


class YTMTAttRelu(nn.Module):
    def __init__(self):
        super(YTMTAttRelu, self).__init__()
        self.relu = nn.ReLU()
        self.res_block_c = Bottleneck(32, 32)
        self.res_block_g = Bottleneck(32, 32)

    def forward(self, input_c, input_g, Att):
        out_c = self.res_block_c(input_c) * Att + input_c
        out_g = self.res_block_g(input_g) * Att + input_g

        out_cp, out_cn = self.relu(out_c), out_c - self.relu(out_c)
        out_gp, out_gn = self.relu(out_g), out_g - self.relu(out_g)
        out_c = out_cp + out_gn
        out_g = out_gp + out_cn

        return out_c, out_g


###### Network
class FSNet(nn.Module):
    def __init__(self):
        super(SABSNet, self).__init__()

        self.conv_in = nn.Sequential(
            conv3x3(5, 32),
            nn.ReLU(True)
        )
        self.SAM1 = SAM(64, 64, 1)
        self.conv_out_c = nn.Sequential(
            conv3x3(32, 5)
        )
        self.conv_out_g = nn.Sequential(
            conv3x3(32, 5)
        )
        self.ytmt_relu1 = YTMTRelu()
        self.ytmt_relu2 = YTMTRelu()
        self.ytmt_relu3 = YTMTRelu()
        self.ytmt_relu4 = YTMTRelu()
        self.ytmt_relu5 = YTMTRelu()
        #self.ytmt_relu6 = YTMTRelu()

        self.ytmt_attrelu1 = YTMTAttRelu()
        self.ytmt_attrelu2 = YTMTAttRelu()
        self.relu = nn.ReLU(True)

    def forward(self, x):
        outc = self.conv_in(x)
        outg = self.conv_in(x)

        outc, outg = self.ytmt_relu1(outc, outg)
        outc, outg = self.ytmt_relu2(outc, outg)
        outc, outg = self.ytmt_relu3(outc, outg)


        Att = self.SAM1(torch.cat([outc,outg],dim=1))
        # Att = self.SAM1(outg)
        outc, outg = self.ytmt_attrelu1(outc, outg, Att)
        outc, outg = self.ytmt_attrelu2(outc, outg, Att)
        # outc, outg = self.ytmt_attrelu3(outc, outg,Att)
        """
        outc,outg = self.ytmt_attrelu(outc,outg,Attg)

        Attg = self.SAM1(outg)
        outc,outg = self.ytmt_attrelu(outc,outg,Attg)
        outc,outg = self.ytmt_attrelu(outc,outg,Attg)
        outc,outg = self.ytmt_attrelu(outc,outg,Attg)

        Attg = self.SAM1(outg)
        outc, outg = self.ytmt_attrelu(outc, outg, Attg)
        outc, outg = self.ytmt_attrelu(outc, outg, Attg)
        outc, outg = self.ytmt_attrelu(outc, outg, Attg)

        Attg = self.SAM1(outg)
        outc, outg = self.ytmt_attrelu(outc, outg, Attg)
        outc, outg = self.ytmt_attrelu(outc, outg, Attg)
        outc, outg = self.ytmt_attrelu(outc, outg, Attg)
        """
        outc, outg = self.ytmt_relu4(outc, outg)
        outc, outg = self.ytmt_relu5(outc, outg)

        outc = self.conv_out_c(outc)
        outg = self.conv_out_g(outg)

        outc = self.relu(outc)
        outg = self.relu(outg)

        return outc, outg, Att