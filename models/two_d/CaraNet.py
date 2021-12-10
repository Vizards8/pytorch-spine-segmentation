# https://github.com/AngeLouCN/CaraNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.two_d.Res2Net import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s
# from pretrain.Res2Net_v1b import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s
import math
# from lib.conv_layer import Conv, BNPReLU
# from lib.axial_atten import AA_kernel
# from lib.context_module import CFPModule
# from lib.partial_decoder import aggregation
import os


## --------------conv_layer-----------------
class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


## ----------------self_attention----------------------
class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1), stride=1, padding=0)
        self.key_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1), stride=1, padding=0)
        self.value_conv = Conv(in_channels, in_channels, kSize=(1, 1), stride=1, padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out


## ------------------axial_atten-------------------------
class AA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AA_kernel, self).__init__()
        self.conv0 = Conv(in_channel, out_channel, kSize=1, stride=1, padding=0)
        self.conv1 = Conv(out_channel, out_channel, kSize=(3, 3), stride=1, padding=1)
        self.Hattn = self_attn(out_channel, mode='h')
        self.Wattn = self_attn(out_channel, mode='w')

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(Hx)

        return Wx


## ----------------context_module----------------------
class CFPModule(nn.Module):
    def __init__(self, nIn, d=1, KSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.bn_relu_2 = BNPReLU(nIn)
        self.conv1x1_1 = Conv(nIn, nIn // 4, KSize, 1, padding=1, bn_acti=True)

        self.dconv_4_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),
                              dilation=(d + 1, d + 1), groups=nIn // 16, bn_acti=True)

        self.dconv_4_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),
                              dilation=(d + 1, d + 1), groups=nIn // 16, bn_acti=True)

        self.dconv_4_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(1 * d + 1, 1 * d + 1),
                              dilation=(d + 1, d + 1), groups=nIn // 16, bn_acti=True)

        self.dconv_1_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(1, 1),
                              dilation=(1, 1), groups=nIn // 16, bn_acti=True)

        self.dconv_1_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(1, 1),
                              dilation=(1, 1), groups=nIn // 16, bn_acti=True)

        self.dconv_1_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(1, 1),
                              dilation=(1, 1), groups=nIn // 16, bn_acti=True)

        self.dconv_2_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),
                              dilation=(int(d / 4 + 1), int(d / 4 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_2_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),
                              dilation=(int(d / 4 + 1), int(d / 4 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_2_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(int(d / 4 + 1), int(d / 4 + 1)),
                              dilation=(int(d / 4 + 1), int(d / 4 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_3_1 = Conv(nIn // 4, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),
                              dilation=(int(d / 2 + 1), int(d / 2 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_3_2 = Conv(nIn // 16, nIn // 16, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),
                              dilation=(int(d / 2 + 1), int(d / 2 + 1)), groups=nIn // 16, bn_acti=True)

        self.dconv_3_3 = Conv(nIn // 16, nIn // 8, (dkSize, dkSize), 1, padding=(int(d / 2 + 1), int(d / 2 + 1)),
                              dilation=(int(d / 2 + 1), int(d / 2 + 1)), groups=nIn // 16, bn_acti=True)

        self.conv1x1 = Conv(nIn, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        inp = self.bn_relu_1(input)
        inp = self.conv1x1_1(inp)

        o1_1 = self.dconv_1_1(inp)
        o1_2 = self.dconv_1_2(o1_1)
        o1_3 = self.dconv_1_3(o1_2)

        o2_1 = self.dconv_2_1(inp)
        o2_2 = self.dconv_2_2(o2_1)
        o2_3 = self.dconv_2_3(o2_2)

        o3_1 = self.dconv_3_1(inp)
        o3_2 = self.dconv_3_2(o3_1)
        o3_3 = self.dconv_3_3(o3_2)

        o4_1 = self.dconv_4_1(inp)
        o4_2 = self.dconv_4_2(o4_1)
        o4_3 = self.dconv_4_3(o4_2)

        output_1 = torch.cat([o1_1, o1_2, o1_3], 1)
        output_2 = torch.cat([o2_1, o2_2, o2_3], 1)
        output_3 = torch.cat([o3_1, o3_2, o3_3], 1)
        output_4 = torch.cat([o4_1, o4_2, o4_3], 1)

        ad1 = output_1
        ad2 = ad1 + output_2
        ad3 = ad2 + output_3
        ad4 = ad3 + output_4
        output = torch.cat([ad1, ad2, ad3, ad4], 1)
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        return output + input


## -----------------partial_decoder----------------------
class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = Conv(32, 32, 3, 1, padding=1)
        self.conv_upsample2 = Conv(32, 32, 3, 1, padding=1)
        self.conv_upsample3 = Conv(32, 32, 3, 1, padding=1)
        self.conv_upsample4 = Conv(32, 32, 3, 1, padding=1)
        self.conv_upsample5 = Conv(2 * 32, 2 * 32, 3, 1, padding=1)

        self.conv_concat2 = Conv(2 * 32, 2 * 32, 3, 1, padding=1)
        self.conv_concat3 = Conv(3 * 32, 3 * 32, 3, 1, padding=1)
        self.conv4 = Conv(3 * 32, 3 * 32, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(3 * 32, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


## ------------------CaraNet----------------------


class caranet(nn.Module):
    def __init__(self, channel=32, num_classes=20):
        super().__init__()

        # ---- ResNet Backbone ----
        self.resnet = res2net101_v1b_26w_4s(pretrained=False)

        # Receptive Field Block
        self.rfb2_1 = Conv(512, 32, 3, 1, padding=1, bn_acti=True)
        self.rfb3_1 = Conv(1024, 32, 3, 1, padding=1, bn_acti=True)
        self.rfb4_1 = Conv(2048, 32, 3, 1, padding=1, bn_acti=True)

        # Partial Decoder
        self.agg1 = aggregation(channel)

        self.CFP_1 = CFPModule(32, d=8)
        self.CFP_2 = CFPModule(32, d=8)
        self.CFP_3 = CFPModule(32, d=8)
        ###### dilation rate 4, 62.8

        self.ra1_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.ra2_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.ra3_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)

        self.aa_kernel_1 = AA_kernel(32, 32)
        self.aa_kernel_2 = AA_kernel(32, 32)
        self.aa_kernel_3 = AA_kernel(32, 32)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(1, num_classes, 1)
        self.final_2 = nn.Conv2d(1, num_classes, 1)
        self.final_3 = nn.Conv2d(1, num_classes, 1)
        self.final_5 = nn.Conv2d(1, num_classes, 1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 88, 88

        # ----------- low-level features -------------

        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11

        x2_rfb = self.rfb2_1(x2)  # 512 - 32
        x3_rfb = self.rfb3_1(x3)  # 1024 - 32
        x4_rfb = self.rfb4_1(x4)  # 2048 - 32

        decoder_1 = self.agg1(x4_rfb, x3_rfb, x2_rfb)  # 1,44,44
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=4, mode='bilinear')

        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.25, mode='bilinear')
        cfp_out_1 = self.CFP_3(x4_rfb)  # 32 - 32
        decoder_2_ra = -1 * (torch.sigmoid(decoder_2)) + 1
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        aa_atten_3_o = decoder_2_ra.expand(-1, 32, -1, -1).mul(aa_atten_3)

        ra_3 = self.ra3_conv1(aa_atten_3_o)  # 32 - 32
        ra_3 = self.ra3_conv2(ra_3)  # 32 - 32
        ra_3 = self.ra3_conv3(ra_3)  # 32 - 1

        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3, scale_factor=16, mode='bilinear')

        # ------------------- atten-two -----------------------
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear')
        cfp_out_2 = self.CFP_3(x3_rfb)  # 32 - 32
        decoder_3_ra = -1 * (torch.sigmoid(decoder_3)) + 1
        aa_atten_2 = self.aa_kernel_2(cfp_out_2)
        aa_atten_2_o = decoder_3_ra.expand(-1, 32, -1, -1).mul(aa_atten_2)

        ra_2 = self.ra2_conv1(aa_atten_2_o)  # 32 - 32
        ra_2 = self.ra2_conv2(ra_2)  # 32 - 32
        ra_2 = self.ra2_conv3(ra_2)  # 32 - 1

        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2, scale_factor=8, mode='bilinear')

        # ------------------- atten-three -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear')
        cfp_out_3 = self.CFP_1(x2_rfb)  # 32 - 32
        decoder_4_ra = -1 * (torch.sigmoid(decoder_4)) + 1
        aa_atten_1 = self.aa_kernel_1(cfp_out_3)
        aa_atten_1_o = decoder_4_ra.expand(-1, 32, -1, -1).mul(aa_atten_1)

        ra_1 = self.ra1_conv1(aa_atten_1_o)  # 32 - 32
        ra_1 = self.ra1_conv2(ra_1)  # 32 - 32
        ra_1 = self.ra1_conv3(ra_1)  # 32 - 1

        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1, scale_factor=4, mode='bilinear')

        # return lateral_map_5, lateral_map_3, lateral_map_2, lateral_map_1
        # final layer
        final_1 = self.final_1(lateral_map_1)
        final_2 = self.final_2(lateral_map_2)
        final_3 = self.final_3(lateral_map_3)
        final_5 = self.final_5(lateral_map_5)
        return (final_5 + final_3 + final_2 + final_1) / 4


if __name__ == '__main__':
    ras = caranet(num_classes=20).cuda()
    input_tensor = torch.randn(1, 1, 176, 176).cuda()

    from torchsummary import summary

    print(summary(ras, (1, 176, 176)))
    out = ras(input_tensor)
    # while True:
    print(out.shape)
