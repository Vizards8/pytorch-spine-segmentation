# https://github.com/sacmehta/EdgeNets
import torch
from torch.nn import init
from nn_layers.espnet_utils import *
from nn_layers.efficient_pyramid_pool import EfficientPyrPool
from nn_layers.efficient_pt import EfficientPWConv
from model.classification.espnetv2 import EESPNet
from utilities.print_utils import *
from torch.nn import functional as F


## --------espnet_utils-----------------
import torch.nn as nn

#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================

class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        # output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''

    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output


class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False,
                              groups=groups)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False,
                              groups=groups)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut,kSize, stride=stride, padding=padding, bias=False,
                              dilation=d, groups=groups)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class CDilatedB(nn.Module):
    '''
    This class defines the dilated convolution with batch normalization.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut,kSize, stride=stride, padding=padding, bias=False,
                              dilation=d, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        return self.bn(self.conv(input))

## ---------efficient_pyramid_pool----------------



class ESPNetv2Segmentation(nn.Module):
    '''
    This class defines the ESPNetv2 architecture for the Semantic Segmenation
    '''

    def __init__(self, args, classes=21, dataset='pascal'):
        super().__init__()

        # =============================================================
        #                       BASE NETWORK
        # =============================================================
        self.base_net = EESPNet(args) #imagenet model
        del self.base_net.classifier
        del self.base_net.level5
        del self.base_net.level5_0
        config = self.base_net.config

        #=============================================================
        #                   SEGMENTATION NETWORK
        #=============================================================
        dec_feat_dict={
            'pascal': 16,
            'city': 16,
            'coco': 32
        }
        base_dec_planes = dec_feat_dict[dataset]
        dec_planes = [4*base_dec_planes, 3*base_dec_planes, 2*base_dec_planes, classes]
        pyr_plane_proj = min(classes //2, base_dec_planes)

        self.bu_dec_l1 = EfficientPyrPool(in_planes=config[3], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[0])
        self.bu_dec_l2 = EfficientPyrPool(in_planes=dec_planes[0], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[1])
        self.bu_dec_l3 = EfficientPyrPool(in_planes=dec_planes[1], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[2])
        self.bu_dec_l4 = EfficientPyrPool(in_planes=dec_planes[2], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[3], last_layer_br=False)

        self.merge_enc_dec_l2 = EfficientPWConv(config[2], dec_planes[0])
        self.merge_enc_dec_l3 = EfficientPWConv(config[1], dec_planes[1])
        self.merge_enc_dec_l4 = EfficientPWConv(config[0], dec_planes[2])

        self.bu_br_l2 = nn.Sequential(nn.BatchNorm2d(dec_planes[0]),
                                      nn.PReLU(dec_planes[0])
                                      )
        self.bu_br_l3 = nn.Sequential(nn.BatchNorm2d(dec_planes[1]),
                                      nn.PReLU(dec_planes[1])
                                      )
        self.bu_br_l4 = nn.Sequential(nn.BatchNorm2d(dec_planes[2]),
                                      nn.PReLU(dec_planes[2])
                                      )

        #self.upsample =  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.init_params()

    def upsample(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def get_basenet_params(self):
        modules_base = [self.base_net]
        for i in range(len(modules_base)):
            for m in modules_base[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.PReLU):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_segment_params(self):
        modules_seg = [self.bu_dec_l1, self.bu_dec_l2, self.bu_dec_l3, self.bu_dec_l4,
                       self.merge_enc_dec_l4, self.merge_enc_dec_l3, self.merge_enc_dec_l2,
                       self.bu_br_l4, self.bu_br_l3, self.bu_br_l2]
        for i in range(len(modules_seg)):
            for m in modules_seg[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.PReLU):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def forward(self, x):
        '''
        :param x: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''
        x_size = (x.size(2), x.size(3))
        enc_out_l1 = self.base_net.level1(x)  # 112
        if not self.base_net.input_reinforcement:
            del x
            x = None

        enc_out_l2 = self.base_net.level2_0(enc_out_l1, x)  # 56

        enc_out_l3_0 = self.base_net.level3_0(enc_out_l2, x)  # down-sample
        for i, layer in enumerate(self.base_net.level3):
            if i == 0:
                enc_out_l3 = layer(enc_out_l3_0)
            else:
                enc_out_l3 = layer(enc_out_l3)

        enc_out_l4_0 = self.base_net.level4_0(enc_out_l3, x)  # down-sample
        for i, layer in enumerate(self.base_net.level4):
            if i == 0:
                enc_out_l4 = layer(enc_out_l4_0)
            else:
                enc_out_l4 = layer(enc_out_l4)

        # bottom-up decoding
        bu_out = self.bu_dec_l1(enc_out_l4)

        # Decoding block
        bu_out = self.upsample(bu_out)
        enc_out_l3_proj = self.merge_enc_dec_l2(enc_out_l3)
        bu_out = enc_out_l3_proj + bu_out
        bu_out = self.bu_br_l2(bu_out)
        bu_out = self.bu_dec_l2(bu_out)

        #decoding block
        bu_out = self.upsample(bu_out)
        enc_out_l2_proj = self.merge_enc_dec_l3(enc_out_l2)
        bu_out = enc_out_l2_proj + bu_out
        bu_out = self.bu_br_l3(bu_out)
        bu_out = self.bu_dec_l3(bu_out)

        # decoding block
        bu_out = self.upsample(bu_out)
        enc_out_l1_proj = self.merge_enc_dec_l4(enc_out_l1)
        bu_out = enc_out_l1_proj + bu_out
        bu_out = self.bu_br_l4(bu_out)
        bu_out  = self.bu_dec_l4(bu_out)

        return F.interpolate(bu_out, size=x_size, mode='bilinear', align_corners=True)


def espnetv2_seg(args):
    classes = args.classes
    scale=args.s
    weights = args.weights
    dataset=args.dataset
    model = ESPNetv2Segmentation(args, classes=classes, dataset=dataset)
    if weights:
        import os
        if os.path.isfile(weights):
            num_gpus = torch.cuda.device_count()
            device = 'cuda' if num_gpus >= 1 else 'cpu'
            pretrained_dict = torch.load(weights, map_location=torch.device(device))
        else:
            print_error_message('Weight file does not exist at {}. Please check. Exiting!!'.format(weights))
            exit()
        print_info_message('Loading pretrained basenet model weights')
        basenet_dict = model.base_net.state_dict()
        model_dict = model.state_dict()
        overlap_dict = {k: v for k, v in pretrained_dict.items() if k in basenet_dict}
        if len(overlap_dict) == 0:
            print_error_message('No overlaping weights between model file and pretrained weight file. Please check')
            exit()
        print_info_message('{:.2f} % of weights copied from basenet to segnet'.format(len(overlap_dict) * 1.0/len(model_dict) * 100))
        basenet_dict.update(overlap_dict)
        model.base_net.load_state_dict(basenet_dict)
        print_info_message('Pretrained basenet model loaded!!')
    else:
        print_warning_message('Training from scratch!!')
    return model

if __name__ == "__main__":

    from utilities.utils import compute_flops, model_parameters
    import torch
    import argparse

    parser = argparse.ArgumentParser(description='Testing')
    args = parser.parse_args()

    args.classes = 21
    args.s = 2.0
    args.weights='../classification/model_zoo/espnet/espnetv2_s_2.0_imagenet_224x224.pth'
    args.dataset='pascal'

    input = torch.Tensor(1, 3, 384, 384)
    model = espnetv2_seg(args)
    from utilities.utils import compute_flops, model_parameters
    print_info_message(compute_flops(model, input=input))
    print_info_message(model_parameters(model))
    out = model(input)
    print_info_message(out.size())