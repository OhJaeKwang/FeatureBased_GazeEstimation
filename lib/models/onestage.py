# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Create by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import tanh

import os
import logging
import sys
import easydict

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# from config import config, update_config

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_model_summary
import torchsummary
from easydict import EasyDict as edict

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


def get_preds(scores, conf_score=False):  # 결과 값이 heatmap에서 각 키포인트를 의미하는 각 채널에서의 가장 큰 값과 그 위치를 알아 보는 함수 
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'   #  output의 차원이 4차원이여야함 ( N x C x H x W ) --> (배치사이즈 , 채널, 행수(세로길이) , 열수(가로길이))
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)  # 이미지를 flatten 하게 피고 나서, 각 채널에서 가장 큰 값을 뽑아내는 듯 --> 즉 키포인트의 위치..
    # maxval shape --> [1,50] idx shape --> [1,50]
    maxval = maxval.view(scores.size(0), scores.size(1), 1) # maxval shape --> [1,50,1]
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1 # idx shape --> [1,50,1] , 그리고 각 값에 +1 해줌

    preds = idx.repeat(1, 1, 2).float() # preds shape --> [1,50,2]

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1 # 행에서 몇칸 가야 하는지
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1 # 몇번째 행인지 확인
    
    # preds는 output에서 각 채널에 대해 가장 큰 값을 가지는 인덱스를 (열 , 행) 으로 가지고 있음

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float() # pred_mask는 maxval이 0보다 작지 않다면 다 1로 변환해서 (1,50,2) 의 형태로 가지고 있음
    preds *= pred_mask # 각 maxval를 가지는 위치에 마스크 값을 곱해줌 (여기서 0보다 작은 값들을 골라 내주는 거 같음)

    if conf_score == True:
        return preds, maxval  # 결과적으로 max 값을 가지는 위치값과 , max값을 결과값을 줌  preds의 shape는 [1,50,2] --> 히트맵에서 어떤 위치에 같을 가지고 있는지
    else:                                                                             #   max의 shape는   [1,50,1] --> 각 히트맵에서의 최대 값 --> 키 포인트 값을 의미
        return preds



def conv3x3(in_planes, out_planes, stride=1):   # 3x3 covolution 사이즈는 그대로 하는 convolution
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):   # residual block
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)                      
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):   # bottleneck (expansion 4번) , convolution 3번 
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):              
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SABottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, # 이미지 사이즈 유지
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x        # 입력

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out        


class HighResolutionModule(nn.Module) : 
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(                                                            
            num_branches, blocks, num_blocks, num_inchannels, num_channels)         

        self.num_inchannels = num_inchannels                                        
        self.fuse_method = fuse_method                  
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(                    
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)
                               
    def _check_branches(self, num_branches, blocks, num_blocks,            # branch 갯수 , block , blocks 수 , 입력 채널 수 , 채널의 수  --> branch를 확인 하는 메서드 ? 브랜치가 뭔지 알아야 할듯?
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):                                # 블록 수의 list 길이와 branch의 갯수가 다르면 error
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):                              # 블록 수의   
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion: # 다운 샘플링 필요할 때  
            downsample = nn.Sequential(                                                     
                nn.Conv2d(self.num_inchannels[branch_index],                                       # 입력 채널 , 결과 채널 수 
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []                                                                                
        layers.append(block(self.num_inchannels[branch_index],                                     # layer --> 블록 추가 하는 레이어
                            num_channels[branch_index], stride, downsample))                        
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):                                               #  
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                    # nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[x[i].shape[2], x[i].shape[3]],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}




class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        self.inplanes = 64
        extra = config.MODEL.EXTRA
        super(HighResolutionNet, self).__init__()

        # stem net
        self.residual = nn.Conv2d(3,config.MODEL.NUM_JOINTS,kernel_size=3,stride=1,padding=1,bias=False)
        self.cbam = SABottleneck(50,50)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,   # size 2배 감소
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,  # size 2배 감소
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.sf = nn.Softmax(dim=1)                                        
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4) # (block종류,입력,출력채널,갯수)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        final_inp_channels = sum(pre_stage_channels)                    

        self.head = nn.Sequential(                                                
            nn.ConvTranspose2d(
                in_channels=final_inp_channels,
                out_channels=final_inp_channels,
                kernel_size=4,
                stride=2,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0),
            BatchNorm2d(final_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=final_inp_channels,
                out_channels=final_inp_channels,                # 조인트 갯수로 바꿈              
                kernel_size=4,
                stride=2,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0),
            BatchNorm2d(final_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=final_inp_channels,
                out_channels=config.MODEL.NUM_JOINTS,                # 조인트 갯수로 바꿈              
                kernel_size=3,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
            )
        )

        self.gaze_layer = nn.Sequential(
            nn.Linear(in_features=100,out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128,out_features=2),
            # nn.Tanh() # -1 ~ 1 사이 값으로 뽑아 내고 싶음
        )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        # h, w = x.size(2), x.size(3)
        
        residual = x
        residual = self.residual(residual)
        residual = self.cbam(residual)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)            # stem
        x = self.layer1(x)             

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Head Part
        height, width = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(height, width), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x[2], size=(height, width), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x[3], size=(height, width), mode='bilinear', align_corners=False)
        x = torch.cat([x[0], x1, x2, x3], 1)
        
        
        joints = self.head(x) # joint가 지금 heatmap 이라고 보면 됨 (여기서 값이 가장 큰 놈을 쓰겟다.)

        joints = joints + residual # (batch, 50, 96, 160)
        

        # self attention을 통해서 좀 더 가중치 있는 값을 확인하고 싶은데?
        

        # 이쪽을 많이 고민해봐야됨 굳이 one-stage 가는 이유가 없어질 수도 

        key_point_xy = get_preds(joints) # (batch, 50 ,2)
        
        # 들어가기 전에 눈 중심으로 노멀라이즈
        gaze_layer_input = key_point_xy.clone()

        ws = []
        w  = []
        for i in range(key_point_xy.size(0)):
            for idx in gaze_layer_input[i]:
                ws.append(idx[0])
            w.append(max(ws)-min(ws))
            ws = []

        for i in range(key_point_xy.size(0)):
            for idx in gaze_layer_input[i]:
                idx[0] -= key_point_xy[i][0][0]
                idx[0] /= w[i]
                idx[1] -= key_point_xy[i][0][1]
                idx[1] /= w[i]

        #gaze_layer_input = gaze_layer_input.view(gaze_layer_input.shape[0],100)
        gaze_layer_input = gaze_layer_input.view(-1,100)
        gaze_layer_input = gaze_layer_input*0.5 + 0.5

        x = self.gaze_layer(gaze_layer_input) # (batch,3)의 size
                
        return joints 

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # if os.path.isfile(pretrained):
        #     pretrained_dict = torch.load(pretrained)
        #     logger.info('=> loading pretrained model {}'.format(pretrained))
        #     model_dict = self.state_dict()
        #     pretrained_dict = {k: v for k, v in pretrained_dict.items()
        #                        if k in model_dict.keys()}
        #     for k, _ in pretrained_dict.items():
        #         logger.info(
        #             '=> loading {} pretrained model {}'.format(k, pretrained))
        #     model_dict.update(pretrained_dict)
        #     self.load_state_dict(model_dict)

def get_eye_alignment_net(config, **kwargs):

    model = HighResolutionNet(config, **kwargs)
    pretrained = config.MODEL.PRETRAINED if config.MODEL.INIT_WEIGHTS else ''
    model.init_weights(pretrained=pretrained)

    return model

if __name__=='__main__':
    with open("C:/Users/OJK/Task/Gaze_Estimation/eye_landmarks_detection/experiments/unityeyes/eye_alignment_unityeyes_hrnet_w18.yaml") as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    print(config)
    config = edict(config) 
    print("**********************************")
    model = get_eye_alignment_net(config).eval()
    x = torch.zeros(1,3,96,160)
    htm , output = model(x)
    print(pytorch_model_summary.summary(model, torch.zeros(1,3,96,160),show_input=True))
