# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import math

import torch
import numpy as np

from ..utils.transforms import transform_preds


def get_preds(scores): # 히트맵에서의 좌표
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1   # 몇번 쨰 열인지
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1 # 몇번 째 행인지 

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float() 
    preds *= pred_mask            
    return preds


def compute_nme(preds, meta):

    targets = meta['pts']
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0] # batch
    L = preds.shape[1] # 50  
    rmse = np.zeros(N) # 

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        elif L == 50:
            interocular = np.linalg.norm(pts_gt[0, ] - pts_gt[16, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)

    return rmse

def compute_angle_error(preds,meta):
    target = meta["gaze"]
    preds = preds.cpu().detach().numpy()
    target = target.cpu().numpy()

    N = preds.shape[0]
    angle_errors =[]

    for i in range(N):
        
        px = -1 * np.cos(preds[i][0]) * np.sin(preds[i][1])
        py = np.sin(preds[i][0])
        pz = -1 * np.cos(preds[i][0]) * np.cos(preds[i][1])
        
        pr_gaze = np.array([px,py,pz])
        gt = (target[i]/np.linalg.norm(target[i])).squeeze(0)
        
        angle_errors.append(np.rad2deg(np.arccos(np.dot(pr_gaze,gt))))


    return angle_errors





def decode_preds(output, center, scale, res): # res (행,열) --> (height,widht)
    coords = get_preds(output)  # float type (batchsize, 50 , 2)

    coords = coords.cpu()

    # pose-processing
    for n in range(coords.size(0)): # batch size 마큼
        for p in range(coords.size(1)): # 각 채널 당
            hm = output[n][p]    # htmap에서의 위 치값 hm --> shape(40,24) 
            # print(hm.shape) 
            px = int(math.floor(coords[n][p][0])) # 열
            py = int(math.floor(coords[n][p][1])) # 행
            if (py > 1) and (py < res[0]) and (px > 1) and (px < res[1]):
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()
    
    # Transform back
    # for i in range(coords.size(0)):
    #     preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds
