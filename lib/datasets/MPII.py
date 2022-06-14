# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

#from _typeshed import Self
import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np
import json
import cv2
from matplotlib import pyplot as plt


from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel , transform_preds , decode_preds


class mpii(data.Dataset):
    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = cfg.DATASET.TRAINSET
        else:
            self.csv_file = cfg.DATASET.TESTSET

        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR  # --> 히트맵과 아웃풋과의 스케일 차이
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP
        
        # load annotations
        
        self.landmarks_frame = pd.read_csv(self.csv_file)

        self.mean = np.array([0.506, 0.506, 0.506], dtype=np.float32)  # normalization 값
        self.std = np.array([0.288, 0.288, 0.288], dtype=np.float32)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        image_path = self.data_root + self.landmarks_frame.iloc[idx,0] # 이미지 이름

        pts = self.landmarks_frame.iloc[idx, 1:].values  # 여기서가 50개의 키포인트 값들
        
        pts = pts.astype('float').reshape(-1, 2) # (50,2) 형태로 바꿔주기
        
        nparts = pts.shape[0] # 50

        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
        img = cv2.resize(img,dsize=(160,96),interpolation=cv2.INTER_AREA)

        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy() # 50 개의 랜드마크 좌표
        fpts = pts.copy() # 히트맵에서의 랜드마크 좌표 

        for i in range(nparts):
                target[i] = generate_target(target[i], tpts[i], self.sigma,
                                            label_type=self.label_type)
        
        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)

        meta = {'index': idx, 'pts': torch.Tensor(pts), 'tpts': tpts } # ,'gaze' : gaze}

        return img, target, meta # ,gaze
