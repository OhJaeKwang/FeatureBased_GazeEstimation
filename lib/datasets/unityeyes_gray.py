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



from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel



class unityeyes_gray(data.Dataset):
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
        self.scale_factor = cfg.DATASET.SCALE_FACTOR  # --> 0.25
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP
        


        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file) # 자동으로 첫번째 행을 column으로 ?


        self.mean = np.array([0.3992,0.3992,0.3992], dtype=np.float32)
        self.std = np.array([0.1981,0.1981,0.1981], dtype=np.float32)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        # image_path = os.path.join(self.data_root,
        #                           self.landmarks_frame.iloc[idx, 0])
        image_path = self.data_root + self.landmarks_frame.iloc[idx,0] # 이미지 이름

        scale = self.landmarks_frame.iloc[idx, 1]   # 이미지 scale (0.5 평균값과 그 근방값들)

        center_w = self.landmarks_frame.iloc[idx, 2] # 필요 없는 정보
        center_h = self.landmarks_frame.iloc[idx, 3] # 필요 없는 정보
        center = torch.Tensor([center_w, center_h])  # 이미지의 중앙 정보를 의미하는거 같음

        pts = self.landmarks_frame.iloc[idx, 4:-5].values  # 여기서가 50개의 키포인트 값들
        
        # if(pts.shape)
        pts = pts.astype('float').reshape(-1, 2) # (50,2) 형태로 바꿔주기
        
        gaze = self.landmarks_frame.iloc[idx,-5:-2].values.astype('float').reshape(-1,3)
        pitch_yaw = self.landmarks_frame.iloc[idx,-2:].values.astype('float').reshape(-1,2)
        scale *= 1.25 # 0.5*1.25 정도 --> 0.625 정도?
    
        nparts = pts.shape[0] # 50
        # img = np.array(cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2GRAY), dtype=np.float32)
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32) # row, cols ,channel
        r = 0
        # temp = img.copy()
        
        # if self.is_train: # ?
        #     scale = scale * (random.uniform(1 - self.scale_factor,  # 0.75
        #                                     1 + self.scale_factor)) # 1.25
        #     # r = random.uniform(-self.rot_factor, self.rot_factor) \
        #     #     if random.random() <= 0.6 else 0
        #     if random.random() <= 0.5 and self.flip:
        #         img = np.fliplr(img)
        #         pts = fliplr_joints(pts, width=img.shape[1], dataset='unityeyes')
        #         center[0] = img.shape[1] - center[0]
        # img = crop(img, center, scale, self.input_size, rot=0)

        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy().astype(int)

        for i in range(nparts):
            if tpts[i, 1] >= 0: #( y값이 0보다 크면?)
                # tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                #                                scale, self.output_size, rot=r)
                target[i] = generate_target(target[i], tpts[i], self.sigma,
                                            label_type=self.label_type)

        print(target[0].shape)
        plt.imshow(target[0])
        plt.show()

        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1]) # row, col ,chanenl --> channel,row,col
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)
        gaze = torch.Tensor(gaze)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts ,'gaze' : gaze, 'pitch_yaw': pitch_yaw}

        return img, target, meta ,gaze , pitch_yaw


# if __name__ == '__main__':
#     
