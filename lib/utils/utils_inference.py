import math
import scipy
import scipy.misc
import torch
import numpy as np
from PIL import Image
from lib.models import get_eye_alignment_net#, get_cls_net
from lib.config import config, config_imagenet, merge_configs
import cv2
import pandas as pd
import time
import pytorch_model_summary

def get_model_by_name(checkpoint_path, config_path, model_type='landmarks', device='cuda'):

    merge_configs(config, config_path)

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    if model_type == 'landmarks':
        model = get_eye_alignment_net(config)
    else:
        model = get_cls_net(config_imagenet)

    state_dict = torch.load(checkpoint_path)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model



# input 이 지금 img (raw,col,channel)
def get_lmks_by_img(model, img, output_size=(192, 192), rot=0, conf_score=False):
#   img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

    face_center = torch.Tensor([img.shape[1]//2, img.shape[0]/2])      # (96,96) 
    crop_scale = max((img.shape[1]) / output_size[0], (img.shape[0]) / output_size[1])  # 아웃풋과 이미지와의 스케일 비율
    # hegith 와 width 중 더 큰 스케일 차이  --> 192/192 = 4
    
    img_crop = crop(img, face_center, crop_scale, output_size=output_size, rot=rot)   # face_center : (96,96)
    # 원래 이미지 형태는 행, 열 , 채널  --> 세로 , 가로 , 채널 (0 ,1 ,2)
    img_crop = (img_crop/255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225]) # normalize ? 
    img_crop = img_crop.transpose([2, 0, 1]) # Tensor의 형태로 맞춰 주기 위함  (세로 가로 채널 ) --> (채널 , 세로 , 가로)
    img_crop = torch.tensor(img_crop, dtype=torch.float32).unsqueeze(0).cuda() # batch size의 크기를 넣어줌
    with torch.no_grad():
        pred  = model(img_crop)
    return decode_preds(pred, [face_center], [crop_scale], [output_size[0]/4,output_size[1]/4], conf_score)  
    #  decode_preds( 히트맵 , [[96,96]], [1], res=[48,48] , conf_score = Flase ) 히트맵의 위치 정보만 필요로 함.

def get_lmks_by_img_ykh(model, img, output_size=(192, 192), rot=0, conf_score=False):
#   img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

    face_center = torch.Tensor([img.shape[1]//2, img.shape[0]/2])      # (96,96) 
    crop_scale = max((img.shape[1]) / output_size[0], (img.shape[0]) / output_size[1])  # 아웃풋과 이미지와의 스케일 비율
    # hegith 와 width 중 더 큰 스케일 차이  --> 192/192 = 4
    
    img_crop = crop(img, face_center, crop_scale, output_size=output_size, rot=rot)   # face_center : (96,96)
    # 원래 이미지 형태는 행, 열 , 채널  --> 세로 , 가로 , 채널 (0 ,1 ,2)
    img_crop = (img_crop/255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225]) # normalize ? 
    img_crop = img_crop.transpose([2, 0, 1]) # Tensor의 형태로 맞춰 주기 위함  (세로 가로 채널 ) --> (채널 , 세로 , 가로)
    img_crop = torch.tensor(img_crop, dtype=torch.float32).unsqueeze(0).cuda() # batch size의 크기를 넣어줌
    with torch.no_grad():
        # start_time = time.time()
        pred  = model(img_crop)
        # print(pytorch_model_summary.summary(model, torch.zeros(1,3,96,160).cuda(),show_input=True))
        # print("WorkingTime: {} msec".format((time.time()-start_time)*1000))
    return decode_preds2(pred, [face_center], [crop_scale], [output_size[0]/4,output_size[1]/4], conf_score)  
    #  decode_preds( 히트맵 , [[96,96]], [1], res=[48,48] , conf_score = Flase ) 히트맵의 위치 정보만 필요로 함.


def get_lmks_by_img_v2(model, img, output_size=(96, 160), rot=0, conf_score=False):
#   img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

    face_center = torch.Tensor([img.shape[1]//2, img.shape[0]/2])      # (96,96) 
    crop_scale = max((img.shape[0]) / output_size[0], (img.shape[1]) / output_size[1])  # 아웃풋과 이미지와의 스케일 비율
    # hegith 와 width 중 더 큰 스케일 차이  --> 192/192 = 4
    
    img_crop = crop(img, face_center, crop_scale, output_size=output_size, rot=rot)   # face_center : (96,96)
    # 원래 이미지 형태는 행, 열 , 채널  --> 세로 , 가로 , 채널 (0 ,1 ,2)
    # img_crop = (img_crop/255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225]) # normalize ? 
    img_crop = (img_crop/255.0 - np.array([0.3992,0.3992,0.3992])) / np.array([0.1981,0.1981,0.1981])
    img_crop = img_crop.transpose([2, 0, 1]) # Tensor의 형태로 맞춰 주기 위함  (세로 가로 채널 ) --> (채널 , 세로 , 가로)
    img_crop = torch.tensor(img_crop, dtype=torch.float32).unsqueeze(0).cuda() # batch size의 크기를 넣어줌
    with torch.no_grad():
        pred  = model(img_crop)
    return decode_preds(pred, [face_center], [crop_scale], [output_size[0]/4,output_size[1]/4], conf_score) 
    #  decode_preds( 히트맵 , [[96,96]], [1], res=[48,48] , conf_score = Flase ) 히트맵의 위치 정보만 필요로 함.

def get_lmks_by_img_v3(model, img, output_size=(96, 160), rot=0, conf_score=False):             # onestage_cbam_ori
#   img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

    # img_crop = (img_crop/255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225]) # normalize ? 
    img_norm = (img/255.0 - np.array([0.3992,0.3992,0.3992])) / np.array([0.1981,0.1981,0.1981])
    img_crop = img_norm.transpose([2, 0, 1]) # Tensor의 형태로 맞춰 주기 위함  (세로 가로 채널 ) --> (채널 , 세로 , 가로)
    img_crop = torch.tensor(img_crop, dtype=torch.float32).unsqueeze(0).cuda() # batch size의 크기를 넣어줌
    with torch.no_grad():
        pred   = model(img_crop)
    
    return decode_preds(pred, 0, 0, [output_size[0],output_size[1]], conf_score)   
    #  decode_preds( 히트맵 , [[96,96]], [1], res=[48,48] , conf_score = Flase ) 히트맵의 위치 정보만 필요로 함.

def get_lmks_by_img_v4(model, img, output_size=(96, 160), rot=0, conf_score=False):              # onestage_cbam
#   img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

    # img_crop = (img_crop/255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225]) # normalize ? 
    img_norm = (img/255.0 - np.array([0.3992,0.3992,0.3992])) / np.array([0.1981,0.1981,0.1981])
    img_crop = img_norm.transpose([2, 0, 1]) # Tensor의 형태로 맞춰 주기 위함  (세로 가로 채널 ) --> (채널 , 세로 , 가로)
    img_crop = torch.tensor(img_crop, dtype=torch.float32).unsqueeze(0).cuda() # batch size의 크기를 넣어줌
    with torch.no_grad():
        
        pred  = model(img_crop)[0]
        # print(pytorch_model_summary.summary(model, torch.zeros(1,3,96,160).cuda(),show_input=True))
    return decode_preds(pred, 0, 0, [output_size[0],output_size[1]], conf_score)  
    #  decode_preds( 히트맵 , [[96,96]], [1], res=[48,48] , conf_score = Flase ) 히트맵의 위치 정보만 필요로 함.

def get_lmks_by_img_htmap(model, img, output_size=(96, 160), rot=0, conf_score=False):              # onestage_cbam
#   img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

    # img_crop = (img_crop/255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225]) # normalize ? 
    img_norm = (img/255.0 - np.array([0.3992,0.3992,0.3992])) / np.array([0.1981,0.1981,0.1981])
    img_crop = img_norm.transpose([2, 0, 1]) # Tensor의 형태로 맞춰 주기 위함  (세로 가로 채널 ) --> (채널 , 세로 , 가로)
    img_crop = torch.tensor(img_crop, dtype=torch.float32).unsqueeze(0).cuda() # batch size의 크기를 넣어줌
    with torch.no_grad():
        pred    = model(img_crop)
    return decode_preds(pred, 0, 0, [output_size[0],output_size[1]], conf_score) , pred

def get_lmks_by_img_mpii(model, img, output_size=(96, 160), rot=0, conf_score=False):              # onestage_cbam
#   img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

    # img_crop = (img_crop/255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225]) # normalize ? 
    img_norm = (img/255.0 - np.array([0.506, 0.506, 0.506])) / np.array([0.288, 0.288, 0.288])
    img_crop = img_norm.transpose([2, 0, 1]) # Tensor의 형태로 맞춰 주기 위함  (세로 가로 채널 ) --> (채널 , 세로 , 가로)
    img_crop = torch.tensor(img_crop, dtype=torch.float32).unsqueeze(0).cuda() # batch size의 크기를 넣어줌
    with torch.no_grad():
        pred  = model(img_crop)
    return decode_preds(pred, 0, 0, [output_size[0],output_size[1]], conf_score)

def get_lmks_by_img_mhtmap(model, img, output_size=(96, 160), rot=0, conf_score=False):              # onestage_cbam
#   img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

    # img_crop = (img_crop/255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225]) # normalize ? 
    img_norm = (img/255.0 - np.array([0.506, 0.506, 0.506])) / np.array([0.288, 0.288, 0.288])
    img_crop = img_norm.transpose([2, 0, 1]) # Tensor의 형태로 맞춰 주기 위함  (세로 가로 채널 ) --> (채널 , 세로 , 가로)
    img_crop = torch.tensor(img_crop, dtype=torch.float32).unsqueeze(0).cuda() # batch size의 크기를 넣어줌
    with torch.no_grad():
        pred    = model(img_crop)
    return decode_preds(pred, 0, 0, [output_size[0],output_size[1]], conf_score) , pred


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


def decode_preds(output, center, scale, res, conf_score=False): # output size에 맞게 결과 값을 얻음
    if conf_score == True: 
        coords, scores = get_preds(output, conf_score)  # float type  [1,50,24,48] heat map에서 각 채널 당 가장 큰 값과 그 위치(열 행)를 반 환
    else:
        coords = get_preds(output)  # float type
    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)): # 배치 사이즈 만큼
        for p in range(coords.size(1)): # 각 채널 당
            hm = output[n][p]        # htmap의 모든 정보임
            px = int(math.floor(coords[n][p][0])) # 소수점 버려 버림 (열)
            py = int(math.floor(coords[n][p][1])) # 소수점 버려 버림 (행) 
            if (py > 1) and (py < res[0]) and (px > 1) and (px < res[1]): # --> 이 부분이 이해가 안감
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5 # 왜 더하지????
    preds = coords.clone()
    # Transform back
    yk = False
    if yk:
        for i in range(coords.size(0)): # 열 행 순 --> (갯수, 열위치, 행위치)
            preds[i] = transform_preds(coords[i], center[i], scale[i], res) # ((열위치,행위치),center좌표, scale 값, [24,48])

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    if conf_score == True:
        return preds.cpu().numpy().squeeze(0), scores.cpu().numpy().squeeze(0)
    else:
        return preds.cpu().numpy().squeeze(0)

def decode_preds2(output, center, scale, res, conf_score=False): # output size에 맞게 결과 값을 얻음
    if conf_score == True: 
        coords, scores = get_preds(output, conf_score)  # float type  [1,50,24,48] heat map에서 각 채널 당 가장 큰 값과 그 위치(열 행)를 반 환
    else:
        coords = get_preds(output)  # float type
    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)): # 배치 사이즈 만큼
        for p in range(coords.size(1)): # 각 채널 당
            hm = output[n][p]        # htmap의 모든 정보임
            px = int(math.floor(coords[n][p][0])) # 소수점 버려 버림 (열)
            py = int(math.floor(coords[n][p][1])) # 소수점 버려 버림 (행) 
            if (py > 1) and (py < res[0]) and (px > 1) and (px < res[1]): # --> 이 부분이 이해가 안감
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5 # 왜 더하지????
    preds = coords.clone()
    # Transform back
    yk = True
    if yk:
        for i in range(coords.size(0)): # 열 행 순 --> (갯수, 열위치, 행위치)
            preds[i] = transform_preds(coords[i], center[i], scale[i], res) # ((열위치,행위치),center좌표, scale 값, [24,48])

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    if conf_score == True:
        return preds.cpu().numpy().squeeze(0), scores.cpu().numpy().squeeze(0)
    else:
        return preds.cpu().numpy().squeeze(0)



def crop(img, center, scale, output_size=(192,192), rot=0):
    # center : [center_w, center_h]
    center_new = center.clone()   

    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1] # height , widht
    sf = scale * 200.0 / output_size[0] # sf는 scale = 4 일 것이다. 200 / 192 --> 약 1
    if sf < 2:
        sf = 1
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf))  # 46?
        new_ht = int(np.math.floor(ht / sf)) 
        new_wd = int(np.math.floor(wd / sf))
        if new_size < 2:
            return torch.zeros(output_size[0], output_size[1], img.shape[2]) \
                        if len(img.shape) > 2 else torch.zeros(output_size[0], output_size[1])
        else:
            img = np.array(Image.fromarray(img.astype(np.uint8)).resize((new_wd, new_ht)))
#             img = scipy.misc.imresize(img, [new_ht, new_wd])  # (0-1)-->(0-255)
            center_new[0] = center_new[0] * 1.0 / sf
            center_new[1] = center_new[1] * 1.0 / sf
            scale = scale / sf

    # Upper left point
    ul = np.array(transform_pixel([0, 0], center_new, scale, output_size, invert=1))
    # Bottom right point
    br = np.array(transform_pixel(output_size, center_new, scale, output_size, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]

    new_img = np.zeros(new_shape, dtype=np.float32)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]
    new_img = np.array(Image.fromarray(new_img.astype(np.uint8)).resize(output_size[::-1]))
#     new_img = scipy.misc.imresize(new_img, output_size)
    return new_img


def transform_pixel(pt, center, scale, output_size, invert=0, rot=0):  
    # Transform pixel location to different reference
    t = get_transform(center, scale, output_size, rot=rot) # 픽셀의 위취를 다른 reference로 변환
    if invert: # 여기에 인벌스 메트릭스 적용할 수 있으면 좋을듯?
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def get_transform(center, scale, output_size, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale # 스케일을 왜 2배 늘리는거?  h = 800 ( scale = 4)
    t = np.zeros((3, 3))  # 3x3 행렬 만듬
    t[0, 0] = float(output_size[1]) / h                              #   
    t[1, 1] = float(output_size[0]) / h                              #  | 160/200    0       160 * (96/200 + 0.5) |
    t[0, 2] = output_size[1] * (-float(center[0]) / h + .5)          #  |   0     96/200     96  * (96/200 + 0.5) |
    t[1, 2] = output_size[0] * (-float(center[1]) / h + .5)          #  |   0        0              1             |
    t[2, 2] = 1                                                          
    if not rot == 0: # rotation 진행하는 거 같음
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -output_size[1]/2
        t_mat[1, 2] = -output_size[0]/2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform_preds(coords, center, scale, output_size):       # ((열위치,행위치),center좌표, scale 값, [24,48])

    for p in range(coords.size(0)): # 
        coords[p, 0:2] = torch.tensor(transform_pixel(coords[p, 0:2], center, scale, output_size, 1, 0))
    return coords

if __name__ == "__main__":
    # img = cv2.imread("C:/Users/OJK/Task/Gaze_Estimation/input/UnityEyes_Data/160x96/Train_data/img/1.jpg")
    # label = pd.read_csv("C:/Users/OJK/Task/Gaze_Estimation/input/UnityEyes_Data/160x96/UE_train_labels.csv")
    img = cv2.imread("C:/Users/OJK/Task/Gaze_Estimation/input/UnityEyes_Data/640x480/Train_data/crop_gray_img/1.jpg")
    label =pd.read_csv("C:/Users/OJK/Task/Gaze_Estimation/input/UnityEyes_Data/640x480/Train_data/UE_train_labels.csv")
    center_w = label.iloc[0, 2] # 필요 없는 정보
    center_h = label.iloc[0, 3] # 필요 없는 정보
    center = torch.Tensor([center_w, center_h])
    scale = label.iloc[0, 1]
    img2 = crop(img,center,1,(96,160))
    print(img2.shape) 
    cv2.imshow("plz",img2)
    cv2.waitKey(0)
