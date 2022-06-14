import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F 
import os
import sys
import cv2
import math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.utils import utils_inference
from lib.utils import utils_landmarks
from lib.ransac import ransac
from lib.gaze_estimation import gaze_estimation
import time
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Settings
method = '2st_CBAM_50'   
ckpt = 30   #13                  # check point epoch
model = utils_inference.get_model_by_name('C:/Users/OJK/Task/Gaze_Estimation/eye_landmarks_detection/tools/output/unityeyes/eye_alignment_unityeyes_hrnet_w18/backup/' + method + '/checkpoint_{}.pth'.format(ckpt),
                                          'C:/Users/OJK/Task/Gaze_Estimation/eye_landmarks_detection/experiments/unityeyes/eye_alignment_unityeyes_hrnet_w18.yaml',
                                          device='cuda')


##### Predictions #####

Label = False

if Label:
    print("visualize ground truth")


MPII = False

Unity_vali_path = "D:/Unityeyes_dataset/640x480/Train_data/"
paper_fig = "D:/Unityeyes_dataset/1280x720"


MPII_Root_path = "C:/Users/OJK/Task/Gaze_Estimation/input/MPIIGaze/Train_data_set/images/whole"
file_index = 3           # 33 #730 #734
Image_path =  MPII_Root_path + "/{}.jpg".format(file_index) if MPII else paper_fig + "/{}.jpg".format(file_index)

frame = cv2.imread(Image_path)
frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)

img_size = [192, 192]
half_size = [int(img_size[0] / 2), int(img_size[1] / 2)] # [96,96]
frame_center = [int(frame.shape[0] / 2), int(frame.shape[1] / 2)] # [240,320]
frame = frame[frame_center[0] - half_size[0]:frame_center[0] + half_size[0], frame_center[1] - half_size[1]:frame_center[1] + half_size[1]] 
frame = cv2.resize(frame, dsize=(160, 96), interpolation=cv2.INTER_AREA)
temp = frame.copy()
frame_inf = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
temp = frame_inf.copy()
frame_inf = cv2.cvtColor(frame_inf, cv2.COLOR_GRAY2RGB)

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


##### Eye Landmarks Detection #####
lmks , htmap = utils_inference.get_lmks_by_img_htmap(model, frame_inf)  
img = utils_landmarks.set_circles_on_img(frame.copy(), lmks, circle_size=1, is_copy=False) 
interior, iris = lmks[1:17], lmks[18:]


# heatmap show
htmap_vis = htmap[0].view(50,-1)
htmap_vis = F.softmax(htmap_vis,dim=1)
htmap_vis = htmap_vis.cpu().numpy() 
htmap_vis = htmap_vis.sum(axis=0)
htmap_vis = htmap_vis - htmap_vis.min(axis=0)
htmap_vis = htmap_vis / htmap_vis.max(axis=0)
# for idx in range(50):
# 	htmap_vis[idx] = htmap_vis[idx] - htmap_vis[idx].min(axis=0)
# 	htmap_vis[idx] = htmap_vis[idx] / htmap_vis[idx].max(axis=0)
htmap_vis = htmap_vis.reshape(96,160)
# htmap_vis = np.uint8(255 * htmap_vis[2]).reshape(96,160)
htmap_vis = np.uint8(255 * htmap_vis)
htmap_vis = htmap_vis.astype(np.uint8)
print(htmap_vis.max())
htmap_vis = np.array([htmap_vis,htmap_vis,htmap_vis]).transpose(1,2,0)
heatmap = cv2.applyColorMap(htmap_vis, cv2.COLORMAP_JET)
result = heatmap * 1


cv2.imshow('image', result/255.0)
plt.show()

cv2.waitKey(0)

# cv2.destroyAllWindows()

