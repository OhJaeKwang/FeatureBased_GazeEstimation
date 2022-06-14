import torch
from torch import nn
from torch.nn import Module
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
import pandas as pd
import time
import json


# def transform_gt(img_size,img_path,label):


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Settings
method = 'cbam'   
ckpt =  144 #13                  # check point epoch
model = utils_inference.get_model_by_name('C:/Users/OJK/Task/Gaze_Estimation/eye_landmarks_detection/tools/output/unityeyes/eye_alignment_unityeyes_hrnet_w18/backup/' + method + '/checkpoint_{}.pth'.format(ckpt),
                                          'C:/Users/OJK/Task/Gaze_Estimation/eye_landmarks_detection/experiments/unityeyes/eye_alignment_unityeyes_hrnet_w18.yaml',
                                          device='cuda')

gaze_estimator = gaze_estimation.GazeEstimator().to(device)   
gaze_estimator.load_state_dict(torch.load('C:/Users/OJK/Task/Gaze_Estimation/eye_landmarks_detection/lib/gaze_estimation/gaze_estimator_weights.pth'))
gaze_estimator.eval()


##### Predictions #####

MG_Label = False
UE_Label = True


Unity_vali_path = "D:/Unityeyes_dataset/640x480/Test_data/img"
MPII_Root_path = "C:/Users/OJK/Task/Gaze_Estimation/input/MPIIGaze/Train_data_set/images/whole"
file_index = 2244   # 1, 12 , 666

Image_path =  MPII_Root_path + "/{}.jpg".format(file_index) if MG_Label else Unity_vali_path + "/{}.jpg".format(file_index)

frame = cv2.imread(Image_path)



img_size = [192, 192]
half_size = [int(img_size[0] / 2), int(img_size[1] / 2)] # [96,96]
frame_center = [int(frame.shape[0] / 2), int(frame.shape[1] / 2)] # [240,320]
frame = frame[frame_center[0] - half_size[0]:frame_center[0] + half_size[0], frame_center[1] - half_size[1]:frame_center[1] + half_size[1]] 
# frame = cv2.resize(frame, dsize=(192, 192), interpolation=cv2.INTER_AREA)
frame_inf = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame = cv2.resize(frame , dsize=(160,96),interpolation=cv2.INTER_AREA)
frame_inf = cv2.cvtColor(frame_inf, cv2.COLOR_GRAY2RGB)
frame_inf = cv2.resize(frame_inf,dsize=(160,96))


##### Eye Landmarks Detection #####
lmks , pred_gaze = utils_inference.get_lmks_by_img_v3(model, frame_inf)    

end_t = time.time()

img = utils_landmarks.set_circles_on_img(frame.copy(), lmks, circle_size=1, is_copy=False) 

interior, iris = lmks[1:17], lmks[18:]

##### Gaze Estimation #####

input_data = lmks.copy()   

ws = []   
for idx in input_data:
    ws.append(idx[0])
w = max(ws) - min(ws)   

for idx in input_data:   
    idx[0] -= lmks[0][0]    
    idx[0] /= w
    idx[1] -= lmks[0][1]
    idx[1] /= w
input_data[0][0]=0  
input_data[0][1]=0


x_inputs = torch.from_numpy(input_data).to(device)
x_input = x_inputs.reshape(1, 100) 

x_input = x_input[0] 
y_pred = pred_gaze[0]  
y_pred= y_pred.cpu().detach().numpy()


# ptichyaw_to_gaze
p_pitch , p_yaw = y_pred
px = -1 * np.cos(p_pitch) * np.sin(p_yaw)
py = np.sin(p_pitch)
pz = -1 * np.cos(p_pitch) * np.cos(p_yaw)
y_pred = np.array([px,py,pz])


if(UE_Label):
    with open('D:/Unityeyes_dataset/640x480/Test_data/label/{}.json'.format(file_index)) as f:
        json_data = json.load(f)
    gaze_vector_raw = json_data['eye_details']['look_vec']
    gaze_vector = gaze_vector_raw.lstrip('(').rstrip(')').split(', ')
    gt_vector = [float(gaze_vector[0]), float(gaze_vector[1]), float(gaze_vector[2])]
    gt_vector /= np.linalg.norm(np.array(gt_vector))
    gt_py_value = np.array([math.asin(float(gt_vector[1])),math.atan2(-float(gt_vector[0]),-float(gt_vector[2]))])
    gt_degree = np.rad2deg(gt_py_value)
    print("[GT] gaze : ", gt_vector)
    print("[GT] pitch : {:.3f}°, yaw : {:.3f}°, py_value : {}".format(gt_degree[0],gt_degree[1],gt_py_value))
if(MG_Label):
    gt_file = pd.read_csv("C:/Users/OJK/Task/Gaze_Estimation/input/MPIIGaze/Train_data_set/label/MPII_GAZE_whole.csv")
    gaze_vector_gt = [gt_file["gaze_x"][file_index], gt_file["gaze_y"][file_index], gt_file["gaze_z"][file_index]] 
    gaze_vector_gt = np.array(gaze_vector_gt)
    gx , gy, gz = gaze_vector_gt
    gy = -gy                                  # MPII는 반전 되어 있음 y축 기준으로
    # gaze_vector_gt[1] =  -gaze_vector_gt[1] 
    gaze_vector_gt = gaze_vector_gt/np.linalg.norm(gaze_vector_gt)
    gt_py_value = np.array([math.asin(float(gaze_vector_gt[1])),math.atan2(-float(gaze_vector_gt[0]),-float(gaze_vector_gt[2]))])
    gt_degree = np.rad2deg(gt_py_value)
    print("[GT] gaze : ", gaze_vector_gt)
    print("[GT] pitch : {:.3f}°, yaw : {:.3f}°, py_value : {}".format(gt_degree[0],gt_degree[1],gt_py_value))


cen = (int(lmks[17][0]), int(lmks[17][1]))
if(MG_Label):
    end = (int(lmks[17][0] + y_pred[0]*100), int(lmks[17][1] - y_pred[1] * 100))
if(UE_Label):
    end = (int(lmks[17][0] + y_pred[0]*100), int(lmks[17][1] - y_pred[1] * 100))
if(MG_Label):
    gt_end = (int(lmks[17][0] + gaze_vector_gt[0]*100), int(lmks[17][1] + gaze_vector_gt[1] * 100)) # 
if(UE_Label):
    gt_end = (int(lmks[17][0] + gt_vector[0] * 100 ), int(lmks[17][1] - gt_vector[1] * 100))
cv2.arrowedLine(img, cen, end, (0, 255, 255), 1, cv2.LINE_AA) 
cv2.arrowedLine(img, cen, gt_end, (255,255,0),1,cv2.LINE_AA)  

# Ground truth

# 3D gaze vector 출력
print('[Preds] gaze : ',y_pred )
print('[Preds] pitch : {:.3f}°, yaw : {:.3f}°, py_value : {}'.format(y_pred[0]/np.pi*180, y_pred[1]/np.pi*100, y_pred)) 

cv2.imshow('image', img)

cv2.waitKey(0)

# cv2.destroyAllWindows()
