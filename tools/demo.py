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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Settings
method = 'unityeyes_angle'
ckpt = 13
model = utils_inference.get_model_by_name('C:/Users/OJK/Task/Gaze_Estimation/eye_landmarks_detection/tools/output/unityeyes/eye_alignment_unityeyes_hrnet_w18/backup/' + method + '/checkpoint_{}.pth'.format(ckpt),
                                          'C:/Users/OJK/Task/Gaze_Estimation/eye_landmarks_detection/experiments/unityeyes/eye_alignment_unityeyes_hrnet_w18.yaml',
                                          device='cuda')

gaze_estimator = gaze_estimation.GazeEstimator().to(device)
gaze_estimator.load_state_dict(torch.load('C:/Users/OJK/Task/Gaze_Estimation/eye_landmarks_detection/lib/gaze_estimation/gaze_estimator_weights.pth'))
gaze_estimator.eval()

# Input / Output Settings
input = 'C:/Users/OJK/Task/Gaze_Estimation/eye_landmarks_detection/data/sample/ojk_input2.mp4'   # 
cap = cv2.VideoCapture(input)
#cap = cv2.VideoCapture(0)   

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# output = cv2.VideoWriter('C:/Users/yklee/eye_landmarks_detection/data/sample/webcam_output.avi', fourcc, 30.0, (192, 192))
output = cv2.VideoWriter('C:/Users/OJK/Task/Gaze_Estimation/eye_landmarks_detection/data/sample/ojk_angle2.avi', fourcc, 20.0, (192, 192))

# EAR Threshold Setting
ear_thres = 0.3
ear_adapt = 0

max_frame_cnt = 20
frame_cnt = 1


##### Predictions #####

while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret:
        img_size = [192, 192]
        half_size = [int(img_size[0] / 2), int(img_size[1] / 2)] # [96,96]
        frame_center = [int(frame.shape[0] / 2), int(frame.shape[1] / 2)] # [240,320]
        frame = frame[frame_center[0] - half_size[0]:frame_center[0] + half_size[0], frame_center[1] - half_size[1]:frame_center[1] + half_size[1]] # [ 240-96 : 240+96 , 320-96 : 320+96 ] --> 입력 이미지에서 192x192로 crop --> 입력을 192x192에서 수정해보기?
        # frame = cv2.resize(frame, dsize=(192, 192), interpolation=cv2.INTER_AREA)
        frame_inf = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        ##### Eye Landmarks Detection #####
        lmks = utils_inference.get_lmks_by_img(model, frame_inf)    
        img = utils_landmarks.set_circles_on_img(frame.copy(), lmks, circle_size=1, is_copy=False) 
 
        interior, iris = lmks[1:17], lmks[18:]


        ##### Gaze Estimation #####

        input_data = lmks.copy()   

        ws = []   
        for idx in input_data:
            ws.append(idx[0])
        w = max(ws) - min(ws)  

        for idx in input_data:   # Normalization   
            idx[0] -= lmks[0][0]    
            idx[0] /= w
            idx[1] -= lmks[0][1]
            idx[1] /= w
        input_data[0][0]=0   
        input_data[0][1]=0

        x_inputs = torch.from_numpy(input_data).to(device)
        x_input = x_inputs.reshape(1, 100)
        y_pred = gaze_estimator(x_input.float())

        x_input = x_input[0] 
        y_pred = y_pred[0]  

        yaw = np.arctan2(-y_pred[0],-y_pred[2])
        pitch = np.arcsin(-y_pred[1])

        cen = (int(lmks[17][0]), int(lmks[17][1]))
        end = (int(y_pred[0] * 100 + lmks[17][0]), int(y_pred[1] * 100 + lmks[17][1]))
        cv2.arrowedLine(img, cen, end, (0, 255, 255), 1, cv2.LINE_AA)


        # 3D gaze vector
        print('[Preds] pitch : {:.3f}°, yaw : {:.3f}°, gaze_vector : {}'.format(pitch/np.pi*180, yaw/np.pi*100, y_pred))

        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        output.write(img)
    else:
        break

cap.release()
cv2.destroyAllWindows()
