import torch
from torch import nn
from torch.nn import Module
import os
import sys
import cv2
import math
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.utils import utils_inference
from lib.utils import utils_landmarks
from lib.ransac import ransac
from lib.gaze_estimation import gaze_estimation


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Settings
method = 'unityeyes_angle'
ckpt = 13
model = utils_inference.get_model_by_name('C:/Users/yklee/eye_landmarks_detection/tools/output/unityeyes/eye_alignment_unityeyes_hrnet_w18/backup/' + method + '/checkpoint_{}.pth'.format(ckpt),
                                          'C:/Users/yklee/eye_landmarks_detection/experiments/unityeyes/eye_alignment_unityeyes_hrnet_w18.yaml',
                                          device='cuda')

gaze_estimator = gaze_estimation.GazeEstimator().to(device)
gaze_estimator.load_state_dict(torch.load('C:/Users/yklee/gaze_estimation/gaze_estimator_weights.pth'))
gaze_estimator.eval()

# Input / Output Settings
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('C:/Users/yklee/eye_landmarks_detection/data/sample/webcam_output.avi', fourcc, 12.0, (192, 192))

# EAR Threshold Setting
ear = 0
ear_thres = 0.3
ear_adapt = 0

max_frame_cnt = 20
frame_cnt = 1


################################ Predictions ################################
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        start_time = time.time()

        img_size = [192, 192]
        half_size = [int(img_size[0] / 2), int(img_size[1] / 2)]
        frame_center = [int(frame.shape[0] / 2), int(frame.shape[1] / 2)]
        frame = frame[frame_center[0] - half_size[0]:frame_center[0] + half_size[0], frame_center[1] - half_size[1]:frame_center[1] + half_size[1]]
        # frame = cv2.flip(frame, 1)
        frame_inf = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # shape : (192, 192, 3)

        ##### Eye Landmarks Detection #####
        lmks = utils_inference.get_lmks_by_img(model, frame_inf)   # inference
        inf_time = time.time() - start_time
        img = utils_landmarks.set_circles_on_img(frame.copy(), lmks, circle_size=1, is_copy=False)   # plot
                                                                                                     # shape : (50, 2)

        interior, iris = lmks[1:17], lmks[18:]   # 눈 중심, 가장자리 : 0, 1 ~ 16 / 홍채 중심, 가장자리 : 17, 18 ~ 49
                                                 # shape : (16, 2), (32, 2)

        # EAR
        lx = float((lmks[0][0] + lmks[1][0] + lmks[2][0]) / 3.0)
        rx = float((lmks[8][0] + lmks[9][0] + lmks[10][0]) / 3.0)
        by = float((lmks[12][1] + lmks[13][1] + lmks[14][1]) / 3.0)
        ty = float((lmks[4][1] + lmks[5][1] + lmks[6][1]) / 3.0)

        if (rx - lx) != 0:
            ear = float((ty - by) / (rx - lx))

        if frame_cnt <= max_frame_cnt:   # max_frame_cnt만큼 EAR 샘플링 후
            if ear_adapt < ear * ear_thres:
                ear_adapt = ear * ear_thres

            frame_cnt += 1

        # if ear > ear_adapt:   # EAR
        ##### Hough Transform #####
        xs = [i[0] for i in interior]   # image shape : [y, x, rgb] / landmarks shape : [x, y]
        ys = [i[1] for i in interior]
        min_x, max_x = int(min(xs)), int(max(xs))
        min_y, max_y = int(min(ys)), int(max(ys))

        roi = np.array(frame_inf[min_y:max_y, min_x:max_x, :])   # RoI region 설정
        roi_cost = np.sum(roi, axis=2, dtype=np.int32)   # 검정색에서 먼 pixel일수록 높은 cost를 갖게함
        roi_gray = np.array(roi_cost * (255 / np.amax(roi_cost)), dtype=np.uint8)   # 0~255로 normalization
        roi_thres = np.array((roi_gray > 35) * 255, dtype=np.uint8)   # 0 or 1로 thresold 적용

        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 0, 0), 1)

        all_cost = np.sum(frame_inf.copy(), axis=2, dtype=np.int32)   # 전체 이미지에 대하여 검정색에서 먼 pixel일수록 높은 cost를 갖게함
        all_gray = np.array(all_cost * (10 / np.amax(all_cost)), dtype=np.uint8)   # 0~255로 normalization

        # plt.imshow(roi_thres, cmap='Greys')
        # plt.show()

        circles = cv2.HoughCircles(roi_thres, cv2.HOUGH_GRADIENT, 1, 30, 75, 25, 10, 40)   # Hough transform 적용하여 원들 찾기
        circles = np.uint16(np.around(circles))   # 원들 자료형 변환

        #### Circles Post-procssing #####
        inner_costs = []   # 원 내부 pixels cost 수집을 위한 리스트 선언
        if len(circles.shape) == 3:   # 탐지된 원이 없는 경우, 건너뛰기
            for i, circle in enumerate(circles[0, :]):   # 원들 개별 접근 및 indexing
                if circle[1] < roi.shape[0] and circle[0] < roi.shape[1]:   # 중심 좌표가 image shape를 넘어가는 index를 갖는 원 건너뛰기
                    cen_x, cen_y, radius = int(circle[0]), int(circle[1]), int(circle[2])
                    inner_circle = all_gray[cen_y - radius + min_y:cen_y + radius + min_y, cen_x - radius + min_x:cen_x + radius + min_x]

                    # cv2.rectangle(img, (cen_x - radius + min_x, cen_y - radius + min_y), (cen_x + radius + min_x, cen_y + radius + min_y), (0, 0, 0), 1)
                    # cv2.circle(img, (cen_x + min_x, cen_y + min_y), radius, (0, 0, 0), 1)
                    # plt.imshow(img)
                    # plt.show()

                    inner_cost = np.mean(inner_circle)
                    inner_costs.append([i, inner_cost])

            costs = [cost[1] for cost in inner_costs]
            min_cost = costs.index(min(costs))
            min_idx = inner_costs[min_cost][0]
            best_circle = circles[0][min_idx]

            img = cv2.circle(img, (best_circle[0] + min_x, best_circle[1] + min_y), best_circle[2], (0, 0, 255), 1)

        total_time = time.time() - start_time
        print('[Speed] Inference : {:.2f} FPS, Post-processing : {:.2f} FPS, Total : {:.2f} FPS'.format(
              1 / inf_time, 1 / (total_time - inf_time), 1 / total_time))

        ##### Gaze Estimation #####
        cen = (int(lmks[0][0]), int(lmks[0][1]))
        end = (int(lmks[17][0]), int(lmks[17][1]))
        cv2.arrowedLine(img, cen, end, (0, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(img, '[Preds] pitch : {:.3f}°, yaw : {:.3f}°, gaze_vector : {}' \
        #             .format(pitch_pred*100, yaw_pred*100, y_pred), (0, 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255))

        cv2.imshow('Image', img)
        output.write(img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
