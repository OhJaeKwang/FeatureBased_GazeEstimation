import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.utils import utils_inference
from lib.utils import utils_landmarks
from lib.ransac import ransac

### Eye Landmarks Detection
method = '2st'
ckpt = 50
model = utils_inference.get_model_by_name('C:/Users/OJK/Task/Gaze_Estimation/eye_landmarks_detection/tools/output/unityeyes/eye_alignment_unityeyes_hrnet_w18/backup/' + method + '/checkpoint_{}.pth'.format(ckpt),
                                          'C:/Users/OJK/Task/Gaze_Estimation/eye_landmarks_detection/experiments/unityeyes/eye_alignment_unityeyes_hrnet_w18.yaml',
                                          device='cuda')

# img = plt.imread('C:/Users/yklee/eye_landmarks_detection/data/unityeyes/images/40001.jpg')
img = plt.imread('D:/Unityeyes_dataset/640x480/Test_data/img/5.jpg')

crop_size = 192
img_shape = img.shape
if img_shape[0] != crop_size or img_shape[1] != crop_size:
    cen_x = int(img_shape[1] / 2)
    cen_y = int(img_shape[0] / 2)
    img = img[cen_y-int(crop_size/2):cen_y+int(crop_size/2), cen_x-int(crop_size/2):cen_x+int(crop_size/2)]

img = cv2.resize(img,dsize=(160,96))
lmks, conf_score = utils_inference.get_lmks_by_img_v3(model, img, conf_score=False)
utils_landmarks.show_landmarks(img, lmks)

### Ellipse RANSAC
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
pnts = list(lmks[18:])

# 일부 Landmarks만 사용 -> 비추천
# lmks, conf_score = list(lmks), np.reshape(np.array(conf_score), 50)
# iris_lmks, iris_score = lmks[18:], conf_score[18:]
#
# conf_argsort = iris_score.argsort()
#
# pnts = []
# for i in range(16):
#     pnts.append(iris_lmks[conf_argsort[i]])

ellipse_params = ransac.FitEllipse_RANSAC(np.array(pnts), gray)
print(ellipse_params)

# for circle in pnts:
#     cv2.circle(img, (int(np.round(circle[0])), int(np.round(circle[1]))), 2, (0, 0, 255), -1)
cv2.ellipse(img, ellipse_params, (0, 0, 255), 1)
plt.imshow(img)
plt.show()
