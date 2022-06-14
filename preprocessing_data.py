import os
import argparse
import numpy as np
import pandas as pd
import scipy.io
import cv2


def convert_pose(vect):                  # 로드리게스로 얻는것이 무언인가?  헤드 포지션을 피치 야로 ?
    M, _ = cv2.Rodrigues(np.array(vect).astype(np.float32))        
    vec = M[:, 2]
    yaw = np.arctan2(vec[0], vec[2])
    pitch = np.arcsin(vec[1])
    return np.array([yaw, pitch])


def convert_gaze(vect):                  # 3축의 gaze vector를 피치,야로 변경
    x, y, z = vect
    yaw = np.arctan2(-x, -z)
    pitch = np.arcsin(-y)
    return np.array([yaw, pitch])

def conver_gaze_inver(pitch_yaw):
    pitch , yaw = pitch_yaw
    x_z = np.tan(pitch)     

    return


def get_eval_info(subject_id, evaldir):   # evaldir = C:/Users/OJK/Task/Gaze_Estimation/input/MPIIGaze/Evaluation Subset/sample list for eye image
    df = pd.read_csv(
        os.path.join(evaldir, '{}.txt'.format(subject_id)),           
        delimiter=' ',
        header=None,
        names=['path', 'side'])            
    df['day'] = df.path.apply(lambda path: path.split('/')[0])
    df['filename'] = df.path.apply(lambda path: path.split('/')[1])
    df = df.drop(['path'], axis=1)
    return df                                       


def get_subject_data(subject_id, datadir, evaldir):
    left_images = {}
    left_poses = {}
    left_gazes = {}
    right_images = {}
    right_poses = {}
    right_gazes = {}
    filenames = {}
    dirpath = os.path.join(datadir, subject_id)         # dirpath = C:/Users/OJK/Task/Gaze_Estimation/input/MPIIGaze/Data/Normalized/p00
    for name in sorted(os.listdir(dirpath)):
        path = os.path.join(dirpath, name)
        matdata = scipy.io.loadmat(
            path, struct_as_record=False, squeeze_me=True) # 각각에 day에 대한 mat 파일 읽기
        data = matdata['data']              

        day = os.path.splitext(name)[0]
        left_images[day] = data.left.image
        left_poses[day] = data.left.pose
        left_gazes[day] = data.left.gaze

        right_images[day] = data.right.image
        right_poses[day] = data.right.pose
        right_gazes[day] = data.right.gaze

        filenames[day] = matdata['filenames']

        if not isinstance(filenames[day], np.ndarray):    # np.array로 다 바꿔주는 작업    isinstance(클래스, 타입) 클래스의 elements값의 타입이 일치하면 True
            left_images[day] = np.array([left_images[day]])
            left_poses[day] = np.array([left_poses[day]])
            left_gazes[day] = np.array([left_gazes[day]])
            right_images[day] = np.array([right_images[day]])
            right_poses[day] = np.array([right_poses[day]])
            right_gazes[day] = np.array([right_gazes[day]])
            filenames[day] = np.array([filenames[day]])

    images = []
    poses = []
    gazes = []
    df = get_eval_info(subject_id, evaldir)

    for _, row in df.iterrows():
        day = row.day
        index = np.where(filenames[day] == row.filename)[0][0]    # filenames 에서의 day에서의 filename이 가지는 인덱스 반환
        if row.side == 'left':
            image = left_images[day][index]       
            pose = convert_pose(left_poses[day][index]) 
            gaze = convert_gaze(left_gazes[day][index])
        else:
            image = right_images[day][index][:, ::-1]     # 왼쪽 눈 모양처럼 뒤집어 주기 
            pose = convert_pose(right_poses[day][index]) * np.array([-1, 1])   # 왼쪽 눈 모양처럼 뒤집어 주기 
            gaze = convert_gaze(right_gazes[day][index]) * np.array([-1, 1])   # 왼쪽 눈 모양처럼 뒤집어 주기 
        images.append(image)
        poses.append(pose)
        gazes.append(gaze)

    images = np.array(images).astype(np.float32) / 255
    poses = np.array(poses).astype(np.float32)
    gazes = np.array(gazes).astype(np.float32)

    return images, poses, gazes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = "" , type=str, required=True)
    parser.add_argument('--outdir', default = "" ,type=str, required=True)
    args = parser.parse_args()


    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for subject_id in range(15):
        subject_id = 'p{:02}'.format(subject_id)

        datadir = os.path.join(args.dataset, 'Data', 'Normalized')
        evaldir = os.path.join(args.dataset, 'Evaluation Subset','sample list for eye image')

        # C:\\Users\\OJK\\Task\\Gaze_Estimation\\input\\MPIIGaze\\Evaluation Subset\\sample list for eye image
        # datadir = C:/Users/OJK/Task/Gaze_Estimation/input/MPIIGaze/Data/Normalized/
        # evaldir = C:/Users/OJK/Task/Gaze_Estimation/input/MPIIGaze/Evaluation Subset/sample list for eye image

        images, poses, gazes = get_subject_data(subject_id, datadir, evaldir)

        outpath = os.path.join(outdir, subject_id)
        np.savez(outpath, image=images, pose=poses, gaze=gazes)


if __name__ == '__main__':
    main()