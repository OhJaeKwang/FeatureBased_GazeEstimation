# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import torch
import torch.nn.functional as F
import numpy as np
import math
import cv2
from lib.utils import utils_landmarks
from PIL import Image
from tensorboardX import SummaryWriter



from .evaluation import decode_preds, compute_nme , compute_angle_error

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0

    return res


def train(config, train_loader, model, criterion, optimizer,
          epoch, writer_dict):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()


    model.train()
    nme_count = 0
    nme_batch_sum = 0

    angle_errors_sum = 0
    angle_errors_count = 0

    end = time.time()

    for i, ( inp, target, meta, gaze, pitch_yaw ) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time()-end)

        # compute the output
        htmap , outputs = model(inp) # shape [32, 50, 24, 40] ,
        # 히트맵 무조건 4배로 줄게끔 설게 되있나 보다..
        target = target.cuda(non_blocking=True) # [32, 50, 24, 40] -> [imgs, lmks, res]
        # shape [32, 50, 24, 40]
    
        MSE_loss = criterion(htmap, target)
        
        # NME  ???
        
        score_map = htmap.data.cpu()
        preds = decode_preds(score_map, meta['center'], meta['scale'], [96, 160]) # [32, 50, 2] -> [imgs, lmks, coords] , [rows,cols]
        
        # show gt
        # gt = decode_preds(target, meta['center'], meta['scale'], [24, 40])
        # temp = temp[0].squeeze(dim=0)
        # temp = np.array([t.detach().numpy() for t in temp])
        # # img = img.transpose([1,2,0])
        # gt = gt[0].squeeze(dim=0)
        # gtt = np.array([t.detach().numpy() for t in gt])
        # img2 = temp.copy()
        # cv2.imshow("bbb",img2)
        # img2 = utils_landmarks.set_circles_on_img(img2, gtt,circle_size=1,is_copy=False)
        # cv2.imshow("aaa",img2)
        # cv2.waitKey(0)

        # preds는 192x192에서의 landmarks 좌표
        nme_batch = compute_nme(preds, meta)
        nme_batch_sum = nme_batch_sum + np.sum(nme_batch)
        nme_count = nme_count + preds.size(0)


        # angle_errors_batch = compute_angle_error(outputs,meta)
        # angle_errors_sum = angle_errors_sum + np.sum(angle_errors_batch)
        # angle_errors_count = angle_errors_count + preds.size(0)

        # # Distance Loss
        # with torch.no_grad():
        #     preds1 = preds.view(preds.shape[0], -1)
        #     pd_dist = pdist(preds1, squared=False)
        #     mean_pd = pd_dist[pd_dist > 0].mean()
        #     pd_dist = pd_dist / mean_pd
        #
        # meta_pts1 = meta['pts'].view(preds.shape[0], -1)
        # md_dist = pdist(meta_pts1, squared=False)
        # mean_md = md_dist[md_dist > 0].mean()
        # md_dist = md_dist / mean_md
        #
        # distance_loss = F.smooth_l1_loss(pd_dist, md_dist, reduction='elementwise_mean')

        # Angle Loss

        preds2 = preds.view(preds.shape[0], -1) # [bathchsize, 100] , flatten
        pd_angle = (preds2.unsqueeze(0) - preds2.unsqueeze(1)) #[bathchsize,bathchsize,100] --> 각 장의 이미지를 기준으로 좌표값들이 어떻게 되어 있는지 설명
        norm_pd = F.normalize(pd_angle, p=2, dim=2) # 각 장의 이미지의 값들로 normalize
        pd_angle = torch.bmm(norm_pd, norm_pd.transpose(1, 2)).view(-1)
        

        meta_pts2 = meta['pts'].view(preds.shape[0], -1)
        md_angle = (meta_pts2.unsqueeze(0) - meta_pts2.unsqueeze(1))
        norm_md = F.normalize(md_angle, p=2, dim=2)
        md_angle = torch.bmm(norm_md, norm_md.transpose(1, 2)).view(-1)
        angle_loss = F.smooth_l1_loss(pd_angle, md_angle, reduction='mean') # 이게 왜 angle loss지 ?
        
        # print("angle_loss_grd:",angle_loss.requires_grad)

        # Cosine distance Loss

        

        # cosine 유도 값이 1이 되야함 따라서 이놈은 cos(세타) =1 값
        predict_value = []
        
        pitch_yaw = pitch_yaw.view(-1,2)
        pitch_yaw = pitch_yaw.cuda(non_blocking=True)

        # meta_gaze = meta_gaze.view(-1,3)
        # meta_gaze = meta_gaze.cuda(non_blocking=True)

        ct = torch.nn.MSELoss(size_average=True).cuda()

        # print(f'pit_y shape : {pitch_yaw.shape}')
        # print(f'outp shape : {outputs.shape}')
        # print(f'pit_y shape : {pitch_yaw}')
        # print(f'outp shape : {outputs}')

        gaze_loss = ct(outputs, pitch_yaw.float())
        # Total Loss
        MSE_ratio = 1
        # distance_ratio = 0.01
        angle_ratio = 0.01
        cosine_ratio = 1 
        gaze_ratio = 100

        # print("m: ", MSE_loss)
        # print("g: ", gaze_loss)

        if(np.isnan(gaze_loss.cpu().detach().numpy())):
            print("gaze sibal")
        if(np.isnan(MSE_loss.cpu().detach().numpy())):
            print("heatmap sibal")
        if(np.isnan(angle_loss.cpu().detach().numpy())):
            print("aaaaa sibal")
        
        # loss = (MSE_ratio * MSE_loss) + (distance_ratio * distance_loss) + (angle_ratio * angle_loss)
        # loss = (MSE_ratio * MSE_loss) + (angle_ratio * angle_loss)
        loss = (MSE_ratio * MSE_loss) + (angle_ratio * angle_loss) 

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))
        # tensorboard
        

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    nme = nme_batch_sum / nme_count
    angle_loss = angle_errors_sum / angle_errors_count
    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} angle:{:4f}'\
        .format(epoch, batch_time.avg, losses.avg, nme ,angle_loss)
    logger.info(msg)

def validate(config, val_loader, model, criterion, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta, gaze) in enumerate(val_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            target = target.cuda(non_blocking=True)

            score_map = output.data.cpu()
            # loss
            loss = criterion(output, target)

            preds = decode_preds(score_map, meta['center'], meta['scale'], [24, 40])
            # NME
            nme_temp = compute_nme(preds, meta)
            # Failure Rate under different threshold
            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_nme', nme, global_steps)

        writer_dict['valid_global_steps'] = global_steps + 1

    return nme, predictions


def inference(config, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(data_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            score_map = output.data.cpu()
            preds = decode_preds(score_map, meta['center'], meta['scale'], [48, 48])

            # NME
            nme_temp = compute_nme(preds, meta)

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    return nme, predictions
