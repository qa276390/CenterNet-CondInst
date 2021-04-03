from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from torchvision.utils import make_grid
import cv2

from models.losses import FocalLoss,DiceLoss, MaskBCELoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer
from models.decode import mtseg_decode

import time



class MtsegLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MtsegLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.crit_mask = MaskBCELoss()
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, mask_loss = 0, 0, 0,0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])
            output['saliency_map'] = _sigmoid(output['saliency_map'])
            output['local_shape'] = _sigmoid(output['local_shape'])


            if opt.eval_oracle_hm:
                output['hm'] = batch['hm']
            if opt.eval_oracle_wh:
                output['wh'] = torch.from_numpy(gen_oracle_map(
                    batch['wh'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
            if opt.eval_oracle_offset:
                output['reg'] = torch.from_numpy(gen_oracle_map(
                    batch['reg'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                if opt.dense_wh:
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    wh_loss += (
                                       self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                                    batch['dense_wh'] * batch['dense_wh_mask']) /
                                       mask_weight) / opt.num_stacks
                elif opt.cat_spec_wh:
                    wh_loss += self.crit_wh(
                        output['wh'], batch['cat_spec_mask'],
                        batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
                else:
                    wh_loss += self.crit_reg(
                        output['wh'], batch['reg_mask'],
                        batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks
            #################################################################################
            #                                  head to mask                                 #
            #                                                                               #
            #################################################################################
            #start_time = time.time()
            mask_loss+=self.crit_mask(output['local_shape'], output['saliency_map'], output['wh'], output['reg'],
                                      batch['reg_mask'], batch['ind'], batch['wh'], batch['instance_mask'], batch['reg'])
            #print(f'full batch loss: {time.time()-start_time}')
        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
               opt.off_weight * off_loss + opt.seg_weight * mask_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss,"mask_loss":mask_loss}
        return loss, loss_stats


class MtsegTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MtsegTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss','mask_loss']
        loss = MtsegLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        writer = self.writer
        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None
        """
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=opt.cat_spec_wh, K=opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, :4] *= opt.down_ratio
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= opt.down_ratio
        """
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        saliency_map = output['saliency_map'].sigmoid_()
        local_shape = output['local_shape'].sigmoid_()

        dets_mt, masks, pred_local_shape, resize_local_shape = mtseg_decode(hm, wh, saliency_map, local_shape, reg=reg, cat_spec_wh=opt.cat_spec_wh, K=opt.K)

       

        for i in range(1):
            
            #debugger = Debugger(
            #    dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            print('img', batch['input'][i].size())
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            img = img.transpose(2, 0, 1)
            """
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                           dets[i, k, 4], img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            """
            #print('pred_local_shape', pred_local_shape.size())
            S = int(pred_local_shape.size(2)**0.5)
            single_local_shape = pred_local_shape[i]
            reshape_local_shape = torch.reshape(single_local_shape, (single_local_shape.size(0), S, S)).unsqueeze(1)
            local_grid = make_grid(reshape_local_shape)
            single_resize_local_shape = resize_local_shape[i]
            _resize_local_shape = single_resize_local_shape.unsqueeze(2).reshape(single_resize_local_shape.size(0)*single_resize_local_shape.size(1), 1, single_resize_local_shape.size(2), single_resize_local_shape.size(3))
            resize_local_grid = make_grid(_resize_local_shape)
            smap = saliency_map[i]
            mask = masks[i].unsqueeze(0)
            _masks = mask.unsqueeze(2).reshape(mask.size(0)*mask.size(1), 1, mask.size(2), mask.size(3))
            mask_grid = make_grid(_masks)
            writer.add_image('pred_masks',mask_grid,iter_id)
            writer.add_image('pred_local_shape',local_grid, iter_id)
            writer.add_image('pred_resize_local_shape',resize_local_grid, iter_id)
            writer.add_image('pred_saliency_map',smap , iter_id)
            writer.add_image('gt_img', img, iter_id)
            """
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                           dets_gt[i, k, 4], img_id='out_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)
            """
    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]