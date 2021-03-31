from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
from pycocotools import  mask as mask_utils
"""
try:
    from external.nms import soft_nms
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
"""
from models.decode import mtseg_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctseg_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector


class MtsegDetector(BaseDetector):
    def __init__(self, opt):
        super(MtsegDetector, self).__init__(opt)

    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model(images)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            saliency_map = output['saliency_map'].sigmoid_()
            local_shape = output['local_shape'].sigmoid_()
            reg = output['reg'] if self.opt.reg_offset else None
            assert not self.opt.flip_test,"not support flip_test"
            torch.cuda.synchronize()
            forward_time = time.time()
            dets, masks, _ = mtseg_decode(hm, wh, saliency_map, local_shape, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)

        if return_time:
            return output, (dets,masks), forward_time
        else:
            return output, (dets,masks)

    def post_process(self, det_seg, meta, scale=1):
        assert scale == 1, "not support scale != 1"
        dets,seg = det_seg
        dets = dets.detach().cpu().numpy()
        seg = seg.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctseg_post_process(
            dets.copy(),seg.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'],*meta['img_size'], self.opt.num_classes)
        return dets[0]

    def merge_outputs(self, detections):
        return detections[0]


    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='mtseg')
        for j in range(1, self.num_classes + 1):
            for i in range(len(results[j]['boxs'])):
                bbox=results[j]['boxs'][i]
                mask = mask_utils.decode(results[j]['pred_mask'][i])
                if bbox[4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='mtseg')
                    debugger.add_coco_seg(mask,img_id='mtseg')
        #debugger.show_all_imgs(pause=self.pause)
        debugger.save_all_imgs()
