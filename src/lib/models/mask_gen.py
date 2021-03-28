import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

def multiply_local_shape_and_map(local_shape, saliency_map, pred_wh, ind):
  """
    local_shape (batch, max_objects, dim) 
    saliency_map (batch, 1, h, w)
    pred_wh (batch, max_objects, 2)
    ind (batch, max_objects)
  """
  
  max_objects = ind.size(1)
  batch_size = ind.size(0)
  W = saliency_map.size(3)
  H = saliency_map.size(2)
  S = int(local_shape.size(2)**0.5)
  reshape_local_shape = torch.reshape(local_shape, (local_shape.size(0), local_shape.size(1), S, S)) # (batch, max_objects, S, S) 
  saliency_map_expand = saliency_map.unsqueeze(1).expand(saliency_map.size(0), max_objects, saliency_map.size(1), saliency_map.size(2), saliency_map.size(3)) 
  # saliency_map_expand (batch, max_objects, 1, h, w )
  masking_with_local_shape = torch.zeros_like(saliency_map_expand)
  #start_time = time.time()
  for b in range(batch_size):
    for o in range(max_objects):
      idx = ind[b, o]
      ct_0 = int(idx) % W # W
      ct_1 = math.floor(int(idx) / W) # H
      #print('pred_wh', pred_wh[b, o])
      boxh, boxw = round(float(pred_wh[b, o, 0])), round(float(pred_wh[b, o, 1]))

      hfh_lo_, hfh_up_ = int(boxh / 2), math.ceil(boxh / 2) # h/2
      hfw_lo_, hfw_up_ = int(boxw / 2), math.ceil(boxw / 2) # w/2
      hfw_lo, hfw_up = hfw_lo_ if ct_0 - hfw_lo_ >= 0 else ct_0,  hfw_up_ if ct_0 + hfw_up_ <= W else W - ct_0
      hfh_lo, hfh_up = hfh_lo_ if ct_1 - hfh_lo_ >= 0 else ct_1,  hfh_up_ if ct_1 + hfh_up_ <= H else H - ct_1

      #start_time_1 = time.time()
      if boxh <= 0 or boxw <= 0:
        resized_shape = 0
      else:
        resized_shape = F.interpolate(reshape_local_shape[b, o, :, :].unsqueeze(0).unsqueeze(0), size=tuple((boxh, boxw)))
        resized_shape = resized_shape.squeeze(0)[:, hfh_lo_-hfh_lo:hfh_up+hfh_lo_, hfw_lo_-hfw_lo:hfw_up+hfw_lo_]
      #print(f'interpolate: {time.time()-start_time_1}, (boxh, boxw), {boxh, boxw}')
       
      try:
        #start_time_1 = time.time()
        masking_with_local_shape[b, o, :, ct_1-hfh_lo:ct_1+hfh_up , ct_0-hfw_lo:ct_0+hfw_up] = resized_shape * 1 
        #print(f'assign: {time.time()-start_time_1}')
      except:
        print('error!')
        print('boxh, boxw', boxh, boxw)
        print('hfw_lo_', 'hfw_up_', hfw_lo_, hfw_up_)
        print('hfw_lo', 'hfw_up', hfw_lo, hfw_up)
        print('ct_0', 'ct_1', ct_0, ct_1)
        print('resized_shape', resized_shape.size())
        print('in shape', masking_with_local_shape[b, o, :, ct_1-hfh_lo:ct_1+hfh_up , ct_0-hfw_lo:ct_0+hfw_up].size())
        import sys
        sys.exit(1)
  #print(f'full batch: {time.time()-start_time}')
  inst_segs = saliency_map_expand * masking_with_local_shape
  return inst_segs # (batch, max_objects, h, w )