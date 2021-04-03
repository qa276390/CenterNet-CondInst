import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

def multiply_local_shape_and_map(local_shape, saliency_map, pred_wh, ind, reg):
  
  #  local_shape (batch, max_objects, dim) 
  #  saliency_map (batch, 1, h, w)
  #  pred_wh (batch, max_objects, 2)
  #  ind (batch, max_objects)
  
  
  max_objects = ind.size(1)
  batch_size = ind.size(0)
  W = saliency_map.size(3)
  H = saliency_map.size(2)
  S = int(local_shape.size(2)**0.5)
  reshape_local_shape = torch.reshape(local_shape, (local_shape.size(0), local_shape.size(1), S, S)) # (batch, max_objects, S, S) 
  saliency_map_expand = saliency_map.unsqueeze(1).expand(saliency_map.size(0), max_objects, saliency_map.size(1), saliency_map.size(2), saliency_map.size(3)) 
  # saliency_map_expand (batch, max_objects, 1, h, w )
  masking_with_local_shape = torch.zeros_like(saliency_map_expand)
  masking = torch.zeros_like(saliency_map_expand)

  CT0 = (ind % W).float() + reg[:, :, 0]
  CT1 = (ind // W).float() + reg[:, :, 1]

  dX = -(CT0 - W//2) * 2 / W
  dY = -(CT1 - H//2) * 2 / H
  Theta = torch.tensor([
            [1,0,0],
            [0,1,0]
          ], dtype=torch.float)  
  Theta = Theta.unsqueeze(0).unsqueeze(0).expand(batch_size, max_objects, 2, 3).to(reshape_local_shape.device)
  Theta[:, :, 0, 2] = dX
  Theta[:, :, 1, 2] = dY

  #start_time = time.time()
  for b in range(batch_size):
    for o in range(max_objects):
      boxw, boxh = math.ceil(pred_wh[b, o, 0]),  math.ceil(pred_wh[b, o, 1]) 
      boxw = boxw if boxw % 2 == 0 else boxw + 1
      boxh = boxh if boxh % 2 == 0 else boxh + 1

      if boxh <= 0 or boxw <= 0:
        resized_shape = 0
        masking_with_local_shape[b, o, ...] = resized_shape
      else:
        resized_shape = F.interpolate(reshape_local_shape[b, o, :, :].unsqueeze(0).unsqueeze(0), size=tuple((boxh, boxw)))
      
        try:
          #start_time_1 = time.time()
          pad_shape = F.pad(resized_shape, (W//2-boxw//2, W//2-boxw//2, H//2-boxh//2, H//2-boxh//2))
          theta = Theta[b, o]
          grid = F.affine_grid(theta.unsqueeze(0), pad_shape.size())
          output = F.grid_sample(pad_shape, grid)
          shift_shape = output[0]
          masking_with_local_shape[b, o, ...] = shift_shape
          #print(f'assign: {time.time()-start_time_1}')
        except Exception as e:
          print('-----------------------error!-----------------------', e)
          print('boxh, boxw', boxh, boxw)
          print('ct_0', 'ct_1', CT0[b, o], CT1[b, o])
          print('resized_shape, pad_shape', resized_shape.size(),pad_shape.size())
          import sys
          sys.exit(1)
  #print(f'full batch: {time.time()-start_time}')

  inst_segs = saliency_map_expand * masking_with_local_shape
  #inst_segs = masking_with_local_shape
  #inst_segs = saliency_map_expand * masking
  
  
  #from matplotlib import pyplot as plt
  #for i in range(10):
  #  tshape = masking_with_local_shape[0, i, :, :].cpu().numpy().transpose(1, 2, 0)
  # plt.imshow(tshape)
  #  plt.savefig(f'./cache/tmp_shape_{i}.png')
  #  qshape = (tshape > 0.5).astype(int)
  #  plt.imshow(qshape)
  #  plt.savefig(f'./cache/qun_shape_{i}.png')
  
  return inst_segs, masking_with_local_shape # (batch, max_objects, h, w )

"""

def multiply_local_shape_and_map(local_shape, saliency_map, pred_wh, ind, reg):

  
  max_objects = ind.size(1)
  batch_size = ind.size(0)
  W = saliency_map.size(3)
  H = saliency_map.size(2)
  S = int(local_shape.size(2)**0.5)
  reshape_local_shape = torch.reshape(local_shape, (local_shape.size(0), local_shape.size(1), S, S)) # (batch, max_objects, S, S) 
  saliency_map_expand = saliency_map.unsqueeze(1).expand(saliency_map.size(0), max_objects, saliency_map.size(1), saliency_map.size(2), saliency_map.size(3)) 
  # saliency_map_expand (batch, max_objects, 1, h, w )
  masking_with_local_shape = torch.zeros_like(saliency_map_expand)
  masking = torch.zeros_like(saliency_map_expand)
  #start_time = time.time()
  for b in range(batch_size):
    for o in range(max_objects):
      idx = ind[b, o]
      ct_0 = int(idx) % W # idxH
      ct_1 = math.floor(int(idx) / W) # idxW
      #print('pred_wh', pred_wh[b, o])
      boxw, boxh = int(pred_wh[b, o, 0]) + 1, int(pred_wh[b, o, 1]) + 1

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
        masking[b, o, :, ct_1-hfh_lo:ct_1+hfh_up , ct_0-hfw_lo:ct_0+hfw_up] = 1
        #masking_with_local_shape[b, o, :, ct_1-hfh_lo:ct_1+hfh_up , ct_0-hfw_lo:ct_0+hfw_up] =  1 
        #print(f'assign: {time.time()-start_time_1}')
      except:
        print('-----------------------error!-----------------------')
        print('boxh, boxw', boxh, boxw)
        print('hfw_lo_', 'hfw_up_', hfw_lo_, hfw_up_)
        print('hfw_lo', 'hfw_up', hfw_lo, hfw_up)
        print('ct_0', 'ct_1', ct_0, ct_1)
        print('resized_shape', resized_shape.size())
        print('into shape', masking_with_local_shape[b, o, :, ct_1-hfh_lo:ct_1+hfh_up , ct_0-hfw_lo:ct_0+hfw_up].size())
        import sys
        sys.exit(1)
  #print(f'full batch: {time.time()-start_time}')

  inst_segs = saliency_map_expand * masking_with_local_shape
  #inst_segs = masking_with_local_shape
  #inst_segs = saliency_map_expand * masking

  return inst_segs, masking_with_local_shape # (batch, max_objects, h, w )
"""