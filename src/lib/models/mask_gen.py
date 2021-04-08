import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit

def multiply_local_shape_and_map(local_shape, saliency_map, wh, ind, reg):
  
  #  local_shape (batch, max_objects, dim) 
  #  saliency_map (batch, 1, h, w)
  #  pred_wh (batch, max_objects, 2)
  #  ind (batch, max_objects)
  
  max_objects = ind.size(1)
  batch_size = ind.size(0)
  N_in_batch = max_objects * batch_size
  W = saliency_map.size(3)
  H = saliency_map.size(2)
  S = int(local_shape.size(2)**0.5)
  reshape_local_shape = torch.reshape(local_shape, (local_shape.size(0), local_shape.size(1), S, S)) # (batch, max_objects, S, S) 
  saliency_map_expand = saliency_map.unsqueeze(1).expand(saliency_map.size(0), max_objects, saliency_map.size(1), saliency_map.size(2), saliency_map.size(3)) 
  # saliency_map_expand (batch, max_objects, 1, h, w )

  xs = ((ind % W).float() + reg[:, :, 0]).unsqueeze(2)
  ys = ((ind // W).float() + reg[:, :, 1]).unsqueeze(2)

  Bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)

  masks = torch.reshape(reshape_local_shape, (N_in_batch, S, S))
  bboxes = torch.reshape(Bboxes, (N_in_batch, 4))

  masking_with_local_shape = paste_masks_in_image(masks, bboxes, (H, W), threshold=-1)
  masking_with_local_shape = torch.reshape(masking_with_local_shape, (batch_size, max_objects, 1, H, W))
 
  inst_segs = saliency_map_expand * masking_with_local_shape.float()
  #inst_segs = masking_with_local_shape.float()

  return inst_segs, masking_with_local_shape # (batch, max_objects, h, w )

def _do_paste_mask(masks, boxes, img_h: int, img_w: int, skip_empty: bool = True):
  """
  Args:
      masks: N, 1, H, W
      boxes: N, 4
      img_h, img_w (int):
      skip_empty (bool): only paste masks within the region that
          tightly bound all boxes, and returns the results this region only.
          An important optimization for CPU.

  Returns:
      if skip_empty == False, a mask of shape (N, img_h, img_w)
      if skip_empty == True, a mask of shape (N, h', w'), and the slice
          object for the corresponding region.
  """
  # On GPU, paste all masks together (up to chunk size)
  # by using the entire image to sample the masks
  # Compared to pasting them one by one,
  # this has more operations but is faster on COCO-scale dataset.
  device = masks.device

  if skip_empty and not False:#torch.jit.is_scripting():
    x0_int, y0_int = torch.clamp(boxes.min(dim=0).values.floor()[:2] - 1, min=0).to(
        dtype=torch.int32
    )
    x1_int = torch.clamp(boxes[:, 2].max().ceil() + 1, max=img_w).to(dtype=torch.int32)
    y1_int = torch.clamp(boxes[:, 3].max().ceil() + 1, max=img_h).to(dtype=torch.int32)
  else:
    x0_int, y0_int = 0, 0
    x1_int, y1_int = img_w, img_h
  x0, y0, x1, y1 = torch.split(boxes, 1, dim=1)  # each is Nx1

  N = masks.shape[0]

  img_y = torch.arange(y0_int, y1_int, device=device, dtype=torch.float32) + 0.5
  img_x = torch.arange(x0_int, x1_int, device=device, dtype=torch.float32) + 0.5
  img_y = (img_y - y0) / (y1 - y0) * 2 - 1
  img_x = (img_x - x0) / (x1 - x0) * 2 - 1
  # img_x, img_y have shapes (N, w), (N, h)

  gx = img_x[:, None, :].expand(N, img_y.size(1), img_x.size(1))
  gy = img_y[:, :, None].expand(N, img_y.size(1), img_x.size(1))
  grid = torch.stack([gx, gy], dim=3)

  if not False:#torch.jit.is_scripting():
    if not masks.dtype.is_floating_point:
      masks = masks.float()
  img_masks = F.grid_sample(masks, grid.to(masks.dtype))

  if skip_empty and not False:#torch.jit.is_scripting():
    return img_masks[:, 0], (slice(y0_int, y1_int), slice(x0_int, x1_int))
  else:
    return img_masks[:, 0], ()


def paste_masks_in_image(
    masks, boxes, image_shape, threshold=0.5):
  """
  Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
  The location, height, and width for pasting each mask is determined by their
  corresponding bounding boxes in boxes.

  Note:
      This is a complicated but more accurate implementation. In actual deployment, it is
      often enough to use a faster but less accurate implementation.
      See :func:`paste_mask_in_image_old` in this file for an alternative implementation.

  Args:
      masks (tensor): Tensor of shape (Bimg, Hmask, Wmask), where Bimg is the number of
          detected object instances in the image and Hmask, Wmask are the mask width and mask
          height of the predicted mask (e.g., Hmask = Wmask = 28). Values are in [0, 1].
      boxes (Boxes or Tensor): A Boxes of length Bimg or Tensor of shape (Bimg, 4).
          boxes[i] and masks[i] correspond to the same object instance.
      image_shape (tuple): height, width
      threshold (float): A threshold in [0, 1] for converting the (soft) masks to
          binary masks.

  Returns:
      img_masks (Tensor): A tensor of shape (Bimg, Himage, Wimage), where Bimg is the
      number of detected object instances and Himage, Wimage are the image width
      and height. img_masks[i] is a binary mask for object instance i.
  """

  assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
  N = len(masks)
  if N == 0:
    return masks.new_empty((0,) + image_shape, dtype=torch.uint8)
  if not isinstance(boxes, torch.Tensor):
    boxes = boxes.tensor
  device = boxes.device
  assert len(boxes) == N, boxes.shape

  img_h, img_w = image_shape

  # The actual implementation split the input into chunks,
  # and paste them chunk by chunk.
  if device.type == "cpu" or False:#torch.jit.is_scripting():
    # CPU is most efficient when they are pasted one by one with skip_empty=True
    # so that it performs minimal number of operations.
    num_chunks = N
  else:
    # GPU benefits from parallelism for larger chunks, but may have memory issue
    # int(img_h) because shape may be tensors in tracing
    num_chunks = int(np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
    assert (
        num_chunks <= N
    ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
  chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

  img_masks = torch.zeros(
      N, img_h, img_w, device=device, dtype=torch.bool if threshold >= 0 else torch.float32
  )
  for inds in chunks:
    masks_chunk, spatial_inds = _do_paste_mask(
      masks[inds, None, :, :], boxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
    )

    if threshold >= 0:
      masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
    else:
        # for visualization and debugging
      #masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)
      masks_chunk = (masks_chunk).to(dtype=torch.float32)

    if False:#torch.jit.is_scripting():  # Scripting does not use the optimized codepath
      img_masks[inds] = masks_chunk
    else:
      img_masks[(inds,) + spatial_inds] = masks_chunk
  return img_masks
