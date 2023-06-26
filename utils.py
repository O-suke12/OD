import torch
import numpy as np
import matplotlib.pyplot as plt

# return IOU= area of intersection / area of union
def IoU (box_preds, box_labels, label_type):
  if label_type == "center":
    box1_left = box_preds[...,0:1]-box_preds[...,2:3]/2
    box1_up = box_preds[...,1:2]-box_preds[...,3:4]/2
    box1_right = box_preds[...,0:1]+box_preds[...,2:3]/2
    box1_low = box_preds[...,1:2]+box_preds[...,3:4]/2
    box2_left = box_labels[...,0:1]-box_labels[...,2:3]/2
    box2_up = box_labels[...,1:2]-box_labels[...,3:4]/2
    box2_right = box_labels[...,0:1]+box_labels[...,2:3]/2
    box2_low = box_labels[...,1:2]+box_labels[...,3:4]/2

  elif label_type == "corner":
    #[xmin, ymin, xmax, ymax]
    box1_left = box_preds[...,0:1]
    box1_up = box_preds[...,1:2]
    box1_right = box_preds[...,2:3]
    box1_low = box_preds[...,3:4]
    box2_left = box_labels[...,0:1]
    box2_up = box_labels[...,1:2]
    box2_right = box_labels[...,2:3]
    box2_low = box_labels[...,3:4]

  else:
    return False
  
  box_right = torch.min(box1_right, box2_right)
  box_left = torch.max(box1_left, box2_left)
  box_low = torch.min(box1_low, box2_low)
  box_up = torch.max(box1_up, box2_up)
  intersection_area = (box_right-box_left).clamp(0)*(box_low-box_up).clamp(0)
  union_area = abs((box1_right-box1_left)*(box1_low-box1_up))+abs((box2_right-box2_left)*(box2_low-box2_up))-intersection_area+1e-6
  return intersection_area/union_area


def non_max_suppresion(predictions, thresold, prob_thresold, IoU_thresold):
  assert type(bboxes) == list
  bboxes = [box for box in bboxes if box[1]>prob_thresold]
  bboxes = sorted(bboxes, key = lambda x:x[1], reverse=True)
  bboxes_after_nms = []

  while bboxes:
    chosen_box = bboxes.pop(0)
    #if chosen box class is not the same the box class in the bbox , we don't have to compare
    bboxes = [box for box in bboxes if box[0]!=chosen_box[0] or IoU(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), "corner")< IoU_thresold]
    bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms
     