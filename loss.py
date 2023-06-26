import torch
import torch.nn as nn
from utils import IoU

class Loss(nn.Module):
  def __init__(self, split_num=7, box=2, classes=20):
    super().__init__()
    self.MSE = nn.MSELoss(reduction="sum")
    self.split_num = split_num
    self.box = box
    self.classes = classes
    self.lambda_noobj = 0.5
    self.lambda_coord = 5

  def forward(self, predictions, target):
    predictions = predictions.reshape(-1, self.split_num, self.split_num, self.classes+self.box*5)
    ioub1 = IoU(predictions[...,21:25], target[...,21:25], "center") #21~24: center_x, center_y, width, height 25: objectness confidence
    ioub2 = IoU(predictions[...,26:30], target[...,26:30], "center")
    ious = torch.cat([ioub1.unsqueeze(0), ioub2.unsqueeze(0)], dim=0)
    iou_max, best_box = torch.max(ious, dim=0) #best_box=[0,1] which box is the best (argmax)
    exist_box = target[..., 20].unsqueeze(3) #Iobj_i

    #box regression loss
    box_predictions = exist_box*(best_box*predictions[...,26:30]+(1-best_box)*predictions[...,21:25])
    box_targets = exist_box*target[...,21:25]  #target[x,y,width,height]
    box_predictions[...,2:4] = torch.sign(box_predictions[...,2:4])*torch.sqrt(torch.abs(box_predictions[...,2:4]+1e-6)) #sqrt(prediction[width, height])
    box_targets[...,2:4] = torch.sqrt(box_targets[...,2:4])
    box_loss = self.MSE(
        torch.flatten(box_predictions, end_dim=-2),
        torch.flatten(box_targets, end_dim=-2)
    )

    #objectness confidence loss
    #confidence
    pred_box = (
        best_box*predictions[...,25:26]+(1-best_box)*predictions[...,20:21]
    )
    object_loss = self.MSE(
        torch.flatten(exist_box*pred_box),
        torch.flatten(exist_box*target[...,20:21])
    )

    #no_objectness confidence loss
    no_object_loss = self.MSE(
        torch.flatten((1-exist_box)*predictions[...,20:21], start_dim=1),
        torch.flatten((1-exist_box)*target[...,20:21], start_dim=1)
    )

    #class loss
    class_loss = self.MSE(
        torch.flatten(exist_box*predictions[...,:20],end_dim=-2),
        torch.flatten(exist_box*target[...,:20], end_dim=-2)
    )

    loss = self.lambda_coord*box_loss+object_loss+self.lambda_noobj*no_object_loss+class_loss
    return loss


# sample_image, sample_label = VOC.__getitem__(0)
# y_pred = model(sample_image.unsqueeze(0))
# loss = loss_fn(y_pred, sample_label.unsqueeze(0))
# loss