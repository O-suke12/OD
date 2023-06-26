import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from PIL import Image
import pandas as pd

class VOCDataset(torch.utils.data.Dataset):
  def __init__(self, csv_file, img_dir, label_dir, transform, split_num=7, box_num=2, classes=20):
    self.annotation = pd.read_csv(csv_file)
    self.img_dir = img_dir
    self.label_dir = label_dir
    self.transform = transform
    self.split_num = split_num
    self.box_num = box_num
    self.classes = classes
  
  def __len__(self):
    return len(self.annotation)

  def __getitem__(self, index):
    label_path = os.path.join(self.label_dir, self.annotation.iloc[index,1])
    boxes = []
    with open(label_path) as f:
      for label in f.readlines():
        class_label, x, y, width, height = [
            float(x) if float(x) != int(float(x)) else int(x)
            for x in label.replace("\n","").split()
        ]
        boxes.append([class_label, x, y, width, height])

    img_path = os.path.join(self.img_dir, self.annotation.iloc[index, 0])
    image = Image.open(img_path)
    boxes = torch.tensor(boxes)

    if self.transform:
      image = self.transform(image)

    label_matrix = torch.zeros((self.split_num, self.split_num, self.classes+5*self.box_num))
    for box in boxes:

      class_label, x, y, width, height = box.tolist()
      class_label = int(class_label)
      i, j = int(self.split_num*y), int(self.split_num*x)      #which cell this box belong to
      x_cell, y_cell = self.split_num*x-j, self.split_num*y-i   #position in the cell
      width_cell, height_cell = (
      width*self.split_num,
      height*self.split_num
      )
      if label_matrix[i,j,20] == 0:
        label_matrix[i, j, 20] == 1
        box_coordinates = torch.tensor(
          [x_cell, y_cell, width_cell, height_cell]
        )
      label_matrix[i, j, 21:25] = box_coordinates
      label_matrix[i, j, class_label] = 1
      
    return image, label_matrix