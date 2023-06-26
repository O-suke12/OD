import torch
import torch.nn as nn
from torch.nn.modules import Dropout1d
from torch.nn.modules.pooling import MaxPool2d
class YOLO_v1(nn.Module):
  def __init__(self) -> None:
    super().__init__()

    self.conv_block1 = nn.Sequential(
      nn.Conv2d(in_channels=3 ,out_channels=64, kernel_size=3, stride=2, padding=3),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(0.1),
      nn.MaxPool2d(2,2),
      nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
      nn.BatchNorm2d(192),
      nn.LeakyReLU(0.1),
      nn.MaxPool2d(2,2),
      nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, padding=0),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,  padding=1),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1,  padding=0),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,  padding=1),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.1),
      nn.MaxPool2d(2,2),
    )

    self.conv_block2 = nn.Sequential(
      nn.Conv2d(in_channels=512, out_channels=216, kernel_size=1, padding=1),
      nn.BatchNorm2d(216),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=216, out_channels=512, kernel_size=3),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.1),
    )

    self.conv_block3 = nn.Sequential(
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1,  padding=0),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3,  padding=1),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.1),
      nn.MaxPool2d(2,2),
    )

    self.conv_block4 = nn.Sequential(
      nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.1),
    )

    self.conv_block5 = nn.Sequential(
      nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1), 
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.1),
    )

    self.fully_connective_layer = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_features=7*7*1024, out_features=496),  #Original out features = 4096
      nn.Dropout(0.5),
      nn.LeakyReLU(0.1),
      nn.Linear(in_features=496, out_features=30*7*7)   #(Split_size, Split_size, 30)  Generally each cell have (bounding_box*5, class_possibilities=20)
    )
    
  
  def forward(self, X):
    X = self.conv_block1(X)
    X = self.conv_block2(X)
    X = self.conv_block2(X)
    X = self.conv_block2(X)
    X = self.conv_block2(X)
    X = self.conv_block3(X)
    X = self.conv_block4(X)
    X = self.conv_block4(X)
    X = self.conv_block5(X)
    X = self.fully_connective_layer(X)
    return X