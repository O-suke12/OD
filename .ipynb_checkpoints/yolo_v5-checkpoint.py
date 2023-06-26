import torch
from dataset import VOCDataset
import torchvision.transforms as T

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

img_dir = "dataset/images"
label_dir = "dataset/labels"
sample_csv = "dataset/8examples.csv"
test_csv = "dataset/test.csv"
preprocess = T.Compose([
    T.Resize((448,448))
])

VOC = VOCDataset(sample_csv, img_dir, label_dir, preprocess)

results = model(VOC.__getitem__(0)[0])
print(results)
# results.print()
# results.save()  
