import torch
from tqdm import tqdm
from torchvision import transforms
from dataset import VOCDataset
from torch.utils.data import DataLoader
from model import YOLO_v1
from loss import Loss
from train import train

def main():

    img_dir = "dataset/images"
    label_dir = "dataset/labels"
    sample_csv = "dataset/8examples.csv"
    test_csv = "dataset/test.csv"
    preprocess = transforms.Compose([
        transforms.Resize((448,448)),
        transforms.ToTensor()
    ])
    VOC = VOCDataset(sample_csv, img_dir, label_dir, preprocess)
    train_loader = DataLoader(VOC, batch_size=8, shuffle=True, num_workers=2)
    VOC_test = VOCDataset(test_csv, img_dir, label_dir, preprocess)
    test_loader = DataLoader(VOC_test, batch_size=8, shuffle=True, num_workers=2)

    epochs = 10
    model = YOLO_v1()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=0)
    loss_fn = Loss()
    train(epochs, model, optimizer, loss_fn, train_loader)

if __name__ == "__main__":
    main()

