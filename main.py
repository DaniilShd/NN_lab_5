import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
from skimage import io
import numpy as np
import random
import os
from torchvision import transforms
from u_net import UNet, train
from dataset import download_data, SimpleSegmentationDataset

if __name__ == "__main__":
    # Загрузим данные и соберем их в датасеты
    X_train = download_data(r"data/Resized images/Images/Train/")
    Y_train = download_data(r"data/Resized images/Masks/Train/")

    train_dataset = SimpleSegmentationDataset(X_train, Y_train)

    X_test = download_data(r"data/Resized images/Images/Test/")
    Y_test = download_data(r"data/Resized images/Masks/Test/")

    val_dataset = SimpleSegmentationDataset(X_test, Y_test)

    # Гиперпараметры
    batch_size = 8
    num_epochs = 50
    lr = 1e-4
    num_classes = 35

    model = UNet()

    if not os.path.exists("model/model_weights.pth"):
        train(model, train_dataset, val_dataset,  batch_size, num_epochs, lr, num_classes)
    else:
        print("Модель уже обучена")



