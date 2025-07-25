import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
from skimage import io
import numpy as np
import random
import os
from torchvision import transforms

def download_data(path):
    data = []
    for path_image in sorted(os.listdir(path=path)):
        image = Image.open(path + path_image)
        data.append(np.array(image))
    return data

# Так же определим класс датасета, подходящий для наших данных
class SimpleSegmentationDataset():
    def __init__(self, data, masks):
        self.data = data
        self.y = masks
        self.transform = transforms.Compose(
            [transforms.ToTensor() ] # <-- это приводит HWC → CHW, т.е меняет местами порядок каналов в тензоре изображения
    )
# стандартный порядок: ширина, высота, цветовые каналы
# но пайторч требует: цветовые каналы, ширина, высота
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = self.transform(image)
        target = self.y[idx]
        target = self.transform(target)
        return image, target



if __name__ == "__main__":
    # Загрузим данные и соберем их в датасеты
    X_train = download_data(r"./drive/My Drive/GTA5/Resized images/Images/Train/")
    Y_train = download_data(r"./drive/My Drive/GTA5/Resized images/Masks/Train/")
    train_dataset = SimpleSegmentationDataset(X_train, Y_train)
    X_test = download_data(r"./drive/My Drive/GTA5/Resized images/Images/Test/")
    Y_test = download_data(r"./drive/My Drive/GTA5/Resized images/Masks/Test/")
    val_dataset = SimpleSegmentationDataset(X_test, Y_test)

    num_images = 5
    random_indices = random.sample(range(len(X_train)), num_images)
    plt.figure(figsize=(17, 5))
    for i, idx in enumerate(random_indices):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X_train[idx])
        plt.title(f"Изображение {idx}")
        plt.axis('off')
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(Y_train[idx])
        plt.title(f"Маска {idx}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()