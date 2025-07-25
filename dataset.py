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
class SimpleSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, data, masks):
        self.data = data
        self.masks = masks
        self.image_transform = transforms.Compose([
            transforms.ToTensor()  # Конвертирует в тензор и нормализует в [0,1] для изображений
        ])
        self.mask_transform = transforms.Compose([
            transforms.PILToTensor(),  # Конвертирует в тензор без нормализации
            transforms.Lambda(lambda x: x.squeeze().long())  # Удаляет размерность каналов если нужно
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.fromarray(self.data[idx]) if isinstance(self.data[idx], np.ndarray) else self.data[idx]
        mask = Image.fromarray(self.masks[idx]) if isinstance(self.masks[idx], np.ndarray) else self.masks[idx]

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        return image, mask



if __name__ == "__main__":
    # Загрузим данные и соберем их в датасеты
    X_train = download_data(r"data/Resized images/Images/Train/")
    Y_train = download_data(r"data/Resized images/Masks/Train/")

    train_dataset = SimpleSegmentationDataset(X_train, Y_train)

    X_test = download_data(r"data/Resized images/Images/Test/")
    Y_test = download_data(r"data/Resized images/Masks/Test/")

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

    # Вывод примеров из валидационного (тестового) набора
    random_indices_val = random.sample(range(len(X_test)), num_images)
    plt.figure(figsize=(17, 5))
    for i, idx in enumerate(random_indices_val):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X_test[idx])
        plt.title(f"Val Image {idx}")
        plt.axis('off')
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(Y_test[idx])
        plt.title(f"Val Mask {idx}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()