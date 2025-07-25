import matplotlib.pyplot as plt
import torch
from data import get_datasets
from augmentation import train_transforms, test_transforms


# Берём 7-е изображение из массива NumPy (формат HWC, uint8)


if __name__ == "__main__":
    train_dataset, _ = get_datasets("../data")
    image_np, _ = train_dataset[20]

    plt.figure(figsize=(10, 10))
    for i in range(9):
        # Применяем аугментацию к NumPy-изображению (HWC, uint8)
        augmented_image = train_transforms(image_np)
        # augmented_image — tensor CxHxW в диапазоне [0,1]
        # Переводим в формат HWC для plt.imshow
        img_to_show = augmented_image.permute(1, 2, 0).numpy()

        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(img_to_show)
        plt.axis("off")
    plt.show()
