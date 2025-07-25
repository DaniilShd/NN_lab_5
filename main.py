import os
from u_net import UNet, train_unet
from u_net_vgg import train_vgg, VGGUNet
from dataset import download_data, SimpleSegmentationDataset
import matplotlib.pyplot as plt
import torch
import random


def output_result(model, val_dataset):
    # Количество изображений для визуализации
    num_images = 6
    random_indices = random.sample(range(len(val_dataset)), num_images)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Подготовка графика
    fig, axes = plt.subplots(3, num_images, figsize=(20, 7))
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(random_indices):
            image, true_mask = val_dataset[idx]  # image: [3, H, W], mask: [H,W]
            image_tensor = image.unsqueeze(0).to(device)  # [1, 3, H, W]
            # Запуск модели на тестовых изображениях
            output = model(image_tensor)  # [1, 35, H, W]
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # [H, W]
            # Отображение входного изображения
            axes[0, i].imshow(image.permute(1, 2, 0).cpu().numpy())
            axes[0, i].set_title(f"Изображение {idx}")
            axes[0, i].axis('off')
            # Истинная маска
            axes[1, i].imshow(true_mask.squeeze().cpu().numpy())
            axes[1, i].set_title(f"Маска {idx}")
            axes[1, i].axis('off')
            # Предсказанная маска
            axes[2, i].imshow(pred_mask)
            axes[2, i].set_title(f"Результат {idx}")
            axes[2, i].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Загрузим данные и соберем их в датасеты
    X_train = download_data(r"data/Resized images/Images/Train/")
    Y_train = download_data(r"data/Resized images/Masks/Train/")

    train_dataset = SimpleSegmentationDataset(X_train, Y_train)

    X_test = download_data(r"data/Resized images/Images/Test/")
    Y_test = download_data(r"data/Resized images/Masks/Test/")

    val_dataset = SimpleSegmentationDataset(X_test, Y_test)

    if not os.path.exists("model/model_weights.pth"):
        # Гиперпараметры
        batch_size = 8
        num_epochs = 50
        lr = 1e-4
        num_classes = 35

        model = UNet(num_classes=num_classes)
        train_unet(model, train_dataset, val_dataset,  batch_size, num_epochs, lr, num_classes)
    else:
        print("Модель уже обучена")


    if not os.path.exists("model/model_vgg_weights.pth"):
        # Гиперпараметры
        batch_size = 4
        num_epochs = 40
        lr = 1e-4
        num_classes = 35

        model = VGGUNet(num_classes=num_classes)
        train_vgg(model, train_dataset, val_dataset,  batch_size, num_epochs, lr, num_classes)
    else:
        print("Модель уже обучена")

    model = UNet(num_classes=35)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('model/model_weights.pth', map_location=torch.device('cpu')))
    model.to(device)

    output_result(model, val_dataset)



