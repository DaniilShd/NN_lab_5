import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np


# def save_mask_comparisons(epoch, model, dataloader, device, save_dir="mask_comparisons", num_images=8):
#     """
#     Сохраняет сравнение оригинальных изображений и предсказанных масок через каждые 5 эпох.
#     Автоматически определяет тип модели (UNet или VGG-UNet) для сохранения в разные папки.
#
#     Args:
#         epoch (int): Текущая эпоха
#         model (nn.Module): Модель для предсказания
#         dataloader (DataLoader): Загрузчик данных (val_loader или train_loader)
#         device (torch.device): Устройство (cpu/cuda)
#         save_dir (str): Базовая директория для сохранения
#         num_images (int): Количество изображений для сохранения (<= batch_size)
#     """
#     if epoch % 5 != 0:  # Сохраняем только каждые 5 эпох
#         return
#
#     # Определяем тип модели для структуры папок
#     model_type = "unet" if model.__class__.__name__ == "UNet" else "vgg_unet"
#     model_save_dir = os.path.join(save_dir, model_type)
#     os.makedirs(model_save_dir, exist_ok=True)
#
#     model.eval()
#     with torch.no_grad():
#         try:
#             images, true_masks = next(iter(dataloader))
#             images = images.to(device)
#
#             # Получаем предсказания
#             pred_masks = model(images)
#             pred_classes = pred_masks.argmax(dim=1).cpu().float()
#
#             # Проверка данных
#             print(f"Predicted masks stats - Min: {pred_classes.min()}, Max: {pred_classes.max()}")
#
#             # Визуализация
#             fig, axes = plt.subplots(2, num_images, figsize=(20, 5))
#             if num_images == 1:
#                 axes = axes.reshape(2, 1)
#
#             for i in range(num_images):
#                 # Оригинальное изображение
#                 img = images[i].cpu().permute(1, 2, 0)
#                 if img.shape[2] == 3:  # RGB
#                     axes[0, i].imshow(img)
#                 else:  # Градации серого
#                     axes[0, i].imshow(img.mean(dim=2), cmap='gray')
#                 axes[0, i].axis('off')
#
#                 # Предсказанная маска
#                 mask = pred_classes[i]
#                 if mask.max() == mask.min():  # Все пиксели одного класса
#                     mask = (mask - mask.min()) / 1.0  # Избегаем деления на 0
#                 else:
#                     mask = (mask - mask.min()) / (mask.max() - mask.min())
#
#                 axes[1, i].imshow(mask, cmap='viridis', vmin=0, vmax=1)
#                 axes[1, i].axis('off')
#
#             plt.suptitle(f'Epoch {epoch}', y=1.05)
#             plt.tight_layout()
#             plt.savefig(os.path.join(model_save_dir, f'epoch_{epoch:03d}.png'),
#                         bbox_inches='tight', dpi=150)
#             plt.close()
#
#             print(f"Successfully saved visualizations for epoch {epoch}")
#
#         except Exception as e:
#             print(f"Error saving masks for epoch {epoch}: {str(e)}")


def save_mask_comparisons(epoch, model, dataloader, device, save_dir="result_image", num_images=8):
    """Сохраняет сравнение предсказанных и истинных масок каждые 5 эпох.

    Args:
        epoch (int): Текущая эпоха
        model (nn.Module): Модель для предсказания
        dataloader (DataLoader): Загрузчик данных (валидационный или тестовый)
        device (torch.device): Устройство (CPU или GPU)
        save_dir (str): Директория для сохранения изображений
        num_images (int): Количество сохраняемых изображений
    """

    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        # Получаем один батч данных
        data_iter = iter(dataloader)
        images, true_masks = next(data_iter)
        images, true_masks = next(data_iter)
        images = images.to(device)
        true_masks = true_masks.to(device)

        # Ограничиваем количество изображений
        if len(images) > num_images:
            images = images[:num_images]
            true_masks = true_masks[:num_images]

        # Получаем предсказания
        preds = model(images)
        pred_masks = torch.argmax(preds, dim=1)

        # Переводим тензоры в numpy и меняем порядок каналов для отображения
        images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
        true_masks_np = true_masks.squeeze(1).cpu().numpy()
        pred_masks_np = pred_masks.cpu().numpy()

        # Создаем сетку для отображения
        fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

        for i in range(len(images)):
            # Оригинальное изображение
            img = (images_np[i] * 255).astype(np.uint8)
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Input Image {i + 1}")
            axes[i, 0].axis('off')

            # Предсказанная маска
            axes[i, 1].imshow(pred_masks_np[i], cmap='jet')
            axes[i, 1].set_title(f"Predicted Mask {i + 1}")
            axes[i, 1].axis('off')

            # Истинная маска
            axes[i, 2].imshow(true_masks_np[i], cmap='jet')
            axes[i, 2].set_title(f"True Mask {i + 1}")
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_comparison.png"))
        plt.close()