import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights
from utilities.save import to_json
import torchvision
from utilities.save_image import save_mask_comparisons

class VGGUNet(nn.Module):
    def __init__(self, num_classes=35, input_size=(224, 224)):
        super(VGGUNet, self).__init__()

        vgg = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())

        # Используем только первые 23 слоя из VGG (включая 5 блоков + MaxPool)
        self.enc1 = nn.Sequential(*features[:4])  # Conv1_1, Conv1_2
        self.pool1 = features[4]

        self.enc2 = nn.Sequential(*features[5:9])  # Conv2_1, Conv2_2
        self.pool2 = features[9]
        self.enc3 = nn.Sequential(*features[10:16])  # Conv3_1,2,3
        self.pool3 = features[16]
        self.enc4 = nn.Sequential(*features[17:23])  # Conv4_1,2,3
        self.pool4 = features[23]

        # Заморозим слои VGG чтобы сохранить память об ImageNet и ускорить обучение
        for param in vgg.parameters():
            param.requires_grad = False
        # Но разрешим дообучение последних слоёв VGG16
        for param in self.enc4.parameters():
            param.requires_grad = True
        for param in self.enc3.parameters():
            param.requires_grad = True

        # Боттлнек
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Декодер
        self.up4 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        self.dec4 = self.double_conv(1024 + 512, 512)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = self.double_conv(512 + 256, 256)
        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec2 = self.double_conv(256 + 128, 128)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.double_conv(128 + 64, 64)
        self.up_final = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=1)
        self.final = self.double_conv(64, num_classes)
        self.input_size = input_size  # размер для VGG
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def preprocess(self, x):
        if x.device != self.mean.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)

        x = (x - self.mean) / self.std
        return x

    def forward(self, x):
        original_size = x.shape[2:]  # [H, W]

        x = self.preprocess(x)
        c1 = self.enc1(x)
        p1 = self.pool1(c1)
        c2 = self.enc2(p1)
        p2 = self.pool2(c2)
        c3 = self.enc3(p2)
        p3 = self.pool3(c3)
        c4 = self.enc4(p3)
        p4 = self.pool4(c4)
        b = self.bottleneck(p4)
        u4 = self.up4(b)
        u4 = torch.cat([u4, c4], dim=1)
        d4 = self.dec4(u4)
        u3 = self.up3(d4)
        u3 = torch.cat([u3, c3], dim=1)
        d3 = self.dec3(u3)
        u2 = self.up2(d3)
        u2 = torch.cat([u2, c2], dim=1)
        d2 = self.dec2(u2)
        u1 = self.up1(d2)
        u1 = torch.cat([u1, c1], dim=1)
        d1 = self.dec1(u1)
        out = self.up_final(d1)
        out = nn.functional.relu(out)
        out = F.interpolate(out, size=original_size, mode='bilinear',
                            align_corners=False)
        out = self.final(out)

        return out

def train_vgg(model, train_dataset, val_dataset,  batch_size, num_epochs, lr, num_classes):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"epoch # [{epoch}]")

        save_mask_comparisons(
            epoch=epoch,
            model=model,
            dataloader=val_loader,  # Можно использовать и train_loader
            device=device,
            save_dir="result_image/unetvgg",
            num_images=min(8, batch_size)  # Сохраняем 8 изображений или весь батч, если меньше
        )

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device).long()
            if masks.ndim == 4:  # [B, 1, H, W]
                masks = masks.squeeze(1)  # [B, H, W]
            optimizer.zero_grad()
            outputs = model(images)
            masks = masks.squeeze(1)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Step [{batch_idx}/{len(train_loader)}] :: loss {loss.item()}")

        avg_train_loss = running_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device).long()
                if masks.ndim == 4:  # [B, 1, H, W]
                    masks = masks.squeeze(1)  # [B, H, W]
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss: .4f}")

    to_json("metrics/unetvgg/train", "train_losses", train_loss_history)
    to_json("metrics/unetvgg/validation", "val_losses", val_loss_history)


    torch.save(model.state_dict(), f"model/model_vgg_weights_new.pth") # можно включить сохранение модели на диск