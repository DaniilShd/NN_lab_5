import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader



class UNet(nn.Module):
    def __init__(self, num_classes=35):
        super(UNet, self).__init__()
        # Encoder
        self.conv1 = self.double_conv(3, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = self.double_conv(32, 64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2)
        # Bottleneck
        self.conv3 = self.double_conv(64, 128, kernel_size=7)
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=True)
        self.conv4 = self.double_conv(128 + 64, 64, kernel_size=5)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=True)
        self.conv5 = self.double_conv(64 + 32, 32, kernel_size=3)
        # Output
        self.final = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)

    def double_conv(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x) # [B, 32, H, W]
        p1 = self.pool1(c1) # [B, 32, H/2, W/2]
        c2 = self.conv2(p1) # [B, 64, H/2, W/2]
        p2 = self.pool2(c2) # [B, 64, H/4, W/4]

        # Bottleneck
        c3 = self.conv3(p2) # [B, 128, H/4, W/4]

        # Decoder
        up1 = self.up1(c3) # [B, 128, H/2, W/2]
        merge1 = torch.cat([c2, up1], dim=1) # [B, 192, H/2, W/2]
        c4 = self.conv4(merge1) # [B, 64, H/2, W/2]

        up2 = self.up2(c4) # [B, 64, H, W]
        merge2 = torch.cat([c1, up2], dim=1) # [B, 96, H, W]
        c5 = self.conv5(merge2)  # [B, 32, H, W]

        out = self.final(c5)  # [B, num_classes, H, W]
        return out

def train(model, train_dataset, val_dataset,  batch_size, num_epochs, lr, num_classes):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

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
            print(f"Step [{batch_idx}/{len(train_loader)}] :: loss{loss.item()}")

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
    torch.save(model.state_dict(), f"unet_epoch_{epoch+1}.pth") # можно включить сохранение модели на диск
