# train_ocr_crossmodal.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report

img = np.load("features_ocr_img.npy")
txt = np.load("features_ocr_txt.npy")
y   = np.load("labels.npy")

print(f"图像特征: {img.shape} | 文本特征: {txt.shape}")

X_img    = torch.FloatTensor(img)
X_txt    = torch.FloatTensor(txt)
y_tensor = torch.LongTensor(y)

dataset    = TensorDataset(X_img, X_txt, y_tensor)
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=64)

class CrossModalMLP(nn.Module):
    def __init__(self, dim=512, num_classes=5):
        super().__init__()
        self.img_branch = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.txt_branch = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, txt):
        return self.classifier(
            torch.cat([self.img_branch(img), self.txt_branch(txt)], dim=1)
        )

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

model         = CrossModalMLP().to(device)
class_weights = torch.FloatTensor([1.5, 1.0, 1.0, 1.0, 1.0]).to(device)
criterion     = nn.CrossEntropyLoss(weight=class_weights)
optimizer     = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

best_acc   = 0
no_improve = 0
patience   = 30

for epoch in range(150):
    model.train()
    for img_b, txt_b, yb in train_loader:
        img_b, txt_b, yb = img_b.to(device), txt_b.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(img_b, txt_b), yb)
        loss.backward()
        optimizer.step()
    scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for img_b, txt_b, yb in val_loader:
            img_b, txt_b, yb = img_b.to(device), txt_b.to(device), yb.to(device)
            preds = model(img_b, txt_b).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += len(yb)

    acc = correct / total
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:02d} | 验证准确率: {acc:.4f}")

    if acc > best_acc:
        best_acc   = acc
        no_improve = 0
        torch.save(model.state_dict(), "best_model_ocr.pth")
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

print(f"\n✅ 训练完成！最佳准确率: {best_acc:.4f}")

label_names = open("label_names.txt", encoding="utf-8").read().splitlines()
model.load_state_dict(torch.load("best_model_ocr.pth"))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for img_b, txt_b, yb in val_loader:
        preds = model(img_b.to(device), txt_b.to(device)).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_true.extend(yb.tolist())

print("\n📊 各类别详细报告:")
print(classification_report(all_true, all_preds, target_names=label_names))