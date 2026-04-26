# eval_confusion.py
# 用 seed=42 复现验证集，加载 best_model_ocr.pth，产出混淆矩阵 + 每类指标对比图
# 独立可运行：python eval_confusion.py
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ──────────── 中文字体：SimHei → Microsoft YaHei fallback ────────────
def _setup_chinese_font():
    candidates = ["SimHei", "Microsoft YaHei"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)
    if chosen is None:
        print("[警告] 未找到 SimHei / Microsoft YaHei，中文可能显示为方块")
        chosen = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [chosen, "Microsoft YaHei", "SimHei", "sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    print(f"[字体] 使用: {chosen}")

_setup_chinese_font()

# ──────────── 路径 / 常量 ────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT, "dataset")
LABEL_NAMES = ["尴尬", "开心", "悲伤", "惊讶", "愤怒"]
NUM_CLASSES = len(LABEL_NAMES)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42

# ──────────── 1. 重建 index → filepath 映射并强校验 ────────────
def reconstruct_paths():
    paths, lbls = [], []
    for label_idx, label in enumerate(LABEL_NAMES):
        folder = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"缺少类别目录: {folder}")
        for fname in [f for f in os.listdir(folder) if f.endswith(".jpg")]:
            paths.append(os.path.join(folder, fname))
            lbls.append(label_idx)
    return np.array(paths), np.array(lbls, dtype=np.int64)

paths, reconstructed_labels = reconstruct_paths()
y = np.load(os.path.join(ROOT, "labels.npy"))

if len(paths) != len(y):
    sys.exit(f"[错误] 重建数量 {len(paths)} 与 labels.npy {len(y)} 不一致")
if not np.array_equal(reconstructed_labels, y):
    diff = int((reconstructed_labels != y).sum())
    sys.exit(f"[错误] 重建 labels 与 labels.npy 有 {diff} 处不符，os.listdir 顺序可能变了")
print(f"[映射] 重建 {len(paths)} 条 index→filepath，逐元素校验通过")

# ──────────── 2. 用 seed=42 复现训练时的 train/val 划分 ────────────
img = np.load(os.path.join(ROOT, "features_ocr_img.npy"))
txt = np.load(os.path.join(ROOT, "features_ocr_txt.npy"))

X_img    = torch.FloatTensor(img)
X_txt    = torch.FloatTensor(txt)
y_tensor = torch.LongTensor(y)

dataset    = TensorDataset(X_img, X_txt, y_tensor)
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size

split_gen = torch.Generator().manual_seed(SEED)
train_set, val_set = random_split(dataset, [train_size, val_size], generator=split_gen)

val_indices = np.array(val_set.indices, dtype=np.int64)
np.save(os.path.join(ROOT, "val_indices_seed42.npy"), val_indices)
print(f"[划分] 训练 {train_size} / 验证 {val_size}，val_indices_seed42.npy 已保存")

val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

# ──────────── 3. 模型定义 + 加载权重（与 train/inference 严格一致） ────────────
class CrossModalMLP(nn.Module):
    def __init__(self, dim=512, num_classes=5):
        super().__init__()
        self.img_branch = nn.Sequential(
            nn.Linear(dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.txt_branch = nn.Sequential(
            nn.Linear(dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )
    def forward(self, img, txt):
        return self.classifier(torch.cat([self.img_branch(img), self.txt_branch(txt)], dim=1))

model = CrossModalMLP().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(ROOT, "best_model_ocr.pth"), map_location=DEVICE))
model.eval()

# ──────────── 4. 纯 argmax 推理（不加 0.5 阈值，保证和训练时指标口径一致） ────────────
all_true, all_pred = [], []
with torch.no_grad():
    for img_b, txt_b, yb in val_loader:
        logits = model(img_b.to(DEVICE), txt_b.to(DEVICE))
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(yb.numpy().tolist())

all_true = np.array(all_true)
all_pred = np.array(all_pred)
acc      = (all_true == all_pred).mean()
err_n    = int((all_true != all_pred).sum())

print(f"\n[总体] 验证集 {len(all_true)} 张 | 准确率 {acc:.4f} | 错误 {err_n}")
print("\n[classification_report]")
print(classification_report(all_true, all_pred, target_names=LABEL_NAMES, digits=4))

# ──────────── 5. 混淆矩阵：绝对计数 + 行归一化并排 ────────────
cm      = confusion_matrix(all_true, all_pred, labels=list(range(NUM_CLASSES)))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, mat, title, fmt, cmap in [
    (axes[0], cm,      "混淆矩阵（绝对计数）",  "d",   "Blues"),
    (axes[1], cm_norm, "混淆矩阵（行归一化）",  ".2f", "Oranges"),
]:
    im = ax.imshow(mat, cmap=cmap)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(LABEL_NAMES); ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    thresh = mat.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = mat[i, j]
            ax.text(j, i, format(v, fmt), ha="center", va="center",
                    color="white" if v > thresh else "black", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(f"验证集 (seed=42, n={len(all_true)}) 总准确率 {acc:.4f}", fontsize=15)
plt.tight_layout()
cm_path = os.path.join(ROOT, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[保存] {cm_path}")

# ──────────── 6. 每类 precision/recall/F1 对比柱状图 ────────────
prec, rec, f1, support = precision_recall_fscore_support(
    all_true, all_pred, labels=list(range(NUM_CLASSES)), zero_division=0
)

x = np.arange(NUM_CLASSES)
w = 0.25
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - w, prec, w, label="Precision", color="#4C72B0")
ax.bar(x,     rec,  w, label="Recall",    color="#DD8452")
ax.bar(x + w, f1,   w, label="F1",        color="#55A868")

for i in range(NUM_CLASSES):
    ax.text(x[i] - w, prec[i] + 0.01, f"{prec[i]:.2f}", ha="center", fontsize=9)
    ax.text(x[i],     rec[i]  + 0.01, f"{rec[i]:.2f}",  ha="center", fontsize=9)
    ax.text(x[i] + w, f1[i]   + 0.01, f"{f1[i]:.2f}",   ha="center", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([f"{n}\n(n={s})" for n, s in zip(LABEL_NAMES, support)])
ax.set_ylim(0, 1.1)
ax.set_ylabel("指标值")
ax.set_title(f"各类别 Precision / Recall / F1 对比 (seed=42 验证集)")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
pc_path = os.path.join(ROOT, "per_class_metrics.png")
plt.savefig(pc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[保存] {pc_path}")

# ──────────── 7. 错误分布（方便 Step 3 参考） ────────────
err_mask = all_true != all_pred
err_val_positions = np.where(err_mask)[0]   # 在 val_loader 顺序中的索引
err_dataset_indices = val_indices[err_val_positions]  # 在全量 dataset 中的索引
np.save(os.path.join(ROOT, "val_error_dataset_indices.npy"), err_dataset_indices)
print(f"[保存] val_error_dataset_indices.npy  共 {len(err_dataset_indices)} 个错误样本 dataset 索引")

print("\n[Step 2 完成]")
