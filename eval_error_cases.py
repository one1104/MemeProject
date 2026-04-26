# eval_error_cases.py
# 对 Step 2 标记的 375 张验证集错误样本重跑完整 raw 推理管线
# (CLIP 图像 + EasyOCR + CLIP 文本 + MLP)，按"真实类__误判为__预测类"分组，
# 每组保留置信度最高 3 张，复制到 error_cases/，并输出 _error_summary.csv (UTF-8 BOM)。
#
# 独立可运行：python eval_error_cases.py
import os
import sys
import csv
import shutil
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ──────────── 路径 / 常量 ────────────
ROOT          = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR   = os.path.join(ROOT, "dataset")
LABEL_NAMES   = ["尴尬", "开心", "悲伤", "惊讶", "愤怒"]
NUM_CLASSES   = len(LABEL_NAMES)
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
FALLBACK_TEXT = "一张表情包"   # 与 extract_features_ocr.py / inference.py 保持一致
TOP_K         = 3

ERROR_DIR = os.path.join(ROOT, "error_cases")
CSV_PATH  = os.path.join(ROOT, "_error_summary.csv")

# ──────────── 1. 重建 index → filepath（与 Step 2 同规则 + 强校验） ────────────
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
if not np.array_equal(reconstructed_labels, y):
    sys.exit("[错误] 重建 labels 与 labels.npy 不一致，os.listdir 顺序已变，不能继续")
print(f"[映射] {len(paths)} 条 index->filepath 校验通过")

# 加载 Step 2 产出的错误索引
err_idx_path = os.path.join(ROOT, "val_error_dataset_indices.npy")
if not os.path.exists(err_idx_path):
    sys.exit("[错误] 未找到 val_error_dataset_indices.npy，请先跑 Step 2")
err_indices = np.load(err_idx_path)
print(f"[输入] Step 2 标记错误样本 {len(err_indices)} 张，全部走完整 raw 管线")

# ──────────── 2. 加载 CLIP + MLP ────────────
import clip
print("[CLIP] 加载 ViT-B/32 ...")
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
MODEL.eval()

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

MLP = CrossModalMLP().to(DEVICE)
MLP.load_state_dict(torch.load(os.path.join(ROOT, "best_model_ocr.pth"), map_location=DEVICE))
MLP.eval()
print(f"[MLP] 权重已加载，设备={DEVICE}")

# ──────────── 3. 初始化 EasyOCR（提示首次加载耗时） ────────────
print("[EasyOCR] 正在加载 ch_sim+en 模型，首次加载约 1-2 分钟，请耐心等待（不是卡死）...")
t0 = time.time()
import easyocr
reader = easyocr.Reader(['ch_sim', 'en'], verbose=False)
print(f"[EasyOCR] 就绪，用时 {time.time()-t0:.1f}s")

# ──────────── 4. 对每个错误索引跑完整 raw 管线 ────────────
records = []   # 成功处理的记录（不论最终预测对错）
failed  = []   # 异常样本

t_start = time.time()
for i, idx in enumerate(err_indices):
    idx  = int(idx)
    path = str(paths[idx])
    true_label = int(y[idx])

    try:
        # 图像特征
        with Image.open(path) as im:
            img_input = PREPROCESS(im.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            img_feat = MODEL.encode_image(img_input)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # OCR → 文本
        img_np     = np.array(Image.open(path).convert("RGB"))
        ocr_result = reader.readtext(img_np, detail=0)
        ocr_text_raw = " ".join(ocr_result).strip()
        used_fallback = (ocr_text_raw == "")
        text_for_clip = FALLBACK_TEXT if used_fallback else ocr_text_raw

        # 文本特征
        txt_token = clip.tokenize([text_for_clip], truncate=True).to(DEVICE)
        with torch.no_grad():
            txt_feat = MODEL.encode_text(txt_token)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        # MLP → softmax
        with torch.no_grad():
            logits = MLP(img_feat, txt_feat)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_label = int(probs.argmax())
        confidence = float(probs[pred_label])

        records.append({
            "dataset_idx":   idx,
            "filepath":      path,
            "true_label":    true_label,
            "pred_label":    pred_label,
            "confidence":    confidence,
            "ocr_text":      ocr_text_raw,
            "used_fallback": used_fallback,
            "is_error":      pred_label != true_label,
        })

    except Exception as e:
        failed.append({"dataset_idx": idx, "filepath": path, "error": repr(e)})

    if (i + 1) % 25 == 0 or (i + 1) == len(err_indices):
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(err_indices) - i - 1)
        print(f"  进度 {i+1}/{len(err_indices)} | 已用 {elapsed:.0f}s | 预计剩 {eta:.0f}s")

print(f"\n[完成] 成功 {len(records)} / 失败 {len(failed)}")

# ──────────── 5. 只保留 raw 管线下仍预测错误的样本 ────────────
err_records    = [r for r in records if r["is_error"]]
flipped_to_ok  = len(records) - len(err_records)
if flipped_to_ok:
    print(f"[注意] raw 管线下有 {flipped_to_ok} 张样本由错变对（特征 vs 原始管线有微小差异），已排除")

# ──────────── 6. 按 (true, pred) 分组，置信度降序，每组 top3 ────────────
groups = defaultdict(list)
for r in err_records:
    groups[(r["true_label"], r["pred_label"])].append(r)

selected_ids = set()
for key, items in groups.items():
    items.sort(key=lambda x: -x["confidence"])
    for r in items[:TOP_K]:
        selected_ids.add(id(r))

# ──────────── 7. 复制文件到 error_cases/ ────────────
if os.path.exists(ERROR_DIR):
    shutil.rmtree(ERROR_DIR)
os.makedirs(ERROR_DIR)

copied = 0
for (t_lbl, p_lbl), items in groups.items():
    group_name = f"{LABEL_NAMES[t_lbl]}__误判为__{LABEL_NAMES[p_lbl]}"
    dest_dir   = os.path.join(ERROR_DIR, group_name)
    os.makedirs(dest_dir, exist_ok=True)
    for r in items[:TOP_K]:
        basename = os.path.basename(r["filepath"])
        new_name = f"conf{r['confidence']:.3f}_{basename}"
        shutil.copy2(r["filepath"], os.path.join(dest_dir, new_name))
        copied += 1

# ──────────── 8. 写 CSV（UTF-8 BOM，中文表头） ────────────
with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "数据集索引", "文件路径", "真实类别", "预测类别",
        "置信度", "OCR文本", "是否OCR兜底", "是否入选TOP3案例",
    ])
    # 按 (真实类, 预测类, 置信度降序) 输出，方便人工阅读
    err_records.sort(key=lambda x: (x["true_label"], x["pred_label"], -x["confidence"]))
    for r in err_records:
        writer.writerow([
            r["dataset_idx"],
            r["filepath"],
            LABEL_NAMES[r["true_label"]],
            LABEL_NAMES[r["pred_label"]],
            f"{r['confidence']:.4f}",
            r["ocr_text"],
            "是" if r["used_fallback"] else "否",
            "是" if id(r) in selected_ids else "否",
        ])
print(f"[保存] {CSV_PATH}")

# ──────────── 9. 汇报 ────────────
# (1) 成功/失败
print("\n========== 汇报 ==========")
print(f"(1) 成功处理: {len(records)}  |  失败: {len(failed)}")
if failed:
    print("     失败样本示例:")
    for f_ in failed[:5]:
        print(f"       - idx={f_['dataset_idx']}  {f_['filepath']}  err={f_['error']}")

# (2) 混淆对 TOP 3
pair_counts = sorted(
    [((t, p), len(v)) for (t, p), v in groups.items()],
    key=lambda x: -x[1],
)
print(f"(2) 混淆对 TOP 3（按错误数降序）:")
for (t, p), n in pair_counts[:3]:
    print(f"     {LABEL_NAMES[t]}  ->  {LABEL_NAMES[p]}   {n} 张")

# (3) OCR 兜底占比
fallback_n = sum(1 for r in err_records if r["used_fallback"])
ratio      = fallback_n / len(err_records) if err_records else 0.0
print(f"(3) 错误样本中 OCR 兜底(原始 OCR 为空→'{FALLBACK_TEXT}')占比: "
      f"{fallback_n}/{len(err_records)} = {ratio:.2%}")

# (4) error_cases/ 下的图片数
print(f"(4) error_cases/ 共复制 {copied} 张图（{len(groups)} 个分组，每组 ≤ {TOP_K}）")

print("\n[Step 3 完成]")
