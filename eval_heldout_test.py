# eval_heldout_test.py
# 对 dataset_test/ 做内容哈希清洗，构建真正独立的 held-out test set，
# 并用完整 raw 推理管线 (CLIP 图像 + EasyOCR + CLIP 文本 + MLP, 纯 argmax)
# 分别给出 (a) 原始 1661 张、(b) 内部去重后、(c) 与 dataset 去重后 三种口径的准确率。
#
# 独立可运行：python eval_heldout_test.py
import os
import sys
import csv
import time
import hashlib
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# ──────────── 路径 / 常量 ────────────
ROOT          = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR   = os.path.join(ROOT, "dataset")
TEST_DIR      = os.path.join(ROOT, "dataset_test")
LABEL_NAMES   = ["尴尬", "开心", "悲伤", "惊讶", "愤怒"]
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
FALLBACK_TEXT = "一张表情包"

CSV_PATH = os.path.join(ROOT, "_heldout_test_summary.csv")

# ──────────── 1. 哈希分析 ────────────
def md5(path, bufsize=1 << 16):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()

def iter_class_files(root):
    """yield (class_name, file_name, full_path)，仅处理 LABEL_NAMES 类别"""
    for cls in LABEL_NAMES:
        cd = os.path.join(root, cls)
        if not os.path.isdir(cd):
            continue
        for f in sorted(os.listdir(cd)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                yield cls, f, os.path.join(cd, f)

print("[哈希] 扫描 dataset/ ...")
t0 = time.time()
train_hash_to_labels = defaultdict(set)   # md5 -> set of class names
for cls, _, p in iter_class_files(DATASET_DIR):
    train_hash_to_labels[md5(p)].add(cls)
print(f"[哈希] dataset/ 唯一内容 {len(train_hash_to_labels)}，用时 {time.time()-t0:.1f}s")

print("[哈希] 扫描 dataset_test/ ...")
t0 = time.time()
test_entries = []    # list of dicts，每条代表一张 test 原图
test_hash_first = {} # md5 -> 首次出现的 index（判定内部重复）
for cls, fname, p in iter_class_files(TEST_DIR):
    h = md5(p)
    is_internal_dup = h in test_hash_first
    idx = len(test_entries)
    if not is_internal_dup:
        test_hash_first[h] = idx
    test_entries.append({
        "idx":       idx,
        "class":     cls,
        "fname":     fname,
        "path":      p,
        "md5":       h,
        "is_internal_dup": is_internal_dup,
    })
print(f"[哈希] dataset_test/ 原始 {len(test_entries)}，唯一 {len(test_hash_first)}，"
      f"内部重复 {len(test_entries)-len(test_hash_first)}，用时 {time.time()-t0:.1f}s")

# 打标每张 test 图的"身份"
overlap_same = overlap_cross = clean = 0
for e in test_entries:
    train_labels = train_hash_to_labels.get(e["md5"])
    if train_labels is None:
        e["overlap"] = "clean"     # 与 dataset 完全无重复
        if not e["is_internal_dup"]:
            clean += 1
    elif e["class"] in train_labels:
        e["overlap"] = "same_class"
        if not e["is_internal_dup"]:
            overlap_same += 1
    else:
        e["overlap"] = "cross_class"
        if not e["is_internal_dup"]:
            overlap_cross += 1

print(f"[清洗统计] (按唯一 md5 计):")
print(f"  同类重复 : {overlap_same}")
print(f"  跨类冲突 : {overlap_cross}")
print(f"  纯独立    : {clean}")
print(f"  合计      : {overlap_same + overlap_cross + clean}  (应等于唯一数 {len(test_hash_first)})")

# ──────────── 2. 加载 CLIP + MLP ────────────
import clip
print("\n[CLIP] 加载 ViT-B/32 ...")
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

# ──────────── 3. 初始化 EasyOCR ────────────
print("[EasyOCR] 正在加载 ch_sim+en 模型，首次加载约 1-2 分钟...")
t0 = time.time()
import easyocr
reader = easyocr.Reader(["ch_sim", "en"], verbose=False)
print(f"[EasyOCR] 就绪，用时 {time.time()-t0:.1f}s")

# ──────────── 4. 对所有 1661 张 test 图跑一次完整 raw 管线 ────────────
LABEL_TO_IDX = {n: i for i, n in enumerate(LABEL_NAMES)}
failed = []

t_start = time.time()
for i, e in enumerate(test_entries):
    try:
        with Image.open(e["path"]) as im:
            img_input = PREPROCESS(im.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            img_feat = MODEL.encode_image(img_input)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        img_np = np.array(Image.open(e["path"]).convert("RGB"))
        ocr_result = reader.readtext(img_np, detail=0)
        ocr_raw = " ".join(ocr_result).strip()
        used_fallback = (ocr_raw == "")
        text_in = FALLBACK_TEXT if used_fallback else ocr_raw

        txt_token = clip.tokenize([text_in], truncate=True).to(DEVICE)
        with torch.no_grad():
            txt_feat = MODEL.encode_text(txt_token)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            logits = MLP(img_feat, txt_feat)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx = int(probs.argmax())
        conf     = float(probs[pred_idx])

        true_idx = LABEL_TO_IDX[e["class"]]
        e.update({
            "pred":         LABEL_NAMES[pred_idx],
            "true":         e["class"],
            "confidence":   conf,
            "ocr_text":     ocr_raw,
            "used_fallback": used_fallback,
            "correct":      pred_idx == true_idx,
        })
    except Exception as ex:
        e.update({"pred": None, "true": e["class"], "confidence": 0.0,
                  "ocr_text": "", "used_fallback": False, "correct": False})
        failed.append({"idx": e["idx"], "path": e["path"], "error": repr(ex)})

    if (i + 1) % 100 == 0 or (i + 1) == len(test_entries):
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(test_entries) - i - 1)
        print(f"  进度 {i+1}/{len(test_entries)} | 已用 {elapsed:.0f}s | 预计剩 {eta:.0f}s")

print(f"\n[完成] 成功 {len(test_entries)-len(failed)} / 失败 {len(failed)}")

# ──────────── 5. 三种口径的准确率 ────────────
def acc(entries):
    if not entries: return 0.0, 0, 0
    ok = sum(1 for e in entries if e.get("correct"))
    return ok / len(entries), ok, len(entries)

# (a) 原始 1661
a_entries = [e for e in test_entries if e.get("pred") is not None]
a_acc, a_ok, a_n = acc(a_entries)

# (b) 内部去重 (每个 md5 只保留首次出现)
b_entries = [e for e in test_entries if (not e["is_internal_dup"]) and e.get("pred") is not None]
b_acc, b_ok, b_n = acc(b_entries)

# (c) 与 dataset/ 去重后真正 held-out
c_entries = [e for e in b_entries if e["overlap"] == "clean"]
c_acc, c_ok, c_n = acc(c_entries)

# 额外：被污染子集的准确率（用于对比暴露 contamination 带来的虚高）
contam_entries = [e for e in b_entries if e["overlap"] in ("same_class", "cross_class")]
cc_acc, cc_ok, cc_n = acc(contam_entries)

print("\n========== 三种口径准确率 ==========")
print(f"(a) 原始 1,661 张                 : {a_acc:.4f} ({a_ok}/{a_n})")
print(f"(b) 内部去重 {b_n} 张              : {b_acc:.4f} ({b_ok}/{b_n})")
print(f"(c) 真正 held-out {c_n} 张         : {c_acc:.4f} ({c_ok}/{c_n})")
print(f"(附) 污染子集 {cc_n} 张            : {cc_acc:.4f} ({cc_ok}/{cc_n})")
print(f"     -> (b) 与 (c) 的差 = {(b_acc-c_acc)*100:+.2f}pp  (contamination 造成的虚高)")

# ──────────── 6. 各类别在 held-out 上的表现 ────────────
print("\n[held-out 各类别准确率]")
by_cls = defaultdict(lambda: [0, 0])
for e in c_entries:
    by_cls[e["class"]][0] += int(e["correct"])
    by_cls[e["class"]][1] += 1
for cls in LABEL_NAMES:
    ok, n = by_cls[cls]
    print(f"  {cls:4s}  {ok:>3d}/{n:<3d} = {ok/n:.4f}" if n else f"  {cls}: 0")

# ──────────── 7. 写 CSV (UTF-8 BOM) ────────────
with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "序号", "文件路径", "真实类别", "预测类别", "置信度",
        "OCR文本", "是否OCR兜底", "是否预测正确",
        "重复标记",  # same_class / cross_class / clean / internal_dup
    ])
    for e in test_entries:
        tag = "internal_dup" if e["is_internal_dup"] else e.get("overlap", "clean")
        w.writerow([
            e["idx"], e["path"], e["true"],
            e.get("pred", ""), f"{e.get('confidence', 0):.4f}",
            e.get("ocr_text", ""),
            "是" if e.get("used_fallback") else "否",
            "是" if e.get("correct") else "否",
            tag,
        ])
print(f"\n[保存] {CSV_PATH}")

# ──────────── 8. 论文可用汇总 ────────────
fallback_rate_c = sum(1 for e in c_entries if e.get("used_fallback")) / max(c_n, 1)
print("\n========== 论文可引用汇总 ==========")
print(f"dataset_test/ 清洗:")
print(f"  原始样本数           : {len(test_entries)}")
print(f"  内部 MD5 重复        : {len(test_entries) - len(test_hash_first)}")
print(f"  与 dataset/ 内容重复 : {overlap_same + overlap_cross}  "
      f"(同类 {overlap_same} / 跨类冲突 {overlap_cross})")
print(f"  最终独立测试集       : {c_n}")
print(f"独立测试集准确率       : {c_acc:.4f}  (vs 验证集 0.8529)")
print(f"独立测试集 OCR 兜底率  : {fallback_rate_c:.2%}")
print("\n[Step 独立测试集评估 完成]")
