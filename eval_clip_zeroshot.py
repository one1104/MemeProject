# eval_clip_zeroshot.py
# 在独立 held-out 测试集 (MD5 去污染后 1201 张) 上跑 CLIP ViT-B/32 Zero-shot 基线
# (无需 EasyOCR，无需 MLP 分类头)，用中文 prompt 集合做分类。
# 与 eval_heldout_test.py 的"本文方法 75.85%"严格同口径 (同一批样本、同一份哈希清洗规则)。
#
# 产出: metrics.json (结构化指标)
# 独立可运行: python eval_clip_zeroshot.py
import os
import json
import time
import hashlib
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

# ──────────── 路径 / 常量 ────────────
ROOT         = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(ROOT, "dataset")
TEST_DIR     = os.path.join(ROOT, "dataset_test")
LABEL_NAMES  = ["尴尬", "开心", "悲伤", "惊讶", "愤怒"]
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
METRICS_JSON = os.path.join(ROOT, "metrics.json")

# 中文 prompt ensemble (5 个模板 × 5 类 = 25 条文本)
PROMPT_TEMPLATES = [
    "一张表达{}情绪的表情包",
    "{}的表情包",
    "一个看起来{}的人",
    "表达{}的图片",
    "显得很{}",
]

# ──────────── 1. 哈希清洗，重建与 eval_heldout_test.py 同样的 1201 张 clean 子集 ────────────
def md5(path, bufsize=1 << 16):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()

def iter_class_files(root):
    for cls in LABEL_NAMES:
        cd = os.path.join(root, cls)
        if not os.path.isdir(cd): continue
        for f in sorted(os.listdir(cd)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                yield cls, f, os.path.join(cd, f)

print("[哈希] dataset/ ...")
t0 = time.time()
train_hash_to_labels = defaultdict(set)
for cls, _, p in iter_class_files(DATASET_DIR):
    train_hash_to_labels[md5(p)].add(cls)
print(f"  唯一内容 {len(train_hash_to_labels)}，用时 {time.time()-t0:.1f}s")

print("[哈希] dataset_test/ ...")
t0 = time.time()
seen_hashes = set()
clean_samples = []   # (class, path, md5)
raw_count = 0
for cls, fname, p in iter_class_files(TEST_DIR):
    raw_count += 1
    h = md5(p)
    if h in seen_hashes:
        continue          # 内部 MD5 重复，跳过
    seen_hashes.add(h)
    if h in train_hash_to_labels:
        continue          # 与 dataset/ 内容重叠（同类 or 跨类冲突），丢弃
    clean_samples.append((cls, p, h))
print(f"  原始 {raw_count}，清洗后 clean held-out = {len(clean_samples)}，"
      f"用时 {time.time()-t0:.1f}s")

assert len(clean_samples) == 1201, f"期望 1201 张，得到 {len(clean_samples)}，哈希逻辑与之前不一致"

# ──────────── 2. 加载 CLIP ────────────
import clip
print("\n[CLIP] 加载 ViT-B/32 ...")
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
MODEL.eval()
print(f"[CLIP] 就绪，设备={DEVICE}")

# ──────────── 3. 构建类别文本嵌入（prompt ensemble） ────────────
print("\n[文本编码] 5 类 × 5 模板 = 25 条 prompts ...")
class_text_feats = []   # (5, 512)
for cls in LABEL_NAMES:
    prompts = [tpl.format(cls) for tpl in PROMPT_TEMPLATES]
    tokens  = clip.tokenize(prompts, truncate=True).to(DEVICE)
    with torch.no_grad():
        feats = MODEL.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        mean_feat = feats.mean(dim=0)
        mean_feat = mean_feat / mean_feat.norm()
    class_text_feats.append(mean_feat)
class_text_feats = torch.stack(class_text_feats, dim=0)    # (5, 512)
print(f"  class_text_feats.shape = {tuple(class_text_feats.shape)}")

# ──────────── 4. 对 1201 张图跑图像编码 + 最近类别 ────────────
LABEL_TO_IDX = {n: i for i, n in enumerate(LABEL_NAMES)}
correct   = 0
per_class = defaultdict(lambda: [0, 0])   # cls -> [correct, total]
t_start   = time.time()

for i, (cls, path, _) in enumerate(clean_samples):
    try:
        with Image.open(path) as im:
            img_input = PREPROCESS(im.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            img_feat = MODEL.encode_image(img_input)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits   = (img_feat @ class_text_feats.T).squeeze(0)
            pred_idx = int(logits.argmax().item())
    except Exception as ex:
        print(f"  跳过 {path}: {ex}")
        continue

    true_idx = LABEL_TO_IDX[cls]
    is_ok    = (pred_idx == true_idx)
    correct += int(is_ok)
    per_class[cls][0] += int(is_ok)
    per_class[cls][1] += 1

    if (i + 1) % 200 == 0 or (i + 1) == len(clean_samples):
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(clean_samples) - i - 1)
        print(f"  进度 {i+1}/{len(clean_samples)} | 已用 {elapsed:.0f}s | 预计剩 {eta:.0f}s")

total = sum(v[1] for v in per_class.values())
zs_acc = correct / total if total else 0.0

# 对比本文方法 (held-out = 75.85%)
OUR_METHOD_ACC = 0.7585   # 与 _heldout_test_summary.csv 上计算的数字一致
improvement_pp = (OUR_METHOD_ACC - zs_acc) * 100.0

print("\n========== 结果 ==========")
print(f"CLIP Zero-shot (中文 prompt ensemble): {zs_acc:.4f} ({correct}/{total})")
print(f"本文方法 (跨模态 MLP, held-out n=1201): {OUR_METHOD_ACC:.4f}")
print(f"提升幅度: {improvement_pp:+.2f} pp")
print("\n[各类别 Zero-shot 准确率]")
for cls in LABEL_NAMES:
    ok, n = per_class[cls]
    rate  = ok / n if n else 0.0
    print(f"  {cls:4s}  {ok:>3d}/{n:<3d} = {rate:.4f}")

# ──────────── 5. 写 metrics.json ────────────
payload = {
    "generated_at":      time.strftime("%Y-%m-%d %H:%M:%S"),
    "dataset":           "dataset_test (MD5 cleaned held-out)",
    "n_samples":         total,
    "label_names":       LABEL_NAMES,
    "clip_zero_shot": {
        "backbone":       "ViT-B/32",
        "prompt_templates": PROMPT_TEMPLATES,
        "prompt_ensemble": True,
        "accuracy":       round(zs_acc, 4),
        "correct":        correct,
        "total":          total,
        "per_class_accuracy": {
            cls: {"correct": per_class[cls][0],
                  "total":   per_class[cls][1],
                  "accuracy": round(per_class[cls][0] / per_class[cls][1], 4)
                              if per_class[cls][1] else 0.0}
            for cls in LABEL_NAMES
        },
    },
    "our_method": {
        "name":     "CrossModalMLP (CLIP 图像 + EasyOCR + CLIP 文本 + MLP)",
        "accuracy": OUR_METHOD_ACC,
        "source":   "eval_heldout_test.py, clean held-out n=1201",
    },
    "comparison": {
        "zero_shot_acc":    round(zs_acc, 4),
        "our_method_acc":   OUR_METHOD_ACC,
        "improvement_pp":   round(improvement_pp, 2),
    },
}
with open(METRICS_JSON, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print(f"\n[保存] {METRICS_JSON}")
