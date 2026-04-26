# eval_baselines_full.py
# 在两个子集 (seed=42 验证集 n=2549 & MD5 清洗后 held-out n=1201) 上，
# 统一评估四档 baseline + 本方法，产出:
#   1) baselines_full_metrics.json   (所有指标)
#   2) baselines_comparison.png      (双子集并列柱状图，SimHei/MSYH fallback)
#   3) 更新 metrics.json，补齐 val 部分
#
# Baseline 清单:
#   (B0) 随机基线 (理论 1/5)
#   (B1) CLIP Zero-shot (中文 prompt ensemble)
#   (B2) CLIP Zero-shot (英文 prompt ensemble)
#   (B3) CLIP Linear Probe (冻结图像特征 + sklearn LogisticRegression)
#   (M)  本文方法 CrossModalMLP (CLIP 图像 + EasyOCR + CLIP 文本 + MLP)
#
# 独立可运行: python eval_baselines_full.py
import os
import sys
import json
import time
import hashlib
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ──────────── 中文字体 ────────────
def _setup_chinese_font():
    candidates = ["SimHei", "Microsoft YaHei"]
    available  = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), "sans-serif")
    matplotlib.rcParams["font.sans-serif"] = [chosen, "Microsoft YaHei", "SimHei", "sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    print(f"[字体] {chosen}")
_setup_chinese_font()

# ──────────── 路径/常量 ────────────
ROOT         = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(ROOT, "dataset")
TEST_DIR     = os.path.join(ROOT, "dataset_test")
LABEL_NAMES  = ["尴尬", "开心", "悲伤", "惊讶", "愤怒"]
NUM_CLASSES  = len(LABEL_NAMES)
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

ZH_TEMPLATES = [
    "一张表达{}情绪的表情包",
    "{}的表情包",
    "一个看起来{}的人",
    "表达{}的图片",
    "显得很{}",
]
EN_ADJ = {"尴尬":"embarrassed", "开心":"happy", "悲伤":"sad", "惊讶":"surprised", "愤怒":"angry"}
EN_TEMPLATES = [
    "a meme showing a {} expression",
    "a {} meme",
    "a photo of a {} person",
    "an image of someone who is {}",
    "a {} face",
]

OUR_METHOD_VAL_ACC     = 0.8529
OUR_METHOD_HELDOUT_ACC = 0.7585

# ──────────── 1. 加载 val split (seed=42) + train 分区 ────────────
print("\n===== (1) 加载 seed=42 划分 =====")
img_feats_all = np.load(os.path.join(ROOT, "features_ocr_img.npy"))   # (12743, 512) L2 归一化
labels_all    = np.load(os.path.join(ROOT, "labels.npy"))
val_idx       = np.load(os.path.join(ROOT, "val_indices_seed42.npy"))
mask = np.zeros(len(labels_all), dtype=bool); mask[val_idx] = True
train_idx = np.where(~mask)[0]

X_val_img = img_feats_all[val_idx]          # (2549, 512)
y_val     = labels_all[val_idx]
X_tr_img  = img_feats_all[train_idx]        # (10194, 512)
y_tr      = labels_all[train_idx]
print(f"  train {len(y_tr)} | val {len(y_val)}")

# ──────────── 2. 构建 held-out 干净子集 (1201 张) ────────────
print("\n===== (2) 构建 held-out clean 子集 =====")
def md5(p, bufsize=1<<16):
    h = hashlib.md5()
    with open(p,"rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""): h.update(chunk)
    return h.hexdigest()

def iter_class_files(root):
    for cls in LABEL_NAMES:
        cd = os.path.join(root, cls)
        if not os.path.isdir(cd): continue
        for f in sorted(os.listdir(cd)):
            if f.lower().endswith((".jpg",".jpeg",".png")):
                yield cls, f, os.path.join(cd, f)

t0 = time.time()
train_hashes = {md5(p) for _,_,p in iter_class_files(DATASET_DIR)}
print(f"  dataset/ 唯一内容 {len(train_hashes)}，用时 {time.time()-t0:.1f}s")

seen = set()
heldout = []
for cls, _, p in iter_class_files(TEST_DIR):
    h = md5(p)
    if h in seen or h in train_hashes: continue
    seen.add(h)
    heldout.append((cls, p))
print(f"  held-out clean = {len(heldout)}")
assert len(heldout) == 1201

# ──────────── 3. 加载 CLIP，抽 held-out 图像特征（一次，复用给 B1/B2/B3） ────────────
import clip
print("\n===== (3) 加载 CLIP 并抽 held-out 1201 张图像特征 =====")
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
MODEL.eval()

X_ho_img = np.zeros((len(heldout), 512), dtype=np.float32)
y_ho     = np.array([LABEL_NAMES.index(c) for c,_ in heldout], dtype=np.int64)

t0 = time.time()
for i, (cls, path) in enumerate(heldout):
    with Image.open(path) as im:
        inp = PREPROCESS(im.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = MODEL.encode_image(inp)
        f = f / f.norm(dim=-1, keepdim=True)
    X_ho_img[i] = f.cpu().numpy()
    if (i+1) % 200 == 0 or (i+1) == len(heldout):
        elapsed = time.time()-t0
        print(f"  抽特征 {i+1}/{len(heldout)} | {elapsed:.0f}s")
print(f"  held-out 图像特征 shape = {X_ho_img.shape}")

# ──────────── 4. Zero-shot: 编码类别文本 → 余弦相似度 ────────────
def encode_class_feats(templates, label_to_str=None):
    """label_to_str: dict, 否则用原标签名"""
    if label_to_str is None:
        label_to_str = {n: n for n in LABEL_NAMES}
    cls_feats = []
    for cls in LABEL_NAMES:
        prompts = [t.format(label_to_str[cls]) for t in templates]
        tok = clip.tokenize(prompts, truncate=True).to(DEVICE)
        with torch.no_grad():
            f = MODEL.encode_text(tok)
            f = f / f.norm(dim=-1, keepdim=True)
            mf = f.mean(dim=0); mf = mf / mf.norm()
        cls_feats.append(mf.cpu().numpy())
    return np.stack(cls_feats, axis=0)    # (5,512)

def zero_shot_eval(X_img, y_true, class_feats):
    logits = X_img @ class_feats.T      # (N,5)
    preds  = logits.argmax(axis=1)
    acc    = (preds == y_true).mean()
    per_cls = {}
    for i, cls in enumerate(LABEL_NAMES):
        msk = y_true == i
        n = int(msk.sum())
        if n == 0:
            per_cls[cls] = {"correct":0, "total":0, "accuracy":0.0}
        else:
            c = int((preds[msk] == i).sum())
            per_cls[cls] = {"correct":c, "total":n, "accuracy": round(c/n, 4)}
    return float(acc), int((preds == y_true).sum()), len(y_true), per_cls

print("\n===== (4) Zero-shot 评估 (中文 prompt) =====")
cls_feats_zh = encode_class_feats(ZH_TEMPLATES)
zh_val = zero_shot_eval(X_val_img, y_val, cls_feats_zh)
zh_ho  = zero_shot_eval(X_ho_img,  y_ho,  cls_feats_zh)
print(f"  val  : {zh_val[0]:.4f} ({zh_val[1]}/{zh_val[2]})")
print(f"  held : {zh_ho[0]:.4f}  ({zh_ho[1]}/{zh_ho[2]})")

print("\n===== (5) Zero-shot 评估 (英文 prompt) =====")
cls_feats_en = encode_class_feats(EN_TEMPLATES, label_to_str=EN_ADJ)
en_val = zero_shot_eval(X_val_img, y_val, cls_feats_en)
en_ho  = zero_shot_eval(X_ho_img,  y_ho,  cls_feats_en)
print(f"  val  : {en_val[0]:.4f} ({en_val[1]}/{en_val[2]})")
print(f"  held : {en_ho[0]:.4f}  ({en_ho[1]}/{en_ho[2]})")

# ──────────── 6. Linear Probe: LogisticRegression on CLIP 图像特征 ────────────
print("\n===== (6) Linear Probe (LogReg on CLIP image features) =====")
t0 = time.time()
clf = LogisticRegression(max_iter=2000, n_jobs=-1, C=1.0, solver="lbfgs",
                         multi_class="multinomial")
clf.fit(X_tr_img, y_tr)
print(f"  训练完成，用时 {time.time()-t0:.1f}s")

def lp_eval(clf, X, y):
    preds = clf.predict(X)
    acc   = (preds == y).mean()
    per_cls = {}
    for i, cls in enumerate(LABEL_NAMES):
        msk = y == i; n = int(msk.sum())
        c = int((preds[msk] == i).sum()) if n else 0
        per_cls[cls] = {"correct":c, "total":n, "accuracy": round(c/n,4) if n else 0.0}
    return float(acc), int((preds==y).sum()), len(y), per_cls

lp_val = lp_eval(clf, X_val_img, y_val)
lp_ho  = lp_eval(clf, X_ho_img,  y_ho)
print(f"  val  : {lp_val[0]:.4f} ({lp_val[1]}/{lp_val[2]})")
print(f"  held : {lp_ho[0]:.4f}  ({lp_ho[1]}/{lp_ho[2]})")

# ──────────── 7. 组装 baselines_full_metrics.json ────────────
print("\n===== (7) 写 baselines_full_metrics.json =====")
def pack(res):
    a, c, n, per = res
    return {"accuracy": round(a,4), "correct": c, "total": n, "per_class_accuracy": per}

payload = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "subsets": {
        "val":       {"name": "seed=42 验证集",               "n": int(len(y_val))},
        "held_out":  {"name": "MD5 清洗后独立测试集",          "n": int(len(y_ho))},
    },
    "label_names": LABEL_NAMES,
    "baselines": {
        "random": {
            "note": "理论随机基线 = 1/num_classes",
            "val":       {"accuracy": round(1.0/NUM_CLASSES, 4)},
            "held_out":  {"accuracy": round(1.0/NUM_CLASSES, 4)},
        },
        "clip_zero_shot_zh": {
            "backbone": "ViT-B/32", "prompt_templates": ZH_TEMPLATES, "prompt_ensemble": True,
            "val":      pack(zh_val),
            "held_out": pack(zh_ho),
        },
        "clip_zero_shot_en": {
            "backbone": "ViT-B/32", "prompt_templates": EN_TEMPLATES,
            "label_adjective_map": EN_ADJ, "prompt_ensemble": True,
            "val":      pack(en_val),
            "held_out": pack(en_ho),
        },
        "clip_linear_probe": {
            "note": "冻结 CLIP 图像特征 + sklearn LogisticRegression, 仅图像模态",
            "classifier": "LogisticRegression(multinomial, C=1.0, lbfgs)",
            "train_size": int(len(y_tr)),
            "val":      pack(lp_val),
            "held_out": pack(lp_ho),
        },
    },
    "our_method": {
        "name": "CrossModalMLP (CLIP 图像 + EasyOCR + CLIP 文本 + MLP)",
        "val":       {"accuracy": OUR_METHOD_VAL_ACC,     "source": "eval_confusion.py"},
        "held_out":  {"accuracy": OUR_METHOD_HELDOUT_ACC, "source": "eval_heldout_test.py"},
    },
}
# 对比表
def row(name, val_acc, ho_acc):
    return {"name":name, "val_acc":round(val_acc,4), "held_out_acc":round(ho_acc,4),
            "gap_pp": round((val_acc-ho_acc)*100, 2)}
payload["comparison_table"] = [
    row("随机基线 (理论)",              1/NUM_CLASSES, 1/NUM_CLASSES),
    row("CLIP Zero-shot (中文 prompt)", zh_val[0],     zh_ho[0]),
    row("CLIP Zero-shot (英文 prompt)", en_val[0],     en_ho[0]),
    row("CLIP Linear Probe (图像only)", lp_val[0],     lp_ho[0]),
    row("本文跨模态 MLP (图像+OCR文本)", OUR_METHOD_VAL_ACC, OUR_METHOD_HELDOUT_ACC),
]

OUT_JSON = os.path.join(ROOT, "baselines_full_metrics.json")
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print(f"  保存 {OUT_JSON}")

# ──────────── 8. 更新原 metrics.json，补 val 部分 ────────────
METRICS_JSON = os.path.join(ROOT, "metrics.json")
if os.path.exists(METRICS_JSON):
    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        m = json.load(f)
    m.setdefault("clip_zero_shot", {})["val_subset"] = {
        "n_samples": int(len(y_val)),
        "accuracy":  round(zh_val[0], 4),
        "correct":   zh_val[1],
        "total":     zh_val[2],
        "per_class_accuracy": zh_val[3],
    }
    m.setdefault("clip_zero_shot_en", {})["val_subset"] = {
        "n_samples": int(len(y_val)),
        "accuracy":  round(en_val[0], 4),
        "correct":   en_val[1],
        "total":     en_val[2],
        "per_class_accuracy": en_val[3],
    }
    m["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(f"  更新 {METRICS_JSON} (已补 val_subset)")
else:
    print("  metrics.json 不存在，跳过更新")

# ──────────── 9. 柱状图: 四档 baseline + 本方法，双子集并列 ────────────
print("\n===== (9) 画 baselines_comparison.png =====")
names = [r["name"] for r in payload["comparison_table"]]
val_accs = [r["val_acc"]      for r in payload["comparison_table"]]
ho_accs  = [r["held_out_acc"] for r in payload["comparison_table"]]

x = np.arange(len(names))
w = 0.38
fig, ax = plt.subplots(figsize=(13, 6.5))
b1 = ax.bar(x - w/2, val_accs, w, label=f"验证集 (n={len(y_val)})",   color="#4C72B0")
b2 = ax.bar(x + w/2, ho_accs,  w, label=f"独立测试集 (n={len(y_ho)})", color="#DD8452")

for bars, vals in [(b1, val_accs), (b2, ho_accs)]:
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width()/2, v + 0.01, f"{v*100:.2f}%",
                ha="center", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(names, rotation=12, ha="right")
ax.set_ylabel("准确率")
ax.set_ylim(0, 1.0)
ax.set_yticks(np.arange(0, 1.01, 0.1))
ax.set_yticklabels([f"{int(t*100)}%" for t in np.arange(0, 1.01, 0.1)])
ax.axhline(1/NUM_CLASSES, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
           label=f"随机基线 {1/NUM_CLASSES*100:.0f}%")
ax.set_title("四档 Baseline 对比 (验证集 vs 独立测试集)", fontsize=14)
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()

OUT_PNG = os.path.join(ROOT, "baselines_comparison.png")
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"  保存 {OUT_PNG}")

print("\n========== 汇总 ==========")
print(f"{'Baseline':<35} {'Val':>10}   {'Held-out':>10}   {'Gap':>7}")
print("-"*68)
for r in payload["comparison_table"]:
    print(f"{r['name']:<35} {r['val_acc']*100:>9.2f}%   {r['held_out_acc']*100:>9.2f}%   {r['gap_pp']:>+6.2f}pp")
print("\n[完成]")
