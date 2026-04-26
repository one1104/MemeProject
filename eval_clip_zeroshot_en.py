# eval_clip_zeroshot_en.py
# 同 eval_clip_zeroshot.py 的数据口径 (MD5 去污染后 1201 张 held-out)，
# 但用英文 prompt ensemble 替换中文 prompt，验证 Chinese prompt 是否是 18.23% 过低的主因。
# 结果追加进 metrics.json 的 "clip_zero_shot_en" 节。
#
# 独立可运行: python eval_clip_zeroshot_en.py
import os
import json
import time
import hashlib
from collections import defaultdict

import torch
from PIL import Image

ROOT         = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(ROOT, "dataset")
TEST_DIR     = os.path.join(ROOT, "dataset_test")
LABEL_NAMES  = ["尴尬", "开心", "悲伤", "惊讶", "愤怒"]
# CLIP 训练语料以英文为主，用形容词形式（vs 抽象名词）一般更贴近面部/人像描述
ADJ_MAP = {
    "尴尬": "embarrassed",
    "开心": "happy",
    "悲伤": "sad",
    "惊讶": "surprised",
    "愤怒": "angry",
}
EN_PROMPT_TEMPLATES = [
    "a meme showing a {} expression",
    "a {} meme",
    "a photo of a {} person",
    "an image of someone who is {}",
    "a {} face",
]

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
METRICS_JSON = os.path.join(ROOT, "metrics.json")

# ──────────── 1. 哈希清洗（同 eval_clip_zeroshot.py） ────────────
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
train_hashes = set()
for _, _, p in iter_class_files(DATASET_DIR):
    train_hashes.add(md5(p))
print(f"  唯一内容 {len(train_hashes)}，用时 {time.time()-t0:.1f}s")

seen = set()
clean_samples = []
for cls, fname, p in iter_class_files(TEST_DIR):
    h = md5(p)
    if h in seen or h in train_hashes:
        continue
    seen.add(h)
    clean_samples.append((cls, p))
print(f"[哈希] clean held-out = {len(clean_samples)} 张")
assert len(clean_samples) == 1201, f"期望 1201 张，得到 {len(clean_samples)}"

# ──────────── 2. 加载 CLIP ────────────
import clip
print("\n[CLIP] 加载 ViT-B/32 ...")
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
MODEL.eval()

# ──────────── 3. 英文 prompt ensemble → 类别嵌入 ────────────
print("\n[英文 prompt ensemble]")
print(f"  label 映射: {ADJ_MAP}")
class_feats = []
for cls in LABEL_NAMES:
    adj = ADJ_MAP[cls]
    prompts = [tpl.format(adj) for tpl in EN_PROMPT_TEMPLATES]
    if cls == LABEL_NAMES[0]:
        for p in prompts: print(f"    e.g. {p!r}")
    tokens = clip.tokenize(prompts, truncate=True).to(DEVICE)
    with torch.no_grad():
        f = MODEL.encode_text(tokens)
        f = f / f.norm(dim=-1, keepdim=True)
        mean_f = f.mean(dim=0)
        mean_f = mean_f / mean_f.norm()
    class_feats.append(mean_f)
class_feats = torch.stack(class_feats, dim=0)

# ──────────── 4. 图像 → 最近类别 ────────────
LABEL_TO_IDX = {n: i for i, n in enumerate(LABEL_NAMES)}
correct   = 0
per_class = defaultdict(lambda: [0, 0])
t0 = time.time()

for i, (cls, path) in enumerate(clean_samples):
    try:
        with Image.open(path) as im:
            img_input = PREPROCESS(im.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            img_feat = MODEL.encode_image(img_input)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            logits = (img_feat @ class_feats.T).squeeze(0)
            pred_idx = int(logits.argmax().item())
    except Exception as ex:
        print(f"  跳过 {path}: {ex}")
        continue

    true_idx = LABEL_TO_IDX[cls]
    is_ok = (pred_idx == true_idx)
    correct += int(is_ok)
    per_class[cls][0] += int(is_ok)
    per_class[cls][1] += 1

    if (i + 1) % 200 == 0 or (i + 1) == len(clean_samples):
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(clean_samples) - i - 1)
        print(f"  进度 {i+1}/{len(clean_samples)} | 已用 {elapsed:.0f}s | 预计剩 {eta:.0f}s")

total = sum(v[1] for v in per_class.values())
zs_acc_en = correct / total if total else 0.0
OUR_METHOD_ACC = 0.7585
improvement_pp = (OUR_METHOD_ACC - zs_acc_en) * 100.0

print("\n========== 英文 prompt Zero-shot ==========")
print(f"CLIP Zero-shot (English prompt ensemble): {zs_acc_en:.4f} ({correct}/{total})")
print(f"本文方法: {OUR_METHOD_ACC:.4f}  |  提升 {improvement_pp:+.2f} pp")
print("\n[各类别 Zero-shot 准确率]")
for cls in LABEL_NAMES:
    ok, n = per_class[cls]
    rate = ok / n if n else 0
    print(f"  {cls:4s} ({ADJ_MAP[cls]:>12s})  {ok:>3d}/{n:<3d} = {rate:.4f}")

# ──────────── 5. 写回 metrics.json（追加节，不覆盖已有 clip_zero_shot） ────────────
with open(METRICS_JSON, "r", encoding="utf-8") as f:
    payload = json.load(f)

payload["clip_zero_shot_en"] = {
    "backbone":         "ViT-B/32",
    "prompt_templates": EN_PROMPT_TEMPLATES,
    "label_adjective_map": ADJ_MAP,
    "prompt_ensemble":  True,
    "accuracy":         round(zs_acc_en, 4),
    "correct":          correct,
    "total":            total,
    "per_class_accuracy": {
        cls: {"correct":  per_class[cls][0],
              "total":    per_class[cls][1],
              "accuracy": round(per_class[cls][0] / per_class[cls][1], 4)
                          if per_class[cls][1] else 0.0}
        for cls in LABEL_NAMES
    },
}
# 同步更新 comparison 节，加入 zh/en 对比
payload["comparison"] = {
    "zero_shot_acc_zh":  payload["clip_zero_shot"]["accuracy"],
    "zero_shot_acc_en":  round(zs_acc_en, 4),
    "our_method_acc":    OUR_METHOD_ACC,
    "our_vs_zh_pp":      round((OUR_METHOD_ACC - payload["clip_zero_shot"]["accuracy"]) * 100, 2),
    "our_vs_en_pp":      round(improvement_pp, 2),
    "zh_vs_en_pp":       round((zs_acc_en - payload["clip_zero_shot"]["accuracy"]) * 100, 2),
}
payload["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

with open(METRICS_JSON, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print(f"\n[保存] {METRICS_JSON}  (已追加 clip_zero_shot_en 节)")
