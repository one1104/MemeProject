# extract_features_ocr.py
import torch
import clip
import numpy as np
import os
import json
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
MODEL.eval()

DATASET    = r"C:\Users\meng-\Desktop\MemeProject\dataset"
LABEL_NAMES = ["尴尬", "开心", "悲伤", "惊讶", "愤怒"]

img_feats, txt_feats, labels = [], [], []

try:
    import easyocr
    reader = easyocr.Reader(['ch_sim', 'en'], verbose=False)
    use_ocr = True
except:
    use_ocr = False
    print("EasyOCR不可用，全部用兜底文本")

for label_idx, label in enumerate(LABEL_NAMES):
    folder = os.path.join(DATASET, label)
    if not os.path.isdir(folder):
        print(f"[跳过] {label} 文件夹不存在")
        continue
    imgs = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    print(f"[{label}] 共 {len(imgs)} 张")

    for fname in imgs:
        path = os.path.join(folder, fname)
        try:
            # 图像特征
            img = PREPROCESS(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                img_feat = MODEL.encode_image(img)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            # 文本特征
            text = "一张表情包"
            if use_ocr:
                try:
                    img_np = np.array(Image.open(path).convert("RGB"))
                    result = reader.readtext(img_np, detail=0)
                    ocr_text = " ".join(result).strip()
                    if ocr_text:
                        text = ocr_text
                except:
                    pass

            txt_token = clip.tokenize([text], truncate=True).to(DEVICE)
            with torch.no_grad():
                txt_feat = MODEL.encode_text(txt_token)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            img_feats.append(img_feat.cpu().numpy())
            txt_feats.append(txt_feat.cpu().numpy())
            labels.append(label_idx)

        except Exception as e:
            print(f"  跳过 {fname}: {e}")

img_feats = np.concatenate(img_feats, axis=0)
txt_feats = np.concatenate(txt_feats, axis=0)
labels    = np.array(labels)

np.save("features_ocr_img.npy", img_feats)
np.save("features_ocr_txt.npy", txt_feats)
np.save("labels.npy", labels)

with open("label_names.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(LABEL_NAMES))

print(f"\n图像特征: {img_feats.shape} | 文本特征: {txt_feats.shape}")
print("保存完成")