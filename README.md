# MemeProject · 表情包情绪识别与生成

基于 CLIP + OCR 的跨模态表情包情绪分类，支持 5 类情绪识别（尴尬 / 开心 / 悲伤 / 惊讶 / 愤怒），并能根据情绪生成带文案的动态 GIF。

## 性能

| 口径 | 样本数 | Accuracy |
|---|---:|---:|
| 验证集（seed=42） | 2,549 | **85.29%** |
| 独立测试集（MD5 去污染 held-out） | 1,201 | **75.85%** |

### 基线对比（同一口径）

| 方法 | val | held-out |
|---|---:|---:|
| 随机基线 | 20.00% | 20.00% |
| CLIP Zero-shot（中文 prompt） | 17.81% | 18.23% |
| CLIP Zero-shot（英文 prompt） | 56.02% | 55.79% |
| CLIP Linear Probe（图像 only） | 73.44% | 69.44% |
| **本文方法（CrossModalMLP）** | **85.29%** | **75.85%** |

详细指标、错误分析与论文表述建议见 [`PAPER_STATS.md`](PAPER_STATS.md)。

## 功能

- **情绪识别**：上传表情包图片 + 可选文字，返回情绪类别和各类别概率
- **文案库**：根据情绪返回推荐文案（来自 `meme_texts_3000.json`）
- **动态 GIF 生成**：把图片 + 文案合成为带动效的 GIF
- **Web 界面**：Flask + 静态页面，浏览器直接用

## 技术栈

- **特征提取**：OpenAI CLIP (ViT-B/32) 图像/文本双塔编码
- **OCR**：EasyOCR（中英文）
- **分类器**：双分支 MLP（图像分支 + 文本分支 → 拼接 → 分类头）
- **后端**：Flask + flask-cors
- **前端**：原生 HTML/CSS/JS

## 目录结构

```
MemeProject/
├── app_flask.py              # Flask 服务入口
├── inference.py              # 情绪推理 + GIF 生成
├── extract_features_ocr.py   # CLIP + OCR 特征提取
├── train_ocr_crossmodal.py   # 训练脚本
├── test_inference_final.py   # 推理测试
├── static/                   # 前端页面
│   ├── index.html
│   ├── app.js
│   └── style.css
├── label_names.txt           # 类别标签
├── meme_texts_3000.json      # 文案库
└── README.md
```

## 需要单独下载的文件

出于仓库大小考虑，以下文件**未纳入 git**，需要自行准备：

| 文件 | 说明 | 大小 |
|------|------|------|
| `best_model_ocr.pth` | 训练好的分类器权重 | ~1.7 MB |
| `features_ocr_img.npy` | CLIP 图像特征（训练用） | ~25 MB |
| `features_ocr_txt.npy` | CLIP 文本特征（训练用） | ~25 MB |
| `labels.npy` | 训练标签 | ~50 KB |
| `dataset/` | 原始图片数据集（5 类文件夹） | ~310 MB |

> 如需获取以上资源，请联系仓库作者（或自行运行 `extract_features_ocr.py` 重新生成特征）。

## 安装

```bash
# 1. 克隆仓库
git clone https://github.com/one1104/MemeProject.git
cd MemeProject

# 2. 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. 安装依赖
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install easyocr flask flask-cors pillow numpy scikit-learn
```

## 使用

### 启动 Web 服务

```bash
python app_flask.py
```

浏览器打开 http://localhost:5000 。

### 重新训练（可选）

```bash
# 1. 准备数据集：dataset/尴尬/*.jpg, dataset/开心/*.jpg ...
# 2. 提取特征
python extract_features_ocr.py
# 3. 训练分类器
python train_ocr_crossmodal.py
# 产物：best_model_ocr.pth
```

## API

| 路由 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 前端页面 |
| `/api/predict` | POST | 情绪识别（multipart: image, text） |
| `/api/generate` | POST | 生成 GIF（json: emotion, text, img_id） |
| `/api/texts/<emotion>` | GET | 获取某情绪的推荐文案 |

## License

个人学习项目。
