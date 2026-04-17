import torch
import torch.nn as nn
import clip
import numpy as np
import os
import random
import time
import imageio
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)

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

LABEL_NAMES = open("label_names.txt", encoding="utf-8").read().splitlines()
MLP = CrossModalMLP().to(DEVICE)
MLP.load_state_dict(torch.load("best_model_ocr.pth", map_location=DEVICE))
MLP.eval()

# EasyOCR 全局单例，首次调用时懒加载，后续复用
_OCR_READER = None

def _get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        import easyocr
        _OCR_READER = easyocr.Reader(['ch_sim', 'en'], verbose=False)
    return _OCR_READER


def predict_emotion(img_path, user_text=""):
    try:
        with Image.open(img_path) as img:
            img_input = PREPROCESS(img.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            img_feat = MODEL.encode_image(img_input)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        if not user_text.strip():
            try:
                reader = _get_ocr_reader()
                img_np = np.array(Image.open(img_path).convert("RGB"))
                ocr_result = reader.readtext(img_np, detail=0)
                user_text = " ".join(ocr_result).strip()
            except Exception:
                user_text = ""

        if not user_text:
            user_text = "一张表情包"

        txt_input = clip.tokenize([user_text], truncate=True).to(DEVICE)
        with torch.no_grad():
            txt_feat = MODEL.encode_text(txt_input)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            logits = MLP(img_feat, txt_feat)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        probs_dict = {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))}
        probs_dict = dict(sorted(probs_dict.items(), key=lambda x: x[1], reverse=True))
        top_emotion = list(probs_dict.keys())[0]
        top_prob    = list(probs_dict.values())[0]

        if top_prob < 0.5:
            return "未知", probs_dict

        return top_emotion, probs_dict

    except Exception as e:
        print(f"识别失败: {e}")
        return "中性", {name: 1/len(LABEL_NAMES) for name in LABEL_NAMES}


def get_font(size):
    for path in ["msyh.ttc", "simhei.ttf",
                 "C:/Windows/Fonts/msyh.ttc",
                 "C:/Windows/Fonts/simhei.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text_on_rgba(img, text, x_off=0, y_off=0, alpha=255):
    img = img.convert("RGBA")
    txt_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)
    w, h = img.size

    base_size = int(h / 9)
    font = get_font(base_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    if tw > w * 0.9:
        font = get_font(int(base_size * (w * 0.9) / tw))
        bbox = draw.textbbox((0, 0), text, font=font)

    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (w - tw) // 2 + x_off
    y = h - th - 30 + y_off

    for dx, dy in [(-2,-2),(2,-2),(-2,2),(2,2)]:
        draw.text((x+dx, y+dy), text, font=font, fill=(0, 0, 0, alpha))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, alpha))

    return Image.alpha_composite(img, txt_layer)


# ══════════════════════════════════════
# 缓动函数
# ══════════════════════════════════════
def ease_out_cubic(t):
    return 1 - (1 - t) ** 3

def ease_in_out_cubic(t):
    return 4*t*t*t if t < 0.5 else 1 - (-2*t+2)**3/2

def ease_out_elastic(t):
    if t == 0 or t == 1: return t
    return pow(2, -10*t) * np.sin((t*10 - 0.75) * (2*np.pi)/3) + 1

def ease_out_bounce(t):
    if t < 1/2.75:   return 7.5625*t*t
    elif t < 2/2.75: t -= 1.5/2.75;  return 7.5625*t*t + 0.75
    elif t < 2.5/2.75: t -= 2.25/2.75; return 7.5625*t*t + 0.9375
    else:             t -= 2.625/2.75; return 7.5625*t*t + 0.984375

def ease_in_cubic(t):
    return t * t * t


# ══════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════
def overlay_color(img, r, g, b, alpha_val):
    layer = Image.new("RGBA", img.size, (r, g, b, max(0, min(255, int(alpha_val)))))
    return Image.alpha_composite(img.convert("RGBA"), layer)

def zoom_center(img, scale):
    w, h = img.size
    nw, nh = int(w*scale), int(h*scale)
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left, top = (nw-w)//2, (nh-h)//2
    return resized.crop((left, top, left+w, top+h))


# ══════════════════════════════════════
# 主生成函数
# ══════════════════════════════════════
def glitch_channel(img, dx, dy, channel_idx):
    """将指定通道(0=R,1=G,2=B)整体偏移，制造故障感"""
    r, g, b, a = img.split()
    channels = [r, g, b]
    ch = channels[channel_idx]
    shifted = Image.new('L', img.size, 0)
    paste_x = max(0, dx) if dx >= 0 else 0
    paste_y = max(0, dy) if dy >= 0 else 0
    crop_x  = max(0, -dx)
    crop_y  = max(0, -dy)
    crop_w  = img.width  - abs(dx)
    crop_h  = img.height - abs(dy)
    if crop_w > 0 and crop_h > 0:
        shifted.paste(ch.crop((crop_x, crop_y, crop_x+crop_w, crop_y+crop_h)),
                      (paste_x, paste_y))
    channels[channel_idx] = shifted
    return Image.merge('RGBA', channels + [a])


def add_vignette(img, strength):
    """四边暗角，strength 0-255"""
    w, h = img.size
    vign = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(vign)
    steps = 18
    for k in range(steps):
        t   = k / steps
        inset = int(min(w, h) * 0.5 * t)
        alpha = int(strength * (1 - t) ** 2)
        draw.rectangle([inset, inset, w-inset, h-inset],
                       outline=(0, 0, 0, alpha))
    return Image.alpha_composite(img.convert("RGBA"), vign)


def create_dynamic_gif(img_path, text, emotion):
    base_img = Image.open(img_path).convert("RGBA")
    w, h     = base_img.size
    frames   = []
    TOTAL    = 30
    DURATION = 0.04

    for i in range(TOTAL):
        p     = i / (TOTAL - 1)
        f     = base_img.copy()
        x_off, y_off, alpha = 0, 0, 255
        txt_scale = 1.0   # 文字缩放（用于弹入）

        # ── 愤怒：剧烈震颤+对比度+红光+暗角 ─────────────────
        if emotion == "愤怒":
            # 分两段：前半段加速抖动，后半段减缓
            shake = 18 * abs(np.sin(p * np.pi * 5)) * (1 - 0.4 * p)
            x_off = int(random.choice([-1, 1]) * shake)
            y_off = int(random.choice([-1, 1]) * shake * 0.6)
            # 对比度随震颤脉冲
            contrast = 1.0 + 0.5 * abs(np.sin(p * np.pi * 5))
            f = ImageEnhance.Contrast(f.convert("RGB")).enhance(contrast).convert("RGBA")
            red_a = int(100 * abs(np.sin(p * np.pi * 4)))
            f = overlay_color(f, 255, 20, 20, red_a)
            f = add_vignette(f, int(120 * abs(np.sin(p * np.pi * 4))))
            # 文字跟着抖动，渐显
            alpha = int(180 + 75 * abs(np.sin(p * np.pi * 3)))

        # ── 开心：三次弹跳循环+亮度+彩虹暖光 ────────────────
        elif emotion == "开心":
            # 三次弹跳：用 sin 三峰模拟
            bounce_val = abs(np.sin(p * np.pi * 3))
            scale = 1.0 + 0.07 * ease_out_bounce(bounce_val)
            f = zoom_center(f, scale)
            y_off = -int(20 * bounce_val)
            # 亮度提升
            bright = 1.0 + 0.15 * bounce_val
            f = ImageEnhance.Brightness(f.convert("RGB")).enhance(bright).convert("RGBA")
            warm_a = int(30 * bounce_val)
            f = overlay_color(f, 255, 200, 50, warm_a)
            # 文字弹入
            txt_scale = 0.7 + 0.3 * ease_out_bounce(p) if p < 0.5 else 1.0
            alpha = int(255 * min(1.0, p * 4))

        # ── 悲伤：ping-pong去色心跳+蓝调+垂直下坠 ──────────
        elif emotion == "悲伤":
            # ping-pong: 0→1→0 的去色心跳，循环两次
            cycle = abs(np.sin(p * np.pi * 2))
            desat = 0.3 + 0.6 * cycle        # 0.3~0.9 之间脉动
            f = ImageEnhance.Color(f).enhance(1.0 - desat)
            # 蓝调随去色加深
            f = overlay_color(f, 60, 100, 220, int(50 * cycle))
            # 轻微模糊
            blur_r = 0.8 * cycle
            if blur_r > 0.3:
                f = f.filter(ImageFilter.GaussianBlur(radius=blur_r))
            # 缓慢下坠（单向，不循环）
            y_off = int(12 * ease_in_cubic(p))
            alpha = int(255 - 70 * ease_in_cubic(p))
            # 暗角压抑感
            f = add_vignette(f, int(80 * cycle))

        # ── 惊讶：双弹性波+强闪光+文字弹入 ──────────────────
        elif emotion == "惊讶":
            # 第一波大（0~0.4），第二波小（0.5~0.9）
            if p < 0.45:
                t_e = ease_out_elastic(p / 0.45)
                scale = 1.0 + 0.18 * t_e
            else:
                t_e = ease_out_elastic((p - 0.5) / 0.5) if p > 0.5 else 0
                scale = 1.0 + 0.07 * t_e
            f = zoom_center(f, scale)
            # 闪光在最开始
            flash = int(max(0, 90 * (1 - p * 5)))
            if flash > 0:
                f = overlay_color(f, 255, 255, 255, flash)
            # 亮度随弹性脉冲
            f = ImageEnhance.Brightness(f.convert("RGB")).enhance(
                1.0 + 0.2 * abs(np.sin(p * np.pi * 2))).convert("RGBA")
            # 文字：先隐后弹出
            alpha = int(255 * min(1.0, max(0.0, (p - 0.1) * 5)))
            txt_scale = ease_out_elastic(max(0, p - 0.15) / 0.85)

        # ── 尴尬：减幅震荡+泛黄+轻微压扁 ────────────────────
        elif emotion == "尴尬":
            # 减幅震荡：振幅随时间衰减
            decay = np.exp(-3 * p)
            x_off = int(18 * decay * np.sin(p * np.pi * 6))
            # 轻微垂直压扁（越尴尬越缩）
            squish = 1.0 - 0.04 * decay * abs(np.sin(p * np.pi * 6))
            f = f.resize((w, int(h * squish)), Image.Resampling.LANCZOS)
            f = f.crop((0, 0, w, h))
            yellow = int(45 * decay * abs(np.sin(p * np.pi * 6)))
            f = overlay_color(f, 255, 225, 40, yellow)
            # 饱和度轻微下降
            f = ImageEnhance.Color(f.convert("RGB")).enhance(
                1.0 - 0.2 * (1 - decay)).convert("RGBA")
            # 文字跟着晃
            alpha = int(200 + 55 * (1 - decay))

        # ── 未知：Glitch故障效果+随机通道偏移 ────────────────
        elif emotion == "未知":
            # RGB 通道随机偏移，制造数字故障感
            glitch_intensity = int(12 * abs(np.sin(p * np.pi * 7)))
            if glitch_intensity > 2:
                dx_r = random.randint(-glitch_intensity, glitch_intensity)
                dx_g = random.randint(-glitch_intensity // 2, glitch_intensity // 2)
                dy_b = random.randint(-glitch_intensity, glitch_intensity)
                f = glitch_channel(f, dx_r, 0, 0)
                f = glitch_channel(f, dx_g, 0, 1)
                f = glitch_channel(f, 0, dy_b, 2)
            # 随机水平扫描线撕裂
            if random.random() < 0.4:
                tear_y = random.randint(0, h - 1)
                tear_h = random.randint(2, 8)
                tear_x = random.randint(-20, 20)
                strip  = f.crop((0, tear_y, w, min(h, tear_y + tear_h)))
                f_copy = f.copy()
                f_copy.paste(strip, (tear_x, tear_y))
                f = f_copy
            # 颜色在冷暖之间游移
            hue_shift = int(25 * np.sin(p * np.pi * 5))
            f = overlay_color(f, 100 + hue_shift, 50, 200 - hue_shift,
                              int(30 * abs(np.sin(p * np.pi * 5))))
            # 亮度闪烁
            blink = 1.0 + 0.2 * np.sin(p * np.pi * 9)
            f = ImageEnhance.Brightness(f.convert("RGB")).enhance(blink).convert("RGBA")
            alpha = int(255 * abs(np.sin(p * np.pi * 4 + 0.5)))
            alpha = max(120, alpha)

        # ── 绘制文字（带情绪专属入场动画） ───────────────────
        if text:
            # txt_scale < 1 时缩小文字模拟弹入（仅开心/惊讶用到）
            if txt_scale < 0.99:
                scaled_h = int(h * txt_scale) or 1
                scaled_w = int(w * txt_scale) or 1
                small = f.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
                canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                ox = (w - scaled_w) // 2
                oy = (h - scaled_h) // 2
                canvas.paste(small, (ox, oy))
                frame_img = draw_text_on_rgba(canvas, text, x_off, y_off, alpha)
            else:
                frame_img = draw_text_on_rgba(f, text, x_off, y_off, alpha)
        else:
            frame_img = f.convert("RGBA")

        frames.append(np.array(frame_img.convert("RGB")))

    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/meme_{int(time.time())}.gif"
    imageio.mimsave(out_path, frames, duration=DURATION, loop=0)
    return out_path