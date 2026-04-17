from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os, base64, tempfile, json, uuid
from PIL import Image
import inference

app = Flask(__name__, static_folder="static")
CORS(app)

# 启动时预加载文案库
_MEME_TEXTS = {}
if os.path.exists("meme_texts_3000.json"):
    with open("meme_texts_3000.json", "r", encoding="utf-8") as f:
        _raw = json.load(f)
    for d in _raw:
        emo = d.get("emotion", "")
        _MEME_TEXTS.setdefault(emo, []).append(d["text"])

# 用 img_id 映射临时图片路径，避免并发时文件竞争
_temp_images: dict[str, str] = {}

# ── 前端页面 ──────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

# ── 情绪识别 API ──────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "没有上传图片"}), 400

    file = request.files["image"]
    user_text = request.form.get("text", "")

    suffix = os.path.splitext(file.filename)[-1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file.save(tmp.name)
    tmp.close()

    try:
        emotion, probs = inference.predict_emotion(tmp.name, user_text)
        with open(tmp.name, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        # 用 UUID 保存输入图片，供 generate 使用
        img_id = uuid.uuid4().hex[:12]
        saved_path = os.path.join("uploads", f"{img_id}.jpg")
        Image.open(tmp.name).convert("RGB").save(saved_path)
        _temp_images[img_id] = saved_path

        return jsonify({
            "emotion": emotion,
            "probs": probs,
            "preview": img_b64,
            "ext": suffix.lstrip("."),
            "img_id": img_id,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp.name)

# ── 生成GIF API ───────────────────────────────────
@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.json
    emotion  = data.get("emotion", "中性")
    text     = data.get("text", "")
    img_id   = data.get("img_id", "")
    img_path = _temp_images.get(img_id, "temp_input.jpg")

    if not os.path.exists(img_path):
        return jsonify({"error": "请先上传图片"}), 400

    try:
        gif_path = inference.create_dynamic_gif(img_path, text, emotion)
        with open(gif_path, "rb") as f:
            gif_b64 = base64.b64encode(f.read()).decode()
        return jsonify({"gif": gif_b64, "path": gif_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── 文案库 API ────────────────────────────────────
@app.route("/api/texts/<emotion>")
def get_texts(emotion):
    texts = _MEME_TEXTS.get(emotion, ["很有精神！"])
    return jsonify({"texts": texts})

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    print("启动成功：http://localhost:5000")
    app.run(debug=False, port=5000)
