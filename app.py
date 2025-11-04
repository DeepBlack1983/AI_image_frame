import os
import io
import random
from flask import Flask, send_file
from PIL import Image
import numpy as np
from fusionbrain_sdk_python import FBClient, PipelineType
import base64

# --- Настройка ---
app = Flask(__name__)

# Получаем ключи из переменных окружения (будем задавать в Render)
FB_API_KEY = os.getenv("FB_API_KEY")
FB_API_SECRET = os.getenv("FB_API_SECRET")

if not FB_API_KEY or not FB_API_SECRET:
    raise RuntimeError("❌ Отсутствуют FB_API_KEY или FB_API_SECRET в переменных окружения!")

PROMPTS = [
    "abstract geometric pattern",
    "minimalist line art",
    "monochrome ink sketch",
    "high contrast black and white illustration",
    "dithered portrait",
    "zen circle on white background"
]

def get_prompt():
    return f"{random.choice(PROMPTS)}, black and white, high contrast, no color"

def generate_image_from_fusionbrain(prompt: str) -> Image.Image:
    print(f"🎨 Запрос: {prompt}")
    client = FBClient(x_key=FB_API_KEY, x_secret=FB_API_SECRET)

    pipelines = client.get_pipelines_by_type(PipelineType.TEXT2IMAGE)
    if not pipelines:
        raise Exception("Нет доступных пайплайнов")

    pipeline = pipelines[0]
    print(f"⚙️ Пайплайн: {pipeline.name}")

    run = client.run_pipeline(pipeline_id=pipeline.id, prompt=prompt)
    result = client.wait_for_completion(run.uuid, run.status_time)

    if result.status != "DONE":
        raise Exception(f"Генерация не удалась: {result.status}")

    img_data = base64.b64decode(result.result.files[0])
    img = Image.open(io.BytesIO(img_data))

    # Обработка альфа-канала → grayscale
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg.convert("L")
    elif img.mode != "L":
        img = img.convert("L")

    return img

def dither_to_1bit_png(img: Image.Image, width=400, height=300) -> bytes:
    img = img.resize((width, height), Image.LANCZOS)
    img_1bit = img.convert("1", dither=Image.FLOYDSTEINBERG)
    buf = io.BytesIO()
    img_1bit.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

# --- Роуты ---
@app.route("/")
def index():
    return """
    <h2>🖼️ FusionBrain e-Ink Image Server</h2>
    <p>Перейдите по <a href="/image">/image</a>, чтобы увидеть сгенерированное 1-битное изображение.</p>
    """

@app.route("/image")
def serve_image():
    try:
        prompt = get_prompt()
        img = generate_image_from_fusionbrain(prompt)
        png_bytes = dither_to_1bit_png(img)
        return send_file(
            io.BytesIO(png_bytes),
            mimetype="image/png"
        )
    except Exception as e:
        return f"<h3>❌ Ошибка:</h3><pre>{str(e)}</pre>", 500

# --- Запуск ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
