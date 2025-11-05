import os
import io
import random
import numpy as np
from flask import Flask, send_file
from PIL import Image
from fusionbrain_sdk_python import FBClient, PipelineType
import base64

app = Flask(__name__)

FB_API_KEY = os.getenv("FB_API_KEY")
FB_API_SECRET = os.getenv("FB_API_SECRET")

if not FB_API_KEY or not FB_API_SECRET:
    raise RuntimeError("❌ Отсутствуют FB_API_KEY или FB_API_SECRET")

PROMPTS = [
    "pretty dog",
    "retrowave futuristic",
    "retrofuturistic robot",
    "high contrast black and white illustration",
    "dithered portrait",
    "castle in mountains and clouds",
    "cyberpunk city skyline",
    "industrial landscape",
    "fantsy characters in magic forest",
    "steampunk victorian illustration",
    "sovietpunk poster"
]

def get_prompt():
    return f"{random.choice(PROMPTS)}, black and white, high contrast, no color"

def generate_image_from_fusionbrain(prompt: str) -> Image.Image:
    client = FBClient(x_key=FB_API_KEY, x_secret=FB_API_SECRET)
    pipelines = client.get_pipelines_by_type(PipelineType.TEXT2IMAGE)
    pipeline = pipelines[0]
    run = client.run_pipeline(pipeline_id=pipeline.id, prompt=prompt)
    result = client.wait_for_completion(run.uuid, run.status_time)
    if result.status != "DONE":
        raise Exception(f"Генерация провалена: {result.status}")
    img_data = base64.b64decode(result.result.files[0])
    img = Image.open(io.BytesIO(img_data))
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg.convert("L")
    elif img.mode != "L":
        img = img.convert("L")
    return img

def image_to_raw_1bit(img: Image.Image, width=400, height=300) -> bytes:
    img = img.resize((width, height), Image.LANCZOS)
    img_1bit = img.convert("1", dither=Image.FLOYDSTEINBERG)
    arr = np.array(img_1bit, dtype=np.uint8)
    # Опционально: инвертируйте, если e-Ink показывает негатив
    arr = 1 - arr
    packed = np.packbits(arr, axis=1)
    return packed.tobytes()

@app.route("/")
def index():
    return "🖼️ ESP32 e-Ink Server — запросите /image для получения 15000 RAW байт"

@app.route("/image")
def serve_raw_image():
    try:
        prompt = get_prompt()
        img = generate_image_from_fusionbrain(prompt)
        raw_bytes = image_to_raw_1bit(img)

        if len(raw_bytes) != 15000:
            app.logger.error(f"Неверный размер: {len(raw_bytes)} байт")
            return "Internal error: wrong image size", 500

        return send_file(
            io.BytesIO(raw_bytes),
            mimetype="application/octet-stream"
        )
    except Exception as e:
        app.logger.error(f"Ошибка генерации: {str(e)}")
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
