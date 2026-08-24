from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
import torch
import io
from typing import Any
import uvicorn
from pyngrok import ngrok
import threading
import time
import os
import json
import argparse

import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Auto-detect device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"\n Using device: {DEVICE}")
print(" Loading Holo1 model...")
model_name = "Hcompany/Holo1-7B"
model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
print("Model loaded successfully!")

app = FastAPI(title="Holo1 General Inference API", version="3.1.0")

def run_inference(messages: list[dict[str, Any]], image: Image.Image) -> str:
    """Run inference with the Holo1 model."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image, padding=True, return_tensors="pt").to(DEVICE)
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

def resize_image_for_model(image: Image.Image):
    image_processor = processor.image_processor
    # Qwen2-VL defaults
    min_pixels = getattr(image_processor, "min_pixels", None) or 256 * 28 * 28
    max_pixels = getattr(image_processor, "max_pixels", None) or 1280 * 28 * 28

    resized_height, resized_width = smart_resize(
        image.height, image.width,
        factor=image_processor.patch_size * image_processor.merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return image.resize(size=(resized_width, resized_height), resample=Image.Resampling.LANCZOS)


@app.post("/generate")
async def generate(image: UploadFile = File(...), messages_json: str = Form(...)):
    """
    Receives an image and messages, then returns the raw model output
    along with timing information for debugging.
    """
    try:
        # Process the image and get dimensions
        image_data = await image.read()
        original_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        original_width, original_height = original_image.size

        pil_image = resize_image_for_model(original_image)
        resized_width, resized_height = pil_image.size

        # Process the prompt from the client
        try:
            messages = json.loads(messages_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for messages_json.")

        # Run and TIME the inference
        print("--- Running Inference ---")
        start_time = time.time()
        raw_model_output = run_inference(messages, pil_image)
        end_time = time.time()
        inference_duration = end_time - start_time
        print(f"--- INFERENCE DURATION: {inference_duration:.4f} seconds ---") # CHECK THIS IN YOUR LOGS


        # Return model output AND dimensions
        return {
            "model_output": raw_model_output,
            "debug_info": { # Add debug info to the response
                "inference_duration_seconds": inference_duration
            },
            "original_dimensions": {
                "width": original_width,
                "height": original_height
            },
            "resized_dimensions": {
                "width": resized_width,
                "height": resized_height
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "Holo1 General Inference API",
        "model": "Hcompany/Holo1-7B",
        "status": "Running",
        "endpoint_info": "Send a POST request to /generate with 'image' and 'messages_json'."
    }


def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Holo1 General Inference API")
    parser.add_argument("--ngrok-token", type=str, default=None, help="Ngrok auth token for public tunnel")
    args = parser.parse_args()

    ngrok_token = args.ngrok_token or os.environ.get("NGROK_AUTHTOKEN", "")

    if ngrok_token:
        try:
            ngrok.kill()
        except Exception:
            pass

        ngrok.set_auth_token(ngrok_token)

        print("Starting Holo1 API Processing Engine...")
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(3)

        public_url = ngrok.connect(8000)
        print("HOLO1 GENERAL INFERENCE API IS LIVE!")
        print(f"Public API URL: {public_url}")
        print("="*70)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
            ngrok.kill()
    else:
        print("Starting Holo1 API locally on http://0.0.0.0:8000")
        print(" Use --ngrok-token to expose publicly.")
        uvicorn.run(app, host="0.0.0.0", port=8000)