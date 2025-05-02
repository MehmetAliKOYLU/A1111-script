# -*- coding: utf-8 -*-
"""
Advanced in‑painting pipeline for AUTOMATIC1111 WebUI
----------------------------------------------------
• Random‑sized mask generation
• Mask blur + mid‑strength denoise for softer edges
• 3× ControlNet stack → lineart (fire mask) + color reference + depth
• Cinematic prompt tweaks & negative prompt clean‑up
Tested on WebUI v1.8.1 + ControlNet v1.2.2 (API).
"""

import os
import random
import argparse
import requests
from PIL import Image, ImageDraw
import base64
from io import BytesIO
import itertools

API_URL = "http://127.0.0.1:7860/sdapi/v1/img2img"

# ——— Hyper‑parameters ————————————————————————————————————————————
MASK_MIN_FRAC, MASK_MAX_FRAC = 0.20, 0.50   # %20‑%50 arası rastgele kutu
MASK_BLUR_PX = 4                            # Inpaint mask blur (WebUI API)
DENOISE_STRENGTH = 0.60                     # Biraz daha özgür in‑paint
INPAINT_FULL_RES = False                    # Parça inpaint + VRAM tasarrufu
LINEART_MODEL = "control_v11p_sd15_lineart [43d4be0d]"
COLOR_MODEL   = "control_v1p_sd15_color [c86c5ea7]"
DEPTH_MODEL   = "control_v11p_sd15_depth [cfd03158]"
PROMPT_SUFFIX = (", warm orange glow reflected on nearby surfaces, realistic soft shadows, ""physically accurate light falloff, global cinematic color grading")
NEG_SUFFIX = ", oversaturated colors, posterization, pure orange blob, glowing mush"
# ————————————————————————————————————————————————————————————————

def encode_image(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def load_prompts(prompt_file):
    if not os.path.exists(prompt_file):
        return []
    with open(prompt_file, "r", encoding="utf-8") as f:
        return [p.strip() for p in f.read().splitlines() if p.strip()]


def create_mask(image: Image.Image) -> tuple[Image.Image, tuple, tuple]:
    """Rastgele boyutlu dikdörtgen maske (YOLO box + bbox)"""
    w, h = image.size
    box_w = random.randint(int(w * MASK_MIN_FRAC), int(w * MASK_MAX_FRAC))
    box_h = random.randint(int(h * MASK_MIN_FRAC), int(h * MASK_MAX_FRAC))
    x0 = random.randint(0, w - box_w)
    y0 = random.randint(0, h - box_h)
    x1, y1 = x0 + box_w, y0 + box_h

    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle([x0, y0, x1, y1], fill=255)

    # YOLO format (class cx cy w h) normalized
    yolo_box = (
        0,
        ((x0 + x1) / 2) / w,
        ((y0 + y1) / 2) / h,
        box_w / w,
        box_h / h,
    )
    return mask, yolo_box, (x0, y0, x1, y1)


def build_controlnet_units(init_img_b64: str, fire_mask_b64: str) -> list[dict]:
    """3‑lü ControlNet bloğu: Lineart (ateş), Color, Depth"""
    return [
        # 1) Fire mask → Lineart (kuvvetli)
        {
            "input_image": fire_mask_b64,
            "module": "lineart",
            "model": LINEART_MODEL,
            "weight": 1.0,
            "resize_mode": 1,
            "low_vram": True,
            "guess_mode": False,
            "control_mode": 0,
            "conditioning_scale": 1.0,
        },
        # 2) Arka plan → Color reference‑only (stil/LUT aktarımı)
        {
            "input_image": init_img_b64,
            "module": "reference_only",
            "model": COLOR_MODEL,
            "weight": 0.8,
            "resize_mode": 0,
            "low_vram": True,
            "guess_mode": False,
            "control_mode": 0,
            "conditioning_scale": 1.0,
        },
        # 3) Arka plan → Depth (ışık & gölge hizalama)
        {
            "input_image": init_img_b64,
            "module": "depth",
            "model": DEPTH_MODEL,
            "weight": 0.5,
            "resize_mode": 0,
            "low_vram": True,
            "guess_mode": True,
            "control_mode": 0,
            "conditioning_scale": 1.0,
        },
    ]


def send_to_api(init_img: Image.Image, fire_mask: Image.Image, prompt: str, negative_prompt: str):
    init_b64 = encode_image(init_img)
    mask_b64 = encode_image(fire_mask)
    payload = {
        "init_images": [init_b64],
        "mask": mask_b64,
        "mask_blur": MASK_BLUR_PX,
        "denoising_strength": DENOISE_STRENGTH,
        "inpainting_fill": 1,
        "inpaint_full_res": INPAINT_FULL_RES,
        "inpaint_full_res_padding": 32,
        "prompt": prompt + PROMPT_SUFFIX,
        "negative_prompt": (negative_prompt + NEG_SUFFIX).strip(", "),
        "sampler_name": "Euler",
        "steps": 20,
        "cfg_scale": 7,
        "width": init_img.size[0],
        "height": init_img.size[1],
        "controlnet_units": build_controlnet_units(init_b64, mask_b64),
    }

    resp = requests.post(API_URL, json=payload, timeout=300)
    resp.raise_for_status()
    img_data = resp.json()["images"][0]
    img_b64 = img_data.split(",", 1)[1] if "," in img_data else img_data
    return Image.open(BytesIO(base64.b64decode(img_b64)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--count", type=int, required=True)
    ap.add_argument("-p", "--prompt_file", default="prompt.txt")
    ap.add_argument("-np", "--neg_prompt_file", default="negative_prompt.txt")
    ap.add_argument("-o", "--output_dir", default="output")
    ap.add_argument("-i", "--input_dir", default="input")
    ap.add_argument("-ci", "--control_dir", default="control")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prompts = load_prompts(args.prompt_file) or ["fire"]
    neg_prompts = load_prompts(args.neg_prompt_file)

    imgs = [f for f in os.listdir(args.input_dir) if f.lower().endswith(("png", "jpg", "jpeg"))]
    ctrls = [f for f in os.listdir(args.control_dir) if f.lower().endswith(("png", "jpg", "jpeg"))]
    if not imgs or not ctrls:
        raise SystemExit("Input / Control klasörleri boş!")

    for i, img_name in enumerate(itertools.islice(itertools.cycle(imgs), args.count), 1):
        with Image.open(os.path.join(args.input_dir, img_name)).convert("RGB") as bg_img:
            mask, yolo_box, _ = create_mask(bg_img)
            prompt = random.choice(prompts)
            neg = random.choice(neg_prompts) if neg_prompts else ""

            out_img = send_to_api(bg_img, mask, prompt, neg)
            out_path = os.path.join(args.output_dir, f"out_{i}.png")
            out_img.save(out_path)
            with open(os.path.join(args.output_dir, f"out_{i}.txt"), "w", encoding="utf-8") as f:
                f.write(" ".join(map(str, yolo_box)))
            print(f"[✓] Kayıt: {out_path}")


if __name__ == "__main__":
    main()
