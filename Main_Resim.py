# -*- coding: utf-8 -*-
"""
Advanced in‑painting pipeline for AUTOMATIC1111 WebUI
----------------------------------------------------
• Random‑sized mask generation + 4 px blur for yumuşak kenar
• Triple‑ControlNet stack: lineart (fire mask) + color reference + depth
• Cinematic prompt tweaks & negative prompt clean‑up
• **NEW:** save_side_by_side → orijinal + inpaint’li kareyi tek PNG’de karşılaştırma
"""

import os
import random
import argparse
import requests
from PIL import Image, ImageDraw, ImageFilter
import base64
from io import BytesIO
import itertools

API_URL = "http://127.0.0.1:7860/sdapi/v1/img2img"

# ---------------------------- yardımcı fonksiyonlar ----------------------------

def encode_image(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def load_prompts(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [p.strip() for p in f if p.strip()]


def create_mask(img: Image.Image):
    """Rastgele boyutta kutu maskesi ve YOLO bbox döndürür."""
    w, h = img.size
    min_frac, max_frac = 0.2, 0.5
    box_w = random.randint(int(w * min_frac), int(w * max_frac))
    box_h = random.randint(int(h * min_frac), int(h * max_frac))
    x0 = random.randint(0, w - box_w)
    y0 = random.randint(0, h - box_h)
    x1, y1 = x0 + box_w, y0 + box_h

    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle([x0, y0, x1, y1], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(4))  # 4 px blur

    # YOLO (cls, cx, cy, w, h) normalised
    cx, cy = ((x0 + x1) / 2) / w, ((y0 + y1) / 2) / h
    yolo = (0, cx, cy, box_w / w, box_h / h)
    return mask, yolo, (x0, y0, x1, y1)


def save_side_by_side(orig: Image.Image, gen: Image.Image, out_path: str):
    """Kaynak ve üretilen görselleri yan yana yapıştırıp kaydeder."""
    w1, h1 = orig.size
    w2, h2 = gen.size
    canvas = Image.new("RGB", (w1 + w2, max(h1, h2)))
    canvas.paste(orig, (0, 0))
    canvas.paste(gen, (w1, 0))
    canvas.save(out_path)


# ---------------------------- API çağrısı ----------------------------

def send_to_api(init_img: Image.Image, ctrl_line: Image.Image, ctrl_color: Image.Image,
                ctrl_depth: Image.Image, mask_img: Image.Image,
                prompt: str, neg_prompt: str):
    payload = {
        "init_images": [encode_image(init_img)],
        "mask": encode_image(mask_img),
        "denoising_strength": 0.6,
        "inpainting_fill": 1,  # latent noise
        "inpaint_full_res": False,
        "sampler_name": "Euler a",
        "steps": 22,
        "cfg_scale": 7,
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "width": init_img.size[0],
        "height": init_img.size[1],
        "controlnet_units": [
            {  # 0 → lineart (ateş maskesi)
                "input_image": encode_image(ctrl_line),
                "module": "lineart",
                "model": "control_v11p_sd15_lineart [43d4be0d]",
                "weight": 1.0,
                "resize_mode": 1,
                "low_vram": True,
            },
            {  # 1 → color reference
                "input_image": encode_image(ctrl_color),
                "module": "reference_only",
                "model": "control_v1p_sd15_color [f19423b0]",
                "weight": 0.8,
                "control_mode": 1,
                "low_vram": True,
            },
            {  # 2 → depth
                "input_image": encode_image(ctrl_depth),
                "module": "depth",
                "model": "control_v11p_sd15_depth [cfd03158]",
                "weight": 0.5,
                "guess_mode": True,
                "low_vram": True,
            },
        ],
    }
    r = requests.post(API_URL, json=payload, timeout=120)
    r.raise_for_status()
    img_b64 = r.json()["images"][0].split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(img_b64)))


# ---------------------------- ana döngü ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--count", type=int, required=True)
    ap.add_argument("-p", "--prompt_file", default="prompt.txt")
    ap.add_argument("-n", "--neg_prompt_file", default="negative_prompt.txt")
    ap.add_argument("-o", "--output_dir", default="output")
    ap.add_argument("-i", "--input_dir", default="input")
    ap.add_argument("-ci", "--control_dir", default="control")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prompts = load_prompts(args.prompt_file)
    neg_prompts = load_prompts(args.neg_prompt_file)

    imgs = [f for f in os.listdir(args.input_dir) if f.lower().endswith(("png", "jpg", "jpeg"))]
    ctrls = [f for f in os.listdir(args.control_dir) if f.lower().endswith(("png", "jpg", "jpeg"))]
    if not imgs or not ctrls:
        print("Gerekli giriş/ control görselleri bulunamadı!")
        return

    for i, name in enumerate(itertools.islice(itertools.cycle(imgs), args.count), 1):
        path = os.path.join(args.input_dir, name)
        with Image.open(path).convert("RGB") as init_img:
            prompt = random.choice(prompts)
            neg = random.choice(neg_prompts) if neg_prompts else ""
            mask, yolo, _ = create_mask(init_img)

            ctrl_name = random.choice(ctrls)
            ctrl_path = os.path.join(args.control_dir, ctrl_name)
            with Image.open(ctrl_path).convert("RGB") as ctrl_img:
                out_img = send_to_api(init_img, ctrl_img, init_img, init_img, mask, prompt, neg)

            out_path = os.path.join(args.output_dir, f"out_{i}.png")
            out_img.save(out_path)

            cmp_path = os.path.join(args.output_dir, f"cmp_{i}.png")
            save_side_by_side(init_img, out_img, cmp_path)

            with open(os.path.join(args.output_dir, f"out_{i}.txt"), "w", encoding="utf-8") as f:
                f.write(" ".join(str(v) for v in yolo))

            print(f"[✓] {out_path}  +  {cmp_path}")


if __name__ == "__main__":
    main()
