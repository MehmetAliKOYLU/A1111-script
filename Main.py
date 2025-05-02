import os
import random
import argparse
import requests
from PIL import Image, ImageDraw
import base64
from io import BytesIO
import itertools

API_URL = "http://127.0.0.1:7860/sdapi/v1/img2img"

def encode_image(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def load_prompts(prompt_file):
    if not os.path.exists(prompt_file):
        return []
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return [p.strip() for p in f.read().splitlines() if p.strip()]

def create_mask(image: Image.Image, prompt: str = "") -> (Image.Image, tuple, tuple):
    width, height = image.size

    # --- rastgele boyut sınırları ---
    MIN_FRAC, MAX_FRAC = 0.20, 0.50          # %20‑%50 arası
    box_w = random.randint(int(width  * MIN_FRAC), int(width  * MAX_FRAC))
    box_h = random.randint(int(height * MIN_FRAC), int(height * MAX_FRAC))
    # ---------------------------------

    # Kutunun sol‑üst köşesini, kutu tamamen kadraj içinde kalacak şekilde seç
    x0 = random.randint(0, width  - box_w)
    y0 = random.randint(0, height - box_h)
    x1, y1 = x0 + box_w, y0 + box_h

    # Maske resmi
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rectangle([x0, y0, x1, y1], fill=255)

    # YOLO formatına dönüştür (class, cx, cy, w, h) — normalize
    center_x = (x0 + x1) / 2 / width
    center_y = (y0 + y1) / 2 / height
    norm_w   = box_w / width
    norm_h   = box_h / height
    yolo_box = (0, center_x, center_y, norm_w, norm_h)

    return mask, yolo_box, (x0, y0, x1, y1)


def send_to_api(init_img, control_img, mask_img, prompt, negative_prompt):
    payload = {
        "init_images": [encode_image(init_img)],
        "mask": encode_image(mask_img),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "denoising_strength": 0.45,
        "inpainting_fill": 1,
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 32,
        "sampler_name": "Euler",
        "steps": 20,
        "cfg_scale": 7,
        "width": 640,
        "height": 640,
        "controlnet_units": [{
            "input_image": encode_image(control_img),
            "module": "lineart",
            "model": "control_v11p_sd15_lineart [43d4be0d]",
            "weight": 1,
            "resize_mode": 1,
            "low_vram": True,
            "guess_mode": False,
            "control_mode": 0,
            "conditioning_scale": 1.0
        }]
    }

    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        result = response.json()
        img_data = result['images'][0]
        if "," in img_data:
            img_base64 = img_data.split(",", 1)[1]
        else:
            img_base64 = img_data
        return Image.open(BytesIO(base64.b64decode(img_base64)))
    else:
        print(f"[!] Error {response.status_code}: {response.text}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--count', type=int, required=True, help='Toplam üretilecek resim sayısı')
    parser.add_argument('-bs', '--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-bc', '--batch_count', type=int, default=1, help='Batch count')
    parser.add_argument('-p', '--prompt_file', type=str, default='prompt.txt', help='Prompt dosyası (virgülle ayrılmış)')
    parser.add_argument('-np', '--neg_prompt_file', type=str, default='negative_prompt.txt', help='Negative prompt dosyası')
    parser.add_argument('-o', '--output_dir', type=str, default='output', help='Çıktı klasörü')
    parser.add_argument('-i', '--input_dir', type=str, default='input', help='Giriş görselleri klasörü')
    parser.add_argument('-ci', '--control_dir', type=str, default='control', help='Control görselleri klasörü')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prompts = load_prompts(args.prompt_file)
    negative_prompts = load_prompts(args.neg_prompt_file)
    
    images = [img for img in os.listdir(args.input_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
    if len(images) == 0:
        print("Giriş klasöründe görsel bulunamadı.")
        return

    control_images = [img for img in os.listdir(args.control_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
    if len(control_images) == 0:
        print("Control klasöründe görsel bulunamadı.")
        return

    for i, image_name in enumerate(itertools.islice(itertools.cycle(images), args.count)):
        image_path = os.path.join(args.input_dir, image_name)
        with Image.open(image_path).convert("RGB") as init_img:
            prompt = random.choice(prompts)
            neg_prompt = random.choice(negative_prompts) if negative_prompts else ""
            mask, yolo_box, box_coords = create_mask(init_img, prompt)
            control_image_name = random.choice(control_images)
            control_image_path = os.path.join(args.control_dir, control_image_name)
            with Image.open(control_image_path).convert("RGB") as control_img:
                output_img = send_to_api(init_img, control_img, mask, prompt, neg_prompt)

            if output_img:
                # Üretilen resmin üzerine kırmızı kutu çiziliyor.
                #draw = ImageDraw.Draw(output_img)
                #draw.rectangle(box_coords, outline="red", width=3)

                output_path = os.path.join(args.output_dir, f"out_{i+1}.png")
                output_img.save(output_path)
                # Kayıt edilen resmin yanına YOLO koordinatlarını içeren txt dosyası oluşturuluyor.
                annotation_path = os.path.join(args.output_dir, f"out_{i+1}.txt")
                with open(annotation_path, "w", encoding="utf-8") as f:
                    f.write(" ".join(str(val) for val in yolo_box))
                print(f"[✓] Kayıt edildi: {output_path} | Annotation: {annotation_path}")
            else:
                print(f"[✗] İşlenemedi: {image_name}")

if __name__ == "__main__":
    main()
