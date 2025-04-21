import os
import random
import argparse
import requests
from PIL import Image
import base64
from io import BytesIO
from math import ceil

API_URL = "http://127.0.0.1:7860/sdapi/v1/img2img"

def encode_image(image_path):
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

def load_prompts(prompt_file):
    if not os.path.exists(prompt_file):
        return []
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return [p.strip() for p in f.read().split(',') if p.strip()]

def distribute_counts(total, num_items):
    per_item = total // num_items
    remainder = total % num_items
    counts = [per_item] * num_items
    for i in range(remainder):
        counts[i] += 1
    return counts

def main():
    # Stil seçeneklerini burada tanımlayın
    style_choices = ['realistic', 'Cinematic', 'Photographic', 'abstract', 'cyberpunk']

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--count', type=int, required=True, help='Toplam üretilecek resim sayısı')
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-bc', '--batch_count', dest='batch_count', type=int, default=1, help='Batch count')
    parser.add_argument('-p', '--prompt_file', type=str, default='prompt.txt', help='Prompt dosyası (virgülle ayrılmış)')
    parser.add_argument('-np', '--neg_prompt_file', type=str, default='negative_prompt.txt', help='Negative prompt dosyası')
    parser.add_argument('-o', '--output_dir', type=str, default='output', help='Çıktı klasörü')
    parser.add_argument('-i', '--input_dir', type=str, default='input', help='Giriş görselleri klasörü')
    parser.add_argument('--style', choices=style_choices, nargs='+', default=['Cinematic'], help="Bir veya daha fazla stil türü seçin")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    prompts = load_prompts(args.prompt_file)
    neg_prompts = load_prompts(args.neg_prompt_file)
    images = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not images:
        print("Input klasöründe görsel bulunamadı.")
        return

    if not prompts:
        print("Prompt dosyasında içerik bulunamadı.")
        return

    counts = distribute_counts(args.count, len(images))

    for img_path, count in zip(images, counts):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        for i in range(count):
            prompt = random.choice(prompts)
            neg_prompt = random.choice(neg_prompts) if neg_prompts else ""
            
            # Stil türlerini seçiyoruz
            selected_styles = ", ".join(args.style)  # Seçilen stilleri birleştiriyoruz

            # Örneğin, prompt'u stil türlerine göre değiştirebiliriz
            prompt = f"{prompt}, {selected_styles}"

            payload = {
                "init_images": [encode_image(img_path)],
                "prompt": prompt,
                "negative_prompt": neg_prompt,
                "batch_size": args.batch_size,
                "n_iter": args.batch_count,
                "seed": -1,
                "denoising_strength": 0.6,
                "sampler_name": "DPM++ 2M Karras",
                "cfg_scale": 7,
                "steps": 30,
                "width": 512,
                "height": 512
            }

            try:
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                r = response.json()

                for j, img_data in enumerate(r['images']):
                    img_bytes = base64.b64decode(img_data.split(",", 1)[0] if "," in img_data else img_data)
                    out_path = os.path.join(args.output_dir, f"{img_name}_{i}_{j}.png")
                    with open(out_path, "wb") as f:
                        f.write(img_bytes)
                    print(f"✓ {out_path} kaydedildi.")
            except Exception as e:
                print(f"Hata oluştu: {e}")
if __name__ == "__main__":
    main()
