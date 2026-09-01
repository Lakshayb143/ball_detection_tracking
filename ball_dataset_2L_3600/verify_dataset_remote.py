import os
import json
import random
import argparse
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont


def load_coco(ann_path):
    with open(ann_path, "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    catid_to_name = {c["id"]: c["name"] for c in categories}
    return images, anns_by_image, catid_to_name


def draw_annotations(img_path, anns, catid_to_name, out_path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for ann in anns:
        bbox = ann["bbox"]
        x, y, w, h = bbox
        x2, y2 = x + w, y + h
        category_name = catid_to_name.get(ann["category_id"], "Unknown")
        label = category_name

        # ---- Text size fix (textbbox instead of textsize) ----
        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]

        # Bounding box
        draw.rectangle([x, y, x2, y2], outline="red", width=2)

        # Label background box
        text_bg = [x, y - text_h - 4, x + text_w + 4, y]
        draw.rectangle(text_bg, fill="red")

        # Label text
        draw.text((x + 2, y - text_h - 2), label, fill="white", font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-path", default="test/_annotations.coco.json",
                        help="Path to COCO annotation JSON")
    parser.add_argument("--img-dir", default="test",
                        help="Directory with images")
    parser.add_argument("--out-dir", default="test_v",
                        help="Where to save visualized images")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="How many random images to visualize")
    parser.add_argument("--seed", type=int, default=55)
    args = parser.parse_args()

    images, anns_by_image, catid_to_name = load_coco(args.ann_path)
    print(f"Total images in split: {len(images)}")

    random.seed(args.seed)
    num = min(args.num_samples, len(images))
    sampled_images = random.sample(images, num)

    os.makedirs(args.out_dir, exist_ok=True)

    for i, img_info in enumerate(sampled_images, 1):
        file_name = img_info["file_name"]
        img_id = img_info["id"]
        img_path = os.path.join(args.img_dir, file_name)
        out_path = os.path.join(args.out_dir, file_name)

        if not os.path.exists(img_path):
            print(f"[WARN] Missing image: {img_path}")
            continue

        anns = anns_by_image.get(img_id, [])
        print(f"[{i}/{num}] {file_name} with {len(anns)} anns -> {out_path}")

        draw_annotations(img_path, anns, catid_to_name, out_path)

    print(f"✔ Done! Visualized images saved to: {args.out_dir}")


if __name__ == "__main__":
    main()

