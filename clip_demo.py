#!/usr/bin/env python3
"""
CLIP Vision Demo — uses laion/CLIP-ViT-H-14-laion2B-s32B-b79K locally.

Demonstrates:
  1. Zero-shot image classification
  2. Image-text similarity scoring
  3. Image retrieval from text queries
  4. Text retrieval from image queries

Usage:
    python clip_demo.py test.png
    python clip_demo.py test.png --query "a cat sitting on a couch"
    python clip_demo.py --interactive
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow required: pip install pillow")

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    sys.exit("torch + transformers required: pip install torch transformers")

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = SCRIPT_DIR / "model" / "clip_vision"

LABELS = [
    "a photo of a cat", "a photo of a dog", "a photo of a bird",
    "a photo of a car", "a photo of a building", "a photo of a landscape",
    "a photo of a person", "a photo of food", "a photo of a flower",
    "a painting", "an anime illustration", "a digital art piece",
]


def load_model():
    """Load CLIP-ViT-H-14 from HuggingFace (full model with text encoder)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    print(f"Loading {model_id} ...")
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device).eval()
    print(f"Model loaded on {device} ({dtype})")
    return model, processor, device


@torch.no_grad()
def zero_shot_classify(model, processor, image, labels, device):
    """Classify an image against candidate labels using CLIP zero-shot."""
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    logits = outputs.logits_per_image[0]
    probs = logits.softmax(dim=-1).cpu().numpy()
    results = sorted(zip(labels, probs), key=lambda x: -x[1])
    return results


@torch.no_grad()
def compute_similarity(model, processor, image, text, device):
    """Compute cosine similarity between an image and a text prompt."""
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    similarity = torch.nn.functional.cosine_similarity(image_embeds, text_embeds)
    return float(similarity[0])


@torch.no_grad()
def image_retrieval(model, processor, images, query_text, device, top_k=3):
    """Find the most relevant images for a text query."""
    text_inputs = processor(text=[query_text], return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    text_embeds = model.get_text_features(**text_inputs)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    scores = []
    for img_path in images:
        image = Image.open(img_path).convert("RGB")
        img_inputs = processor(images=image, return_tensors="pt")
        img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
        img_embeds = model.get_image_features(**img_inputs)
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        sim = torch.nn.functional.cosine_similarity(text_embeds, img_embeds)
        scores.append((img_path, float(sim[0])))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


def main():
    parser = argparse.ArgumentParser(description="CLIP Vision Demo")
    parser.add_argument("images", nargs="*", help="Image files to analyze")
    parser.add_argument("--query", "-q", type=str, help="Text query for similarity")
    parser.add_argument("--classify", action="store_true", help="Run zero-shot classification")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--labels", nargs="+", type=str, help="Custom labels for classification")
    args = parser.parse_args()

    model, processor, device = load_model()

    if args.interactive:
        print("\n=== CLIP Vision Interactive Mode ===")
        print("Commands:")
        print("  classify <image>              — zero-shot classification")
        print("  similar <image> <text>        — compute similarity")
        print("  quit                          — exit")
        print()
        while True:
            try:
                line = input("clip> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line or line == "quit":
                break
            parts = line.split(None, 2)
            if len(parts) < 2:
                print("Usage: classify <image> | similar <image> <text>")
                continue
            cmd, img_path = parts[0], parts[1]
            if cmd == "classify" and len(parts) >= 2:
                image = Image.open(img_path).convert("RGB")
                labels = args.labels or LABELS
                results = zero_shot_classify(model, processor, image, labels, device)
                print(f"\nClassification results for {img_path}:")
                for label, prob in results:
                    bar = "█" * int(prob * 40) + "░" * (40 - int(prob * 40))
                    print(f"  {bar} {prob*100:5.1f}%  {label}")
                print()
            elif cmd == "similar" and len(parts) >= 3:
                image = Image.open(img_path).convert("RGB")
                text = parts[2]
                score = compute_similarity(model, processor, image, text, device)
                print(f"Similarity: {score:.4f}")
            else:
                print("Unknown command")
        return

    if not args.images:
        parser.print_help()
        return

    image = Image.open(args.images[0]).convert("RGB")
    labels = args.labels or LABELS

    # Zero-shot classification
    if args.classify or not args.query:
        print(f"\n--- Zero-Shot Classification: {args.images[0]} ---")
        results = zero_shot_classify(model, processor, image, labels, device)
        for label, prob in results:
            bar = "█" * int(prob * 40) + "░" * (40 - int(prob * 40))
            print(f"  {bar} {prob*100:5.1f}%  {label}")

    # Similarity query
    if args.query:
        print(f"\n--- Similarity Query ---")
        print(f"  Image: {args.images[0]}")
        print(f"  Text:  \"{args.query}\"")
        score = compute_similarity(model, processor, image, args.query, device)
        print(f"  Cosine similarity: {score:.4f}")


if __name__ == "__main__":
    main()
