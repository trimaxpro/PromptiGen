#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wd14_tagger.py — Offline, high-speed anime image tagger.

Model : WD v1.4 ConvNeXt Tagger v2 (Bercraft/wd-v1-4-convnext-tagger-v2)
Output: Danbooru-style tags exported as .txt / .json / .csv per image.

Usage:
    python wd14_tagger.py image.png
    python wd14_tagger.py ./folder --recursive --general-threshold 0.35
    python wd14_tagger.py            (opens a file/folder picker GUI)

Author: generated for production use. Python 3.8+.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import logging
import os
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    sys.exit("Pillow is required:  pip install pillow")

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    sys.exit("onnxruntime is required:  pip install onnxruntime  (or onnxruntime-gpu)")

try:
    import torch
    import transformers
    from transformers import CLIPModel, CLIPProcessor
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

# --------------------------------------------------------------------------- #
# Constants & configuration
# --------------------------------------------------------------------------- #

MODEL_REPO_URLS = {
    # Only these two files are required for ONNX inference.
    "model.onnx":
        "https://huggingface.co/Bercraft/wd-v1-4-convnext-tagger-v2/resolve/main/model.onnx?download=true",
    "selected_tags.csv":
        "https://huggingface.co/Bercraft/wd-v1-4-convnext-tagger-v2/resolve/main/selected_tags.csv?download=true",
}

# Default cache dir: local model/ folder next to script, or env override.
_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = Path(
    os.environ.get("WD14_CACHE_DIR",
                   str(_SCRIPT_DIR / "model" / "wd-v1-4-convnext-tagger-v2"))
)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}

# Local CLIP vision model path
CLIP_VISION_DIR = _SCRIPT_DIR / "model" / "clip_vision"

# selected_tags.csv category codes (Danbooru convention used by WD taggers).
CATEGORY_GENERAL = 0
CATEGORY_CHARACTER = 4
CATEGORY_RATING = 9

LOG = logging.getLogger("wd14")

# --------------------------------------------------------------------------- #
# CLIP enhanced tag categories (curated for anime/illustration domain)
# --------------------------------------------------------------------------- #

CLIP_TAG_CATEGORIES = {
    "art_style": [
        "anime illustration", "digital painting", "watercolor painting",
        "oil painting", "sketch", "line art", "cel shade", "realistic",
        "semi realistic", "chibi", "manga style", "pixel art", "vector art",
        "3d render", "photorealistic",
    ],
    "quality": [
        "masterpiece", "high quality", "normal quality", "low quality",
        "blurry", "sharp focus", "detailed background", "high resolution",
        "low resolution", "noise", "clean lines",
    ],
    "lighting": [
        "natural lighting", "studio lighting", "dramatic lighting",
        "backlight", "rim lighting", "soft lighting", "hard lighting",
        "candlelight", "neon lighting", "sunlight", "moonlight",
        "volumetric lighting", "cinematic lighting",
    ],
    "composition": [
        "close up", "medium shot", "long shot", "full body",
        "upper body", "portrait", "dynamic angle", "bird's eye view",
        "worm's eye view", "symmetrical", "centered", "rule of thirds",
        "dutch angle", "wide angle", "telephoto",
    ],
    "mood": [
        "cheerful", "serene", "melancholic", "dark", "romantic",
        "mysterious", "energetic", "calm", "tense", "dreamy",
        "whimsical", "dramatic", "peaceful", "lonely",
    ],
    "color_palette": [
        "vibrant colors", "pastel colors", "monochrome", "warm tones",
        "cool tones", "neon colors", "earth tones", "high contrast",
        "low contrast", "muted colors", "colorful", "sepia",
    ],
    "background": [
        "simple background", "complex background", "gradient background",
        "transparent background", "nature background", "cityscape background",
        "indoor setting", "outer space", "pattern background",
        "solid color background", "blurry background",
    ],
    "render_detail": [
        "highly detailed", "intricate details", "rough texture",
        "smooth surface", "glossy", "matte", "shiny", "metallic",
        "soft shading", "hard shading", "cel shading",
    ],
}


@dataclass
class TaggerConfig:
    """Runtime configuration for tagging behaviour."""
    general_threshold: float = 0.35     # confidence cutoff for general tags
    character_threshold: float = 0.85   # confidence cutoff for character tags
    replace_underscores: bool = False   # "long_hair" -> "long hair" (optional)
    include_rating: bool = True         # include rating (safe/questionable/...) in JSON
    batch_size: int = 4                 # inference batch size
    loader_threads: int = 4             # image preprocessing threads
    export_txt: bool = True
    export_json: bool = True
    export_csv: bool = True
    enable_clip: bool = False           # enable CLIP enhanced tagging
    clip_threshold: float = 0.22        # cosine similarity cutoff for CLIP tags
    clip_top_k: int = 3                 # top-k tags per category
    enable_caption: bool = False


@dataclass
class TagResult:
    """Tagging result for a single image."""
    path: Path
    general: List[Tuple[str, float]] = field(default_factory=list)   # (tag, confidence)
    character: List[Tuple[str, float]] = field(default_factory=list)
    rating: List[Tuple[str, float]] = field(default_factory=list)
    clip_enhanced: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Model download & caching
# --------------------------------------------------------------------------- #

def _download_file(url: str, dest: Path) -> None:
    """Stream a URL to disk atomically (tmp file + rename) with progress logging."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    LOG.info("Downloading %s -> %s", url.split("?")[0], dest)
    req = urllib.request.Request(url, headers={"User-Agent": "wd14-tagger/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        read = 0
        last_pct = -1
        while True:
            chunk = resp.read(1 << 20)  # 1 MiB chunks
            if not chunk:
                break
            fh.write(chunk)
            read += len(chunk)
            if total:
                pct = int(read * 100 / total)
                if pct != last_pct and pct % 10 == 0:
                    LOG.info("  %s: %d%% (%.1f MiB)", dest.name, pct, read / 2**20)
                    last_pct = pct
    tmp.replace(dest)  # atomic on same filesystem
    LOG.info("Saved %s (%.1f MiB)", dest.name, dest.stat().st_size / 2**20)


def ensure_model(cache_dir: Path, offline: bool = False) -> Tuple[Path, Path]:
    """
    Ensure model.onnx and selected_tags.csv exist in cache_dir.
    Downloads on first launch; afterwards the app runs fully offline.
    Returns (model_path, tags_csv_path).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in MODEL_REPO_URLS.items():
        dest = cache_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            LOG.debug("Cache hit: %s", dest)
        else:
            if offline:
                raise FileNotFoundError(
                    f"Missing {dest} and --offline was requested. "
                    f"Place the file there manually or run once with network access.")
            for attempt in range(1, 4):  # simple retry with backoff
                try:
                    _download_file(url, dest)
                    break
                except Exception as exc:
                    LOG.warning("Download attempt %d failed: %s", attempt, exc)
                    if attempt == 3:
                        raise
                    time.sleep(2 * attempt)
        paths[name] = dest
    return paths["model.onnx"], paths["selected_tags.csv"]


# --------------------------------------------------------------------------- #
# Label loading
# --------------------------------------------------------------------------- #

def load_labels(csv_path: Path) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse selected_tags.csv.
    Returns (names, rating_idx, general_idx, character_idx) where the index
    arrays map into the model's output vector.
    """
    names: List[str] = []
    categories: List[int] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            names.append(row["name"])
            categories.append(int(row["category"]))
    cats = np.asarray(categories)
    return (
        names,
        np.where(cats == CATEGORY_RATING)[0],
        np.where(cats == CATEGORY_GENERAL)[0],
        np.where(cats == CATEGORY_CHARACTER)[0],
    )


# --------------------------------------------------------------------------- #
# Image preprocessing (exactly as the WD v1.4 taggers expect)
# --------------------------------------------------------------------------- #

def preprocess_image(path: Path, target_size: int) -> np.ndarray:
    """
    WD v1.4 preprocessing:
      1. Load, flatten alpha onto white, convert to RGB.
      2. Pad to square with white background (no cropping, aspect preserved).
      3. Resize to target_size x target_size (bicubic).
      4. Convert RGB -> BGR (the model was trained on BGR input).
      5. float32, raw 0-255 range (NO normalization), NHWC layout.
    """
    with Image.open(path) as img:
        # Flatten transparency onto white — matters a lot for PNG line art.
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        img = img.convert("RGB")

        # Pad to square with white.
        w, h = img.size
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))

        # Resize to model input resolution.
        canvas = canvas.resize((target_size, target_size), Image.BICUBIC)

    arr = np.asarray(canvas, dtype=np.float32)  # HWC, RGB, 0..255
    arr = arr[:, :, ::-1]                       # RGB -> BGR
    return arr                                   # (H, W, 3) float32


# --------------------------------------------------------------------------- #
# Tagger engine
# --------------------------------------------------------------------------- #

class WD14Tagger:
    """ONNX Runtime wrapper: session setup, batched inference, tag extraction."""

    def __init__(self, model_path: Path, tags_csv: Path, cfg: TaggerConfig):
        self.cfg = cfg
        self.names, self.rating_idx, self.general_idx, self.character_idx = load_labels(tags_csv)

        # Provider selection: CUDA if available, CPU fallback. GPU memory is
        # capped via arena settings to stay lightweight alongside other apps.
        providers = []
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            providers.append(("CUDAExecutionProvider", {
                "arena_extend_strategy": "kSameAsRequested",   # frugal GPU allocation
                "cudnn_conv_algo_search": "HEURISTIC",         # faster startup
            }))
        providers.append("CPUExecutionProvider")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3  # silence ORT chatter

        LOG.info("Creating ONNX session (providers=%s)...", [p[0] if isinstance(p, tuple) else p for p in providers])
        print("PHASE:WD14_MODEL_LOADING", flush=True)
        self.session = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
        LOG.info("Active providers: %s", self.session.get_providers())

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # Shape is (batch, H, W, 3); H is 448 for ConvNeXt v2 taggers.
        self.target_size = int(inp.shape[1]) if isinstance(inp.shape[1], int) else 448
        self.output_name = self.session.get_outputs()[0].name
        LOG.info("Model input: %s %s -> resolution %dpx", inp.name, inp.shape, self.target_size)

        n_out = len(self.names)
        LOG.info("Loaded %d labels (%d rating / %d general / %d character)",
                 n_out, len(self.rating_idx), len(self.general_idx), len(self.character_idx))
        print("PHASE:WD14_MODEL_READY", flush=True)

    # ---- inference ------------------------------------------------------- #

    def infer_batch(self, batch: np.ndarray) -> np.ndarray:
        """Run one forward pass. batch: (N, H, W, 3) float32 BGR."""
        return self.session.run([self.output_name], {self.input_name: batch})[0]

    def probs_to_result(self, path: Path, probs: np.ndarray) -> TagResult:
        """Apply thresholds, dedupe, sort by confidence descending."""
        res = TagResult(path=path)
        seen: set = set()

        def collect(indices: np.ndarray, threshold: Optional[float]) -> List[Tuple[str, float]]:
            out = []
            for i in indices:
                conf = float(probs[i])
                if threshold is not None and conf < threshold:
                    continue
                name = self.names[i]
                if name in seen:        # duplicate guard
                    continue
                seen.add(name)
                out.append((name, conf))
            out.sort(key=lambda t: t[1], reverse=True)
            return out

        res.rating = collect(self.rating_idx, None)  # ratings: keep all, sorted
        res.character = collect(self.character_idx, self.cfg.character_threshold)
        res.general = collect(self.general_idx, self.cfg.general_threshold)
        return res


# --------------------------------------------------------------------------- #
# Key mapping: transformers CLIP format → open_clip format
# --------------------------------------------------------------------------- #
# CLIP Enhanced Tagger (zero-shot classification + captioning)
# --------------------------------------------------------------------------- #

class CLIPTagger:
    """CLIP-based zero-shot classification.

    Loads openai/clip-vit-base-patch16 from local cache for fast GPU inference.
    Captioning is handled externally via OpenRouter vision API.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        LOG.info("CLIPTagger device=%s dtype=%s", self.device, self.dtype)

        # --- CLIP model ---
        fast_id = "openai/clip-vit-base-patch16"
        large_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        print(f"CLIP_MODEL:LOADING:{self.device}", flush=True)

        try:
            LOG.info("Loading local CLIP model from %s", CLIP_VISION_DIR)
            self.clip_processor = CLIPProcessor.from_pretrained(large_id)
            self.clip_model = CLIPModel.from_pretrained(
                str(CLIP_VISION_DIR), torch_dtype=self.dtype
            ).to(self.device)
            LOG.info("Successfully loaded local CLIP model.")
        except Exception as e:
            LOG.warning("Local load failed (%s), falling back to online/cache", e)
            try:
                LOG.info("Loading CLIP from %s", fast_id)
                self.clip_processor = CLIPProcessor.from_pretrained(fast_id, local_files_only=True)
                self.clip_model = CLIPModel.from_pretrained(
                    fast_id, torch_dtype=self.dtype, local_files_only=True,
                ).to(self.device)
            except Exception:
                LOG.warning("Local cache miss for %s, trying download", fast_id)
                try:
                    self.clip_processor = CLIPProcessor.from_pretrained(fast_id)
                    self.clip_model = CLIPModel.from_pretrained(
                        fast_id, torch_dtype=self.dtype,
                    ).to(self.device)
                except Exception:
                    self.clip_processor = CLIPProcessor.from_pretrained(large_id)
                    self.clip_model = CLIPModel.from_pretrained(
                        large_id, torch_dtype=self.dtype,
                    ).to(self.device)
        self.clip_model.eval()
        print("CLIP_MODEL:CLIP_READY", flush=True)

        self.tag_categories = CLIP_TAG_CATEGORIES

    @torch.no_grad()
    def _encode_image(self, pil_image) -> torch.Tensor:
        """Encode a PIL image into a normalized feature vector."""
        inputs = self.clip_processor(images=pil_image, return_tensors="pt").to(self.device)
        outputs = self.clip_model.get_image_features(**inputs)
        features = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output
        return features / features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of text strings into normalized feature vectors."""
        inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.clip_model.get_text_features(**inputs)
        features = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output
        return features / features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def classify(self, pil_image, category: str, top_k: int = 3,
                 threshold: float = 0.22) -> List[Tuple[str, float]]:
        """Score image against a category's tag list using CLIP cosine similarity."""
        labels = self.tag_categories.get(category, [])
        if not labels:
            return []

        image_features = self._encode_image(pil_image)
        text_features = self._encode_text(labels)

        # Cosine similarity between normalized CLIP features
        similarity = (image_features @ text_features.T).squeeze(0)
        scores = similarity.cpu().numpy()

        results = []
        for i, label in enumerate(labels):
            conf = float(scores[i])
            if conf >= threshold:
                results.append((label, conf))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @torch.no_grad()
    def classify_all(self, pil_image, top_k: int = 3,
                     threshold: float = 0.15) -> Dict[str, List[Tuple[str, float]]]:
        """Run CLIP classification across all tag categories."""
        return {
            cat: self.classify(pil_image, cat, top_k=top_k, threshold=threshold)
            for cat in self.tag_categories
        }


# --------------------------------------------------------------------------- #
# Export helpers
# --------------------------------------------------------------------------- #

def _fmt_tag(tag: str, cfg: TaggerConfig) -> str:
    """Optional underscore replacement; escapes '(' ')' like common trainers do."""
    if cfg.replace_underscores:
        return tag.replace("_", " ").replace("(", r"\(").replace(")", r"\)")
    return tag


def export_result(res: TagResult, cfg: TaggerConfig) -> None:
    """Write .txt / .json / .csv next to the source image."""
    stem = res.path.with_suffix("")
    all_tags = res.character + res.general  # characters first, then general

    if cfg.export_txt:
        # Comma-separated Danbooru-style caption, sorted by confidence.
        line = ", ".join(_fmt_tag(t, cfg) for t, _ in all_tags)
        Path(f"{stem}.txt").write_text(line, encoding="utf-8")

    if cfg.export_json:
        payload = {
            "image": res.path.name,
            "model": "wd-v1-4-convnext-tagger-v2",
            "general_threshold": cfg.general_threshold,
            "character_threshold": cfg.character_threshold,
            "rating": {t: round(c, 4) for t, c in res.rating} if cfg.include_rating else {},
            "character": {_fmt_tag(t, cfg): round(c, 4) for t, c in res.character},
            "general": {_fmt_tag(t, cfg): round(c, 4) for t, c in res.general},
        }
        if cfg.enable_clip and res.clip_enhanced:
            clip_out = {}
            for cat, tags in res.clip_enhanced.items():
                if tags:
                    clip_out[cat] = {t: round(c, 4) for t, c in tags}
            if clip_out:
                payload["clip_enhanced"] = clip_out
        Path(f"{stem}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

    if cfg.export_csv:
        with open(f"{stem}.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["tag", "category", "confidence"])
            for t, c in res.rating:
                writer.writerow([t, "rating", f"{c:.4f}"])
            for t, c in res.character:
                writer.writerow([_fmt_tag(t, cfg), "character", f"{c:.4f}"])
            for t, c in res.general:
                writer.writerow([_fmt_tag(t, cfg), "general", f"{c:.4f}"])


# --------------------------------------------------------------------------- #
# Input collection & pipeline
# --------------------------------------------------------------------------- #

def collect_images(inputs: Sequence[str], recursive: bool) -> List[Path]:
    """Expand files/folders into a deduplicated, sorted list of image paths."""
    found: List[Path] = []
    for raw in inputs:
        p = Path(raw).expanduser()
        if p.is_file():
            if p.suffix.lower() in SUPPORTED_EXTS:
                found.append(p)
            else:
                LOG.warning("Skipping unsupported file: %s", p)
        elif p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            found.extend(f for f in it if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS)
        else:
            LOG.warning("Input not found: %s", p)
    # Dedupe while preserving determinism.
    uniq = sorted(set(found))
    LOG.info("Collected %d image(s).", len(uniq))
    return uniq


def run_pipeline(images: List[Path], tagger: WD14Tagger, cfg: TaggerConfig) -> List[TagResult]:
    """
    Producer/consumer pipeline:
      * ThreadPoolExecutor preprocesses images concurrently (I/O + PIL bound).
      * Main thread assembles fixed-size batches and runs GPU/CPU inference.
    """
    results: List[TagResult] = []
    total = len(images)
    done = 0
    lock = threading.Lock()

    def load(path: Path):
        try:
            return path, preprocess_image(path, tagger.target_size), None
        except Exception as exc:  # corrupt/unreadable image — never crash the batch
            return path, None, str(exc)

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.loader_threads) as pool:
        pending_paths: List[Path] = []
        pending_arrays: List[np.ndarray] = []

        def flush():
            """Run inference on the accumulated batch and export results."""
            nonlocal done
            if not pending_arrays:
                return
            batch = np.stack(pending_arrays)  # (N, H, W, 3)
            probs = tagger.infer_batch(batch)
            for path, row in zip(pending_paths, probs):
                res = tagger.probs_to_result(path, row)
                try:
                    export_result(res, cfg)
                except Exception as exc:
                    res.error = f"export failed: {exc}"
                    LOG.error("Export failed for %s: %s", path, exc)
                results.append(res)
                with lock:
                    done += 1
                    LOG.info("[%d/%d] %s  (%d char, %d general)",
                             done, total, path.name, len(res.character), len(res.general))
                    print(f"WD14_PROGRESS:{done}:{total}:{path.name}", flush=True)
            pending_paths.clear()
            pending_arrays.clear()

        for path, arr, err in pool.map(load, images):
            if err is not None:
                LOG.error("Failed to load %s: %s", path, err)
                results.append(TagResult(path=path, error=err))
                done += 1
                continue
            pending_paths.append(path)
            pending_arrays.append(arr)
            if len(pending_arrays) >= cfg.batch_size:
                flush()
        flush()  # remainder

    dt = time.time() - t0
    ok = sum(1 for r in results if r.error is None)
    LOG.info("Finished: %d ok, %d failed, %.1fs (%.2f img/s)",
             ok, total - ok, dt, total / dt if dt else 0.0)
    print("PHASE:WD14_DONE", flush=True)
    return results


# --------------------------------------------------------------------------- #
# CLIP combined pipeline (runs after WD14 for each image)
# --------------------------------------------------------------------------- #

def run_clip_pipeline(results: List[TagResult], clip_tagger: CLIPTagger,
                      cfg: TaggerConfig) -> List[TagResult]:
    """Run CLIP zero-shot classification on WD14 results. Captioning is via OpenRouter."""
    total = len(results)
    for i, res in enumerate(results):
        if res.error is not None:
            continue
        try:
            pil_image = Image.open(res.path).convert("RGB")
            if cfg.enable_clip:
                clip_tags = clip_tagger.classify_all(
                    pil_image, top_k=cfg.clip_top_k, threshold=cfg.clip_threshold)
                res.clip_enhanced = {k: v for k, v in clip_tags.items() if v}
            LOG.info("[%d/%d] %s — CLIP done", i + 1, total, res.path.name)
            print(f"CLIP_PROGRESS:{i + 1}:{total}:{res.path.name}", flush=True)
        except Exception as exc:
            LOG.warning("CLIP failed for %s: %s", res.path, exc)
            print(f"CLIP_ERROR:{res.path.name}:{exc}", flush=True)
    print("CLIP_PROGRESS:DONE", flush=True)
    return results


# --------------------------------------------------------------------------- #
# Optional GUI picker (used when launched with no CLI inputs)
# --------------------------------------------------------------------------- #

def gui_pick_inputs() -> List[str]:
    """Minimal tkinter picker: choose files, or cancel to choose a folder."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        LOG.error("tkinter unavailable; pass image paths on the command line.")
        return []
    root = tk.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(
        title="Select images (Cancel to pick a folder instead)",
        filetypes=[("Images", " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTS)))])
    if files:
        root.destroy()
        return list(files)
    folder = filedialog.askdirectory(title="Select a folder of images")
    root.destroy()
    return [folder] if folder else []


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Offline anime image tagger (WD v1.4 ConvNeXt Tagger v2, ONNX).")
    ap.add_argument("inputs", nargs="*", help="Image files and/or folders.")
    ap.add_argument("-r", "--recursive", action="store_true", help="Recurse into folders.")
    ap.add_argument("--general-threshold", type=float, default=0.35)
    ap.add_argument("--character-threshold", type=float, default=0.85)
    ap.add_argument("--replace-underscores", action="store_true",
                    help="Output 'long hair' instead of 'long_hair'.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4, help="Image loader threads.")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_CACHE_DIR,
                    help="Model cache directory.")
    ap.add_argument("--offline", action="store_true",
                    help="Never touch the network; fail if model is not cached.")
    ap.add_argument("--no-txt", action="store_true")
    ap.add_argument("--no-json", action="store_true")
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--clip", action="store_true", help="Enable CLIP enhanced tagging (requires torch+transformers)")
    ap.add_argument("--clip-threshold", type=float, default=0.22, help="CLIP cosine similarity threshold (default: 0.22)")
    ap.add_argument("--clip-top-k", type=int, default=3, help="Top-k CLIP tags per category (default: 3)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    cfg = TaggerConfig(
        general_threshold=args.general_threshold,
        character_threshold=args.character_threshold,
        replace_underscores=args.replace_underscores,
        batch_size=max(1, args.batch_size),
        loader_threads=max(1, args.threads),
        export_txt=not args.no_txt,
        export_json=not args.no_json,
        export_csv=not args.no_csv,
        enable_clip=args.clip,
        clip_threshold=args.clip_threshold,
        clip_top_k=args.clip_top_k,
    )

    inputs = args.inputs or gui_pick_inputs()
    if not inputs:
        LOG.error("No inputs provided.")
        return 2

    images = collect_images(inputs, args.recursive)
    if not images:
        LOG.error("No supported images found.")
        return 2

    try:
        model_path, tags_csv = ensure_model(args.model_dir, offline=args.offline)
    except Exception as exc:
        LOG.error("Model setup failed: %s", exc)
        return 1

    # Step 1: CLIP vision analysis first
    clip_results = [TagResult(path=p) for p in images]
    if cfg.enable_clip:
        if HAS_CLIP:
            try:
                LOG.info("Initializing CLIP models...")
                clip_tagger = CLIPTagger()
                clip_results = run_clip_pipeline(clip_results, clip_tagger, cfg)
            except Exception as exc:
                LOG.warning("CLIP init failed: %s", exc)
                print(f"CLIP_MODEL:UNAVAILABLE", flush=True)
        else:
            LOG.warning("CLIP disabled — torch/transformers not installed. Run: pip install torch transformers")
            print("CLIP_MODEL:UNAVAILABLE", flush=True)

    # Step 2: WD14 tagging
    tagger = WD14Tagger(model_path, tags_csv, cfg)
    results = run_pipeline(images, tagger, cfg)

    # Merge CLIP data into WD14 results
    clip_map = {r.path: r.clip_enhanced for r in clip_results}
    for res in results:
        if res.path in clip_map:
            res.clip_enhanced = clip_map[res.path]

    # Export combined results
    for res in results:
        if res.error is None:
            try:
                export_result(res, cfg)
            except Exception as exc:
                LOG.warning("Re-export failed for %s: %s", res.path, exc)

    return 0 if all(r.error is None for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
