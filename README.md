# PromptiGen

**Local AI Prompt Generator** — A high-performance, fully offline desktop application designed for AI image tagging, analysis, and automated Stable Diffusion prompt engineering.

PromptiGen runs local neural network models to scan illustration/anime images and generate ready-to-use Stable Diffusion prompts. By combining **WD v1.4 ConvNeXt Tagger v2** (Danbooru tags) with zero-shot **CLIP-ViT-H-14** vision models (detecting art style, mood, composition, lighting, and textures) and optional **OpenRouter** AI model optimization, it helps you construct high-quality, professional prompts automatically from visual inputs.

---

## Features

- **WD14 Tagging** — Offline Danbooru-style tagging using WD v1.4 ConvNeXt Tagger v2 (ONNX)
- **CLIP Vision** — Zero-shot image classification for art style, lighting, mood, composition, etc.
- **AI Enhancement** — OpenRouter-powered prompt optimization (free API, multiple models)
- **Uncensored Mode** — NSFW-capable prompt generation with appropriate models
- **Beautiful UI** — Dark neon Gradio interface with live progress tracking
- **Batch Processing** — Tag multiple images at once
- **Fully Offline** — Models run locally after initial download

## Requirements

- **Python 3.10+**
- **Windows** (batch files provided)
- **~3 GB disk space** for AI models
- **NVIDIA GPU** recommended (works on CPU too, just slower)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/trimaxpro/PromptiGen.git
cd PromptiGen
```

### 2. Install

Double-click **`install.bat`** or run in terminal:

```bash
install.bat
```

This will:
- Create a Python virtual environment
- Install all dependencies (numpy, pillow, onnxruntime, gradio, torch, transformers, etc.)
- Download AI models from HuggingFace (~2.9 GB):
  - CLIP-ViT-H-14-laion2B-s32B-b79K (safetensors, ~2.5 GB)
  - WD v1.4 ConvNeXt Tagger v2 (ONNX, ~370 MB)

### 3. Run

Double-click **`run.bat`** or run in terminal:

```bash
run.bat
```

The app will automatically open in your default browser.

## Project Structure

```
PromptiGen/
├── install.bat          # One-click installer (dependencies + models)
├── run.bat              # One-click launcher
├── gradio_app.py        # Main Gradio web application
├── tagger.py            # WD14 tagger engine + CLIP pipeline
├── clip_demo.py         # Standalone CLIP demo script
├── config.json          # User settings (auto-generated, gitignored)
├── model/
│   ├── clip_vision/
│   │   ├── config.json          # CLIP model configuration
│   │   └── model.safetensors    # CLIP-ViT-H-14 weights (~2.5 GB)
│   └── wd-v1-4-convnext-tagger-v2/
│       ├── model.onnx           # WD14 tagger weights (~370 MB)
│       └── selected_tags.csv    # Danbooru tag definitions
└── venv/                # Python virtual environment (auto-created)
```

## AI Models

| Model | Size | Purpose |
|---|---|---|
| [CLIP-ViT-H-14-laion2B](https://huggingface.co/Kuvshin/models-moved) | ~2.5 GB | Vision analysis (art style, lighting, mood, etc.) |
| [WD v1.4 ConvNeXt Tagger v2](https://huggingface.co/Bercraft/wd-v1-4-convnext-tagger-v2) | ~370 MB | Danbooru-style tagging (characters, poses, etc.) |

## Optional: AI Enhancement

PromptiGen can enhance generated tags into polished Stable Diffusion prompts using **OpenRouter** (free tier available):

1. Get a free API key at [openrouter.ai/keys](https://openrouter.ai/keys)
2. Paste it in the app's **AI Enhancement** section
3. Choose from 14+ free models (Gemma, LLaMA, Qwen, etc.)

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)** - see the [LICENSE](file:///d:/zipcrack/tagger/LICENSE) file for details.
