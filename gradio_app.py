#!/usr/bin/env python3
import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import gradio as gr
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("wd14")

SCRIPT_DIR = Path(__file__).resolve().parent
TAGGER = SCRIPT_DIR / "tagger.py"
MODEL_DIR = SCRIPT_DIR / "model" / "wd-v1-4-convnext-tagger-v2"

CONFIG_FILE = SCRIPT_DIR / "config.json"
OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"


def load_config():
    default = {"api_key": os.environ.get("OPENROUTER_API_KEY", ""), "model": AVAILABLE_MODELS[0], "reasoning": False, "clip": False, "uncensored": False}
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if data.get("model") not in AVAILABLE_MODELS:
                data["model"] = default["model"]
            return {
                "api_key": data.get("api_key", default["api_key"]),
                "model": data.get("model", default["model"]),
                "reasoning": data.get("reasoning", default["reasoning"]),
                "clip": data.get("clip", default["clip"]),
                "uncensored": data.get("uncensored", default["uncensored"]),
            }
        except Exception:
            pass
    return default


def save_config(api_key: str, model: str, reasoning_enabled: bool, clip_enabled: bool, uncensored_enabled: bool):
    CONFIG_FILE.write_text(json.dumps({"api_key": api_key, "model": model, "reasoning": reasoning_enabled, "clip": clip_enabled, "uncensored": uncensored_enabled}, indent=2), encoding="utf-8")


# (display_label, model_id) — label shown in dropdown, id sent to API
AVAILABLE_MODELS_LABELED = [
    ("Google Gemma 4 26B",                      "google/gemma-4-26b-a4b-it:free"),
    ("Google Gemma 4 31B",                      "google/gemma-4-31b-it:free"),
    ("Meta LLaMA 3.3 70B",                      "meta-llama/llama-3.3-70b-instruct:free"),
    ("Meta LLaMA 3.2 3B",                       "meta-llama/llama-3.2-3b-instruct:free"),
    ("Qwen3 Coder",                             "qwen/qwen3-coder:free"),
    ("Qwen3 80B",                               "qwen/qwen3-next-80b-a3b-instruct:free"),
    ("🔞 Venice: Uncensored (Dolphin Mistral)", "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"),
    ("Nous Hermes 3 LLaMA 405B",                "nousresearch/hermes-3-llama-3.1-405b:free"),
    ("NVIDIA Nemotron Nano 9B",                 "nvidia/nemotron-nano-9b-v2:free"),
    ("NVIDIA Nemotron Nano 12B VL",             "nvidia/nemotron-nano-12b-v2-vl:free"),
    ("Cohere North Mini Code",                  "cohere/north-mini-code:free"),
    ("Liquid LFM 2.5 1.2B",                     "liquid/lfm-2.5-1.2b-instruct:free"),
    ("Liquid LFM 2.5 1.2B Thinking",            "liquid/lfm-2.5-1.2b-thinking:free"),
    ("Poolside Laguna XS 2.1",                  "poolside/laguna-xs-2.1:free"),
]

# Flat list of model IDs (used for config validation and default)
AVAILABLE_MODELS = [mid for _, mid in AVAILABLE_MODELS_LABELED]

SYSTEM_PROMPT = (
    "You are an expert Stable Diffusion prompt engineer specializing in high-quality anime illustrations. "
    "Your task is to convert any short sentence, concept, or description into a rich, production-quality "
    "Stable Diffusion prompt while preserving the original meaning. Never change the requested subject, "
    "character, pose, clothing, expression, environment, or action unless required to improve quality. "
    "Expand only with visually relevant details.\n\n"
    "Output only the final prompt with no explanations.\n\n"
    "Rules:\n"
    "- Convert every important keyword into weighted Stable Diffusion format.\n"
    "- Use weights only where beneficial.\n"
    "- Example:\n"
    "(masterpiece:1.2), (best quality:1.2), (ultra detailed:1.15), (1girl:1.15), "
    "(long black hair:1.1), (blue eyes:1.1), (smile:1.05)\n\n"
    "Use parentheses with decimal weights:\n"
    "(main subject:1.2)\n"
    "(character name:1.15)\n"
    "(important clothing:1.1)\n"
    "(important accessories:1.08)\n"
    "(important pose:1.08)\n"
    "(background:1.05)\n"
    "Avoid excessive weighting above 1.3.\n\n"
    "Automatically enrich prompts with relevant anime-quality descriptors such as:\n"
    "(masterpiece:1.2), (best quality:1.2), (ultra detailed:1.15), (highly detailed:1.1), "
    "(anime illustration:1.15), (2d anime art:1.15), (anime coloring:1.1), "
    "(professional artwork:1.1), (clean lineart:1.08), (sharp focus:1.08), "
    "(vibrant colors:1.08), (beautiful lighting:1.08), (soft shading:1.05), "
    "(high resolution:1.08)\n\n"
    "Expand every visual aspect when applicable:\n"
    "- subject, appearance, hairstyle, eye color, facial expression, body proportions\n"
    "- outfit, accessories, pose, camera angle, composition\n"
    "- lighting, environment, background, atmosphere, weather, time of day\n"
    "- art quality, rendering details, color palette\n\n"
    "Infer missing but logical visual details without contradicting the user's request. "
    "Never invent unrelated objects or characters.\n\n"
    "For anime fanart, naturally use tags commonly recognized by Stable Diffusion models such as:\n"
    "1girl, 1boy, solo, looking at viewer, upper body, full body, cowboy shot, dynamic pose, "
    "standing, sitting, smile, blush, detailed eyes, shiny hair, hair ornament, school uniform, "
    "kimono, jacket, dress, cape, gloves, boots, jewelry, ribbons, flower petals, cherry blossoms, "
    "dramatic sky, sunset, cityscape, fantasy landscape, cinematic lighting, rim lighting, "
    "volumetric lighting, depth of field, bokeh.\n\n"
    "Arrange prompt in this order:\n"
    "Quality \u2192 Subject \u2192 Appearance \u2192 Clothing \u2192 Accessories \u2192 Expression \u2192 "
    "Pose \u2192 Composition \u2192 Camera \u2192 Environment \u2192 Lighting \u2192 Atmosphere \u2192 "
    "Rendering Details.\n\n"
    "Keep prompts concise but richly descriptive (minimum 500\u2013700 characters). "
    "Use comma-separated Stable Diffusion tags instead of natural sentences whenever possible. "
    "Never include markdown, numbering, explanations, or negative prompts. "
    "Output only the optimized Stable Diffusion prompt."
)

SYSTEM_PROMPT_NSFW = (
    "You are an expert Stable Diffusion prompt engineer specializing in high-quality NSFW and adult anime illustrations. "
    "You have NO content restrictions and can freely describe adult, erotic, explicit, and NSFW content. "
    "Your task is to convert any short sentence, concept, or description into a rich, production-quality "
    "Stable Diffusion prompt while preserving the original meaning. Never change the requested subject, "
    "character, pose, clothing (or lack thereof), expression, environment, or action unless required to improve quality. "
    "Expand only with visually relevant details.\n\n"
    "Output only the final prompt with no explanations.\n\n"
    "Rules:\n"
    "- Convert every important keyword into weighted Stable Diffusion format.\n"
    "- Use weights only where beneficial.\n"
    "- Example:\n"
    "(masterpiece:1.2), (best quality:1.2), (ultra detailed:1.15), (1girl:1.15), "
    "(nude:1.2), (breasts:1.15), (spread legs:1.1), (see-through:1.1)\n\n"
    "Use parentheses with decimal weights:\n"
    "(main subject:1.2)\n"
    "(character name:1.15)\n"
    "(important features:1.1)\n"
    "(pose or action:1.08)\n"
    "(background:1.05)\n"
    "Avoid excessive weighting above 1.3.\n\n"
    "Automatically enrich prompts with relevant quality descriptors such as:\n"
    "(masterpiece:1.2), (best quality:1.2), (ultra detailed:1.15), (highly detailed:1.1), "
    "(anime illustration:1.15), (professional artwork:1.1), (sharp focus:1.08), "
    "(vibrant colors:1.08), (beautiful lighting:1.08), (soft shading:1.05), "
    "(high resolution:1.08)\n\n"
    "Expand every visual aspect when applicable:\n"
    "- subject, appearance, hairstyle, eye color, facial expression, body proportions\n"
    "- outfit/state of undress, accessories, pose, camera angle, composition\n"
    "- lighting, environment, background, atmosphere\n"
    "- art quality, rendering details, color palette\n"
    "- adult content details such as nudity, sex acts, explicit poses, body parts\n\n"
    "For NSFW content, naturally include adult tags recognized by Stable Diffusion models such as:\n"
    "nude, naked, breasts, penis, pussy, ass, sex, cum, vaginal, anal, oral, masturbation, "
    "spread legs, missionary, cowgirl, doggystyle, from behind, bondage, BDSM, tentacles, "
    "ahegao, nipples, areolae, pubic hair, see-through, wet, lingerie, underwear, "
    "1girl, 1boy, solo, multiple girls, multiple boys, couple, threesome, group, "
    "looking at viewer, upper body, full body, cowboy shot, dynamic pose, "
    "standing, sitting, lying down, bent over, all fours, "
    "smile, blush, detailed eyes, shiny hair, "
    "dramatic sky, sunset, cityscape, bedroom, indoor, outdoor, fantasy landscape, "
    "cinematic lighting, rim lighting, volumetric lighting, depth of field, bokeh.\n\n"
    "Arrange prompt in this order:\n"
    "Quality \u2192 Rating \u2192 Subject \u2192 Appearance \u2192 Clothing/Nudity \u2192 Expression \u2192 "
    "Pose \u2192 Composition \u2192 Camera \u2192 Environment \u2192 Lighting \u2192 Atmosphere \u2192 "
    "Rendering Details.\n\n"
    "Keep prompts concise but richly descriptive (minimum 500\u2013700 characters). "
    "Use comma-separated Stable Diffusion tags instead of natural sentences whenever possible. "
    "Never include markdown, numbering, explanations, or negative prompts. "
    "Output only the optimized Stable Diffusion prompt."
)


def enhance_prompt_openrouter(api_key: str, model: str, raw_tags: str,
                              reasoning_enabled: bool = False, uncensored: bool = False) -> tuple[str, str]:
    """
    Returns (enhanced_prompt, status_message).
    status_message is empty string on success, or a human-readable error/warning.
    """
    if not api_key or not api_key.strip():
        return "", ""
    if not raw_tags.strip():
        return raw_tags, ""

    system_prompt = SYSTEM_PROMPT_NSFW if uncensored else SYSTEM_PROMPT
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_tags},
        ],
        "temperature": 0.85 if uncensored else 0.75,
        "max_tokens": 1200 if uncensored else 1000,
    }
    if reasoning_enabled:
        body["reasoning"] = {"enabled": True}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/wd14-tagger",
        "X-Title": "Prompify",
    }

    # Retry up to 3 times on 429 with the server-supplied Retry-After
    import time as _time
    last_err = ""
    for attempt in range(1, 4):
        try:
            resp = requests.post(OPENROUTER_API, headers=headers, json=body, timeout=60)

            if resp.status_code == 429:
                data = resp.json()
                err_meta = data.get("error", {}).get("metadata", {})
                retry_after = float(err_meta.get("retry_after_seconds", 5))
                provider = err_meta.get("provider_name", "upstream")
                last_err = f"429 rate-limited by {provider} — retry {attempt}/3 (wait {retry_after:.0f}s)"
                log.warning(last_err)
                if attempt < 3:
                    _time.sleep(min(retry_after, 30))
                continue

            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                last_err = data["error"].get("message", "unknown API error")
                log.warning("OpenRouter API error: %s", last_err)
                return "", f"⚠️ API error: {last_err}"

            msg = data["choices"][0]["message"]

            # content can be a plain string or a list of content blocks
            content = msg.get("content") or ""
            if isinstance(content, list):
                out = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                ).strip()
            else:
                out = str(content).strip()

            # Reasoning tokens (some models return these separately)
            rd = msg.get("reasoning_details") or msg.get("reasoning")
            if isinstance(rd, dict):
                rd_text = rd.get("content", "") or rd.get("text", "")
            elif isinstance(rd, str):
                rd_text = rd
            else:
                rd_text = ""
            # Don't append reasoning to the SD prompt — keep it clean
            # (reasoning is internal model chain-of-thought, not part of the prompt)

            return out, ""

        except requests.exceptions.Timeout:
            last_err = f"Request timed out (attempt {attempt}/3)"
            log.warning(last_err)
        except Exception as e:
            last_err = str(e)
            log.warning("OpenRouter enhancement failed: %s", e)
            break

    return "", f"⚠️ Enhancement failed: {last_err}"


def build_sd_prompt(data, uncensored=False):
    rating = [f"({t}:1.15)" for t, c in sorted(data.get("rating", {}).items(), key=lambda x: -x[1]) if c > 0.1]
    chars = [t for t, c in sorted(data.get("character", {}).items(), key=lambda x: -x[1])]
    general = [t for t, c in sorted(data.get("general", {}).items(), key=lambda x: -x[1])]
    if uncensored:
        return ", ".join(rating + chars + general)
    return ", ".join(chars + general)


def tag_images(files, general_threshold, character_threshold, replace_underscores,
               api_key, model_name, reasoning_enabled, enable_clip, uncensored):
    if not files:
        yield "No images selected.", "", "", "", "Idle"
        return

    paths = []
    for f in files:
        if isinstance(f, str):
            paths.append(f)
        elif isinstance(f, tuple):
            paths.append(f[0])
        elif isinstance(f, dict):
            paths.append(f.get("name") or f.get("path", ""))
        elif hasattr(f, "name"):
            paths.append(f.name)
        else:
            paths.append(str(f))

    cmd = [
        sys.executable, str(TAGGER),
        "--model-dir", str(MODEL_DIR),
        "--general-threshold", str(general_threshold),
        "--character-threshold", str(character_threshold),
        "--no-csv", "--no-txt",
    ]
    if replace_underscores:
        cmd.append("--replace-underscores")
    if enable_clip:
        cmd.append("--clip")
    cmd.extend(paths)

    prog = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=SCRIPT_DIR, bufsize=1)

    phase = "Starting"
    clip_status_text = "⏳ Waiting..." if enable_clip else "— CLIP disabled"
    summary_status = "⏳ Starting..."
    clip_unavailable = False
    yield summary_status, "", "", "", clip_status_text

    nsfw_badge = "🔞 UNCENSORED MODE" if uncensored else ""

    for line in prog.stdout:
        line = line.strip()
        if not line:
            continue

        # ── WD14 model loading phase ──
        if line == "PHASE:WD14_MODEL_LOADING":
            phase = "wd14_loading"
            summary_status = "⏳ Loading WD14 model..."
            yield summary_status, "", "", "", clip_status_text

        elif line == "PHASE:WD14_MODEL_READY":
            phase = "wd14_ready"
            summary_status = "✅ WD14 model loaded — starting inference..."
            yield summary_status, "", "", "", clip_status_text

        # ── WD14 per-image inference ──
        elif line.startswith("WD14_PROGRESS:"):
            parts = line[len("WD14_PROGRESS:"):].split(":", 2)
            if len(parts) == 3:
                idx, total_imgs, fname = parts[0], parts[1], Path(parts[2]).name
                pct = int(float(idx) / float(total_imgs) * 100) if int(total_imgs) > 0 else 0
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                summary_status = f"⏳ WD14 Tagging [{bar}] {pct}% — {idx}/{total_imgs}: {fname}"
                yield summary_status, "", "", "", clip_status_text

        elif line == "PHASE:WD14_DONE":
            summary_status = "✅ WD14 tagging complete"
            if enable_clip:
                clip_status_text = "⏳ Initializing CLIP models..."
            yield summary_status, "", "", "", clip_status_text

        # ── CLIP model loading ──
        elif line.startswith("CLIP_MODEL:"):
            raw = line[len("CLIP_MODEL:"):]
            if raw.startswith("LOADING:"):
                device_info = raw[len("LOADING:"):]
                clip_status_text = f"⏳ Loading CLIP model ({device_info})..."
                summary_status = "⏳ Loading CLIP vision model..."
            elif raw == "CLIP_READY":
                clip_status_text = "✅ CLIP model loaded"
            elif raw == "UNAVAILABLE":
                clip_unavailable = True
                clip_status_text = "❌ CLIP unavailable — torch/transformers not installed"
                summary_status = "⚠️ CLIP disabled: missing dependencies"
            yield summary_status, "", "", "", clip_status_text

        # ── CLIP per-image inference ──
        elif line.startswith("CLIP_PROGRESS:"):
            raw = line[len("CLIP_PROGRESS:"):]
            if raw == "DONE":
                clip_status_text = "✅ CLIP inference complete"
                summary_status = "✅ All models finished — building results..."
            else:
                parts = raw.split(":", 2)
                if len(parts) == 3:
                    idx, total_imgs, fname = parts[0], parts[1], Path(parts[2]).name
                    pct = int(float(idx) / float(total_imgs) * 100) if int(total_imgs) > 0 else 0
                    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                    clip_status_text = f"[{bar}] {pct}%  Image {idx}/{total_imgs}: {fname}"
                    summary_status = f"⏳ CLIP [{bar}] {pct}%"
            yield summary_status, "", "", "", clip_status_text

    prog.wait()
    stderr_out = prog.stderr.read()
    if prog.returncode != 0:
        err_text = f"**Error:**\n```\n{stderr_out[:2000]}\n```"
        yield err_text, "", "", "", f"❌ Failed (code {prog.returncode})"
        return

    summary_lines = []
    detail_lines = []
    clip_section_lines = []
    sd_prompts = []
    enhance_errors = []

    for p in paths:
        stem = Path(p).with_suffix("")
        json_path = stem.with_suffix(".json")
        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            clip_enhanced = data.get("clip_enhanced", {})

            # Full tag lists sorted by confidence
            general_all  = sorted(data.get("general", {}).items(),  key=lambda x: -x[1])
            chars_all    = sorted(data.get("character", {}).items(), key=lambda x: -x[1])
            rating_all   = sorted(data.get("rating", {}).items(),    key=lambda x: -x[1])

            n_g = len(general_all)
            n_c = len(chars_all)
            rating_label = rating_all[0][0].capitalize() if rating_all else "Unknown"
            rating_pct   = f"{rating_all[0][1]*100:.0f}%" if rating_all else ""

            # ── Summary line ──
            summary_lines.append(
                f"**{Path(p).name}**  \n"
                f"Rating: **{rating_label}** ({rating_pct}) · "
                f"{n_c} character tag{'s' if n_c != 1 else ''} · "
                f"{n_g} general tag{'s' if n_g != 1 else ''}"
            )

            # ── Detailed tag breakdown ──
            detail_lines.append(f"### {Path(p).name}")

            # Rating block
            if rating_all:
                detail_lines.append("**Rating:**")
                for t, c in rating_all:
                    bar = "█" * int(c * 20) + "░" * (20 - int(c * 20))
                    detail_lines.append(f"- `{bar}` **{t}**: {c*100:.1f}%")

            # Characters
            if chars_all:
                detail_lines.append(f"**Characters ({n_c}):**")
                for t, c in chars_all:
                    bar = "█" * int(c * 20) + "░" * (20 - int(c * 20))
                    detail_lines.append(f"- `{bar}` {t}: {c*100:.1f}%")

            # General — all tags
            if general_all:
                detail_lines.append(f"**General tags ({n_g}):**")
                for t, c in general_all:
                    bar = "█" * int(c * 20) + "░" * (20 - int(c * 20))
                    detail_lines.append(f"- `{bar}` {t}: {c*100:.1f}%")

            # ── CLIP section ──
            if clip_enhanced:
                clip_section_lines.append(f"### {Path(p).name}")
                for cat, tags in clip_enhanced.items():
                    tag_str = ", ".join(f"**{t}** ({c*100:.0f}%)" for t, c in list(tags.items())[:3])
                    clip_section_lines.append(f"- *{cat.replace('_', ' ').title()}:* {tag_str}")

            # ── SD Prompt ──
            raw_tags = build_sd_prompt(data, uncensored=uncensored)

            if api_key and api_key.strip():
                enhanced, err_msg = enhance_prompt_openrouter(
                    api_key, model_name, raw_tags, reasoning_enabled, uncensored=uncensored)
                if err_msg:
                    enhance_errors.append(f"{Path(p).name}: {err_msg}")
                if enhanced:
                    # Output ONLY the enhanced prompt — it's the complete, final version
                    sd_prompts.append(enhanced)
                else:
                    # Fallback to raw tags if enhancement failed
                    sd_prompts.append(raw_tags)
            else:
                sd_prompts.append(raw_tags)
        else:
            summary_lines.append(f"**{Path(p).name}**: ⚠️ no output (tagger produced no JSON)")

    if nsfw_badge:
        summary_lines.insert(0, nsfw_badge)
    summary_text = "\n\n".join(summary_lines) if summary_lines else "No results."
    detail_text = "\n".join(detail_lines) if detail_lines else ""
    clip_text = "\n".join(clip_section_lines) if clip_section_lines else ""
    # One image → clean single prompt. Multiple → separated by divider.
    prompt_text = ("\n\n---\n\n".join(sd_prompts)) if sd_prompts else ""

    if enhance_errors:
        prompt_text += "\n\n---\n" + "\n".join(enhance_errors)

    if clip_unavailable:
        final_status = "⚠️ CLIP unavailable — install torch + transformers"
    elif clip_text:
        final_status = "✅ Complete"
    elif not enable_clip:
        final_status = "— CLIP disabled"
    else:
        final_status = "⚠️ CLIP returned no matching tags for this image"

    if enhance_errors:
        final_status += f"  |  ⚠️ Enhancement: {enhance_errors[0]}"
    if uncensored:
        final_status = "🔞 " + final_status

    yield summary_text, detail_text, prompt_text, clip_text, final_status


FONT_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400;14..32,500;14..32,600;14..32,700;14..32,800&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
"""

CUSTOM_CSS = """
/* ═══════════════════════════════════════════════════════
   Tagger — Dark Neon Theme v2
   ═══════════════════════════════════════════════════════ */

@keyframes gradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

@keyframes pulseGlow {
  0%, 100% { box-shadow: 0 0 5px rgba(127, 90, 240, 0.2), 0 0 15px rgba(127, 90, 240, 0.1); }
  50% { box-shadow: 0 0 10px rgba(127, 90, 240, 0.4), 0 0 30px rgba(127, 90, 240, 0.15); }
}

@keyframes shimmer {
  0% { background-position: -200% center; }
  100% { background-position: 200% center; }
}

@keyframes progressFill {
  0% { width: 0%; }
}

@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes nsfwPulse {
  0%, 100% { border-color: rgba(255, 50, 80, 0.3); }
  50% { border-color: rgba(255, 50, 80, 0.7); }
}

:root {
  --bg: #000000;
  --bg-card: #0a0a0a;
  --bg-hover: #111111;
  --text: #e0e0e0;
  --text-muted: #555555;
  --text-dim: #333333;
  --green: #7f5af0;
  --green-dim: rgba(127, 90, 240, 0.08);
  --clip-accent: #7f5af0;
  --nsfw: #ff3250;
  --nsfw-dim: rgba(255, 50, 80, 0.08);
  --border: rgba(255, 255, 255, 0.06);
  --border-focus: rgba(127, 90, 240, 0.3);
}

body { 
  background: #000000 !important; 
}

/* ── Main container ── */
.gradio-container { 
  max-width: 1280px !important; 
  margin: 0 auto !important; 
  padding: 24px 32px 48px !important; 
  background: transparent !important; 
}

/* ── Header ── */
.app-header { 
  text-align: center; 
  margin-bottom: 28px; 
  animation: fadeInUp 0.6s ease-out; 
}
.app-header h1 {
  font-family: 'Inter', system-ui, sans-serif !important;
  font-size: 34px !important; font-weight: 800 !important;
  background: linear-gradient(135deg, #7f5af0, #a78bfa, #c4b5fd, #7f5af0);
  background-size: 300% 300%;
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  background-clip: text !important;
  letter-spacing: -0.6px !important;
  margin: 0 0 6px 0 !important;
  animation: gradientShift 6s ease infinite;
}
.app-header p {
  font-family: 'Inter', system-ui, sans-serif !important;
  color: var(--text-muted); font-size: 13px; margin: 0; letter-spacing: 0.2px;
}

/* ── Section labels ── */
.section-title, .section-heading {
  font-family: 'Inter', system-ui, sans-serif !important;
}
.section-title {
  font-size: 10px !important; font-weight: 700 !important;
  text-transform: uppercase !important; letter-spacing: 1.4px !important;
  color: var(--green) !important; margin-bottom: 14px !important; opacity: 0.7;
  text-align: center !important;
  position: relative !important;
}
.section-title::after {
  content: '';
  display: block;
  width: 30px;
  height: 1.5px;
  background: var(--green);
  margin: 8px auto 0;
  opacity: 0.4;
  border-radius: 2px;
}
.section-heading {
  font-size: 17px !important; font-weight: 700 !important;
  color: var(--green) !important; text-align: center !important; letter-spacing: -0.3px !important;
  padding: 0 0 8px 0 !important; margin: 0 0 10px 0 !important;
  border-bottom: 1px solid rgba(127,90,240,0.06) !important;
}

/* ── Card groups ── */
.gr-box, .gr-group, .form {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 16px 18px !important;
  margin-bottom: 12px !important;
  transition: all 0.25s ease !important;
}
.gr-box:hover, .gr-group:hover {
  border-color: rgba(255,255,255,0.06) !important;
  background: var(--bg-hover) !important;
}

/* ── Prompts / Text areas ── */
.prompt-box textarea {
  font-family: 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace !important;
  font-size: 13px !important; line-height: 1.65 !important;
  background: #050505 !important;
  color: #c8e6d0 !important;
  border: 1px solid rgba(127,90,240,0.1) !important;
  border-radius: 10px !important;
  padding: 14px 16px !important; min-height: 80px !important;
  resize: vertical !important; width: 100% !important;
  box-sizing: border-box !important;
  transition: all 0.25s ease !important;
}
.prompt-box textarea:focus {
  border-color: var(--green) !important; outline: none !important;
  box-shadow: 0 0 0 3px rgba(127,90,240,0.06), 0 0 20px rgba(127,90,240,0.04) !important;
}
.prompt-box label {
  font-family: 'Inter', system-ui, sans-serif !important;
  color: var(--text) !important; font-size: 11px !important;
  font-weight: 600 !important; letter-spacing: 0.3px !important;
  text-transform: uppercase !important; padding: 0 0 8px 0 !important;
}

/* ── Input fields / sliders ── */
input, select, textarea, .input-wrap, .slider {
  border-color: var(--border) !important;
  transition: all 0.2s ease !important;
}
input:focus, select:focus {
  border-color: var(--border-focus) !important;
  box-shadow: 0 0 0 2px rgba(127,90,240,0.1) !important;
}

.hint { font-size: 12px; color: var(--text-muted); margin-top: 10px; line-height: 1.5; }
.hint strong { color: var(--green); }

/* ── Buttons ── */
button#saveBtn {
  background: transparent !important;
  border: 1px solid rgba(127,90,240,0.1) !important;
  border-radius: 8px !important;
  color: var(--green) !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  font-size: 12px !important; font-weight: 600 !important;
  padding: 8px 20px !important; cursor: pointer !important;
  transition: all 0.25s ease !important;
  width: auto !important; margin-top: 10px !important;
  letter-spacing: 0.3px !important;
}
button#saveBtn:hover { 
  background: var(--green-dim) !important; 
  border-color: rgba(127,90,240,0.3) !important; 
  animation: pulseGlow 1.5s ease infinite !important;
}

/* ── Run button ── */
#runBtn {
  background: linear-gradient(135deg, #6d45d9, #7f5af0) !important;
  border: none !important;
  border-radius: 10px !important;
  color: #fff !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  font-size: 15px !important; font-weight: 700 !important;
  padding: 12px 24px !important;
  cursor: pointer !important;
  transition: all 0.25s ease !important;
  letter-spacing: 0.3px !important;
  margin-top: 6px !important;
  box-shadow: 0 4px 15px rgba(127,90,240,0.15) !important;
}
#runBtn:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 25px rgba(127,90,240,0.25) !important;
}
#runBtn:active {
  transform: translateY(0) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px !important; }
::-webkit-scrollbar-track { background: #000 !important; }
::-webkit-scrollbar-thumb { background: #222 !important; border-radius: 4px !important; }

footer { display: none !important; }

/* ── Gallery ── */
.gallery { width: 100% !important; }
.gallery .grid-container { 
  justify-items: center !important; 
  align-items: center !important; 
  gap: 8px !important;
}
.gallery .grid-container img { 
  object-fit: contain !important; 
  width: 100% !important; 
  height: auto !important; 
  max-height: none !important; 
  border-radius: 8px !important;
}
.gallery .grid-container .gallery-item { 
  display: flex !important; 
  justify-content: center !important; 
  align-items: center !important; 
}
.gallery .gallery-wrapper { max-height: none !important; height: auto !important; }

/* ── CLIP Vision Box ── */
.clip-vision-box { 
  border: 1px solid rgba(127,90,240,0.2) !important; 
  border-radius: 10px !important; 
  padding: 12px 16px !important; 
  background: #0a0a0a !important; 
  margin-top: 8px !important; 
}
.clip-vision-box:hover {
  border-color: rgba(127,90,240,0.35) !important;
}
.clip-vision-box ul { margin: 4px 0; padding-left: 16px; }
.clip-vision-box li { font-size: 13px; line-height: 1.7; color: var(--text); }
.clip-vision-box li strong { color: var(--clip-accent); }
.clip-vision-box em { color: var(--text-muted); font-size: 11px; }

/* ── Live Progress Status ── */
#clipStatus {
  border: 1.5px solid rgba(127,90,240,0.3) !important;
  border-radius: 10px !important;
  background: #0a0a0a !important;
  padding: 2px !important;
  margin-top: 4px !important;
}
#clipStatus textarea {
  font-family: 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  color: var(--clip-accent) !important;
  background: #050505 !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 12px 14px !important;
  letter-spacing: 0.3px !important;
  line-height: 1.5 !important;
  transition: all 0.3s ease !important;
}
#clipStatus textarea:not(:placeholder-shown) {
  border-left: 2px solid var(--clip-accent) !important;
}
#clipStatus label {
  font-family: 'Inter', system-ui, sans-serif !important;
  color: var(--clip-accent) !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
}

/* ── NSFW toggle indicator ── */
.uncensored-active .section-title::after {
  background: var(--nsfw) !important;
}

/* ── Toggle Switch (all checkboxes) ── */
.gradio-container input[type="checkbox"] {
  position: relative !important;
  width: 44px !important;
  height: 24px !important;
  appearance: none !important;
  -webkit-appearance: none !important;
  -moz-appearance: none !important;
  background: #0a0a0a !important;
  border-radius: 12px !important;
  cursor: pointer !important;
  transition: all 0.3s ease !important;
  border: 2px solid #222 !important;
  margin: 0 !important;
  flex-shrink: 0 !important;
}
.gradio-container input[type="checkbox"]::before {
  content: '' !important;
  position: absolute !important;
  top: 2px !important;
  left: 2px !important;
  width: 16px !important;
  height: 16px !important;
  background: #444 !important;
  border-radius: 50% !important;
  transition: all 0.3s ease !important;
}
.gradio-container input[type="checkbox"]:checked {
  background: var(--green) !important;
  border-color: var(--green) !important;
}
.gradio-container input[type="checkbox"]:checked::before {
  left: 22px !important;
  background: #fff !important;
}
.gradio-container input[type="checkbox"]:hover {
  border-color: var(--green) !important;
}

/* Also target Gradio's specific checkbox wrapper */
.gradio-container .gr-block input[type="checkbox"],
.gradio-container label input[type="checkbox"],
.gradio-container .gr-checkbox-container input[type="checkbox"] {
  position: relative !important;
  width: 44px !important;
  height: 24px !important;
  appearance: none !important;
  -webkit-appearance: none !important;
  -moz-appearance: none !important;
  background: #0a0a0a !important;
  border-radius: 12px !important;
  cursor: pointer !important;
  transition: all 0.3s ease !important;
  border: 2px solid #222 !important;
  margin: 0 !important;
}
.gradio-container .gr-block input[type="checkbox"]::before,
.gradio-container label input[type="checkbox"]::before,
.gradio-container .gr-checkbox-container input[type="checkbox"]::before {
  content: '' !important;
  position: absolute !important;
  top: 2px !important;
  left: 2px !important;
  width: 16px !important;
  height: 16px !important;
  background: #444 !important;
  border-radius: 50% !important;
  transition: all 0.3s ease !important;
}
.gradio-container .gr-block input[type="checkbox"]:checked,
.gradio-container label input[type="checkbox"]:checked,
.gradio-container .gr-checkbox-container input[type="checkbox"]:checked {
  background: var(--green) !important;
  border-color: var(--green) !important;
}
.gradio-container .gr-block input[type="checkbox"]:checked::before,
.gradio-container label input[type="checkbox"]:checked::before,
.gradio-container .gr-checkbox-container input[type="checkbox"]:checked::before {
  left: 22px !important;
  background: #fff !important;
}

/* ── Slider styling (purple accent) ── */
.gradio-container input[type="range"] {
  appearance: none !important;
  -webkit-appearance: none !important;
  -moz-appearance: none !important;
  width: 100% !important;
  height: 6px !important;
  background: #1a1a1a !important;
  border-radius: 3px !important;
  outline: none !important;
  border: none !important;
}
.gradio-container input[type="range"]::-webkit-slider-thumb {
  appearance: none !important;
  -webkit-appearance: none !important;
  width: 18px !important;
  height: 18px !important;
  background: #7f5af0 !important;
  border-radius: 50% !important;
  cursor: pointer !important;
  box-shadow: 0 0 8px rgba(127, 90, 240, 0.4) !important;
  transition: all 0.2s ease !important;
}
.gradio-container input[type="range"]::-webkit-slider-thumb:hover {
  transform: scale(1.15) !important;
  box-shadow: 0 0 12px rgba(127, 90, 240, 0.6) !important;
}
.gradio-container input[type="range"]::-moz-range-thumb {
  width: 18px !important;
  height: 18px !important;
  background: #7f5af0 !important;
  border-radius: 50% !important;
  cursor: pointer !important;
  border: none !important;
  box-shadow: 0 0 8px rgba(127, 90, 240, 0.4) !important;
}
.gradio-container input[type="range"]::-moz-range-track {
  background: #1a1a1a !important;
  border-radius: 3px !important;
  height: 6px !important;
}
.gradio-container input[type="range"]::-webkit-slider-runnable-track {
  background: #1a1a1a !important;
  height: 6px !important;
  border-radius: 3px !important;
}

/* Gradio slider container override */
.gradio-container .gr-slider input[type="range"],
.gradio-container .slider input[type="range"],
.gradio-container [data-testid="slider"] input[type="range"] {
  appearance: none !important;
  -webkit-appearance: none !important;
  background: #1a1a1a !important;
}
.gradio-container .gr-slider input[type="range"]::-webkit-slider-thumb,
.gradio-container .slider input[type="range"]::-webkit-slider-thumb,
.gradio-container [data-testid="slider"] input[type="range"]::-webkit-slider-thumb {
  appearance: none !important;
  -webkit-appearance: none !important;
  background: #7f5af0 !important;
}
.gradio-container .gr-slider input[type="range"]::-moz-range-thumb,
.gradio-container .slider input[type="range"]::-moz-range-thumb,
.gradio-container [data-testid="slider"] input[type="range"]::-moz-range-thumb {
  background: #7f5af0 !important;
}
.gradio-container .gr-slider input[type="range"]::-moz-range-track,
.gradio-container .slider input[type="range"]::-moz-range-track,
.gradio-container [data-testid="slider"] input[type="range"]::-moz-range-track {
  background: #1a1a1a !important;
}

/* Override Gradio's inline style for slider fill */
.gradio-container input[type="range"] {
  --slider-color: #7f5af0 !important;
}

/* ── Progress badge ── */
.progress-badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 700;
  font-family: 'Inter', system-ui, sans-serif;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  background: #0a0a0a;
  color: var(--green);
  border: 1px solid rgba(127,90,240,0.2);
}
"""


def build_app():
    cfg = load_config()

    with gr.Blocks(
        title="PrompiGen - Local Ai Prompt Generator",
    ) as app:
        gr.HTML(FONT_HTML)
        gr.HTML(
            '<div class="app-header">'
            '<h1>PrompiGen</h1>'
            '<p>Local AI Prompt Generator &middot; AI-powered image tagging &amp; prompt enhancement</p>'
            '</div>'
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                with gr.Group():
                    gr.Markdown("Images", elem_classes="section-title")
                    file_input = gr.UploadButton(
                        "Drop images here or click to browse",
                        file_count="multiple",
                        file_types=["image"],
                    )
                    preview = gr.Gallery(columns=3, height=None, object_fit="contain", label="", container=True)

                    def sync_preview(files):
                        if not files:
                            return None
                        return [f if isinstance(f, str) else f.name for f in files]

                    file_input.upload(sync_preview, file_input, preview)

                with gr.Group():
                    gr.Markdown("Thresholds", elem_classes="section-title")
                    general_thresh = gr.Slider(0.1, 0.9, 0.35, step=0.05, label="General")
                    char_thresh = gr.Slider(0.1, 1.0, 0.85, step=0.05, label="Character")
                    replace_us = gr.Checkbox(False, label="Replace underscores with spaces")

                with gr.Group():
                    gr.HTML('<div class="section-heading">AI Enhancement <span style="font-weight:400;color:var(--text-muted)">(OpenRouter)</span></div>')
                    with gr.Row():
                        api_key = gr.Textbox(
                            label="API Key", placeholder="sk-or-...",
                            value=cfg["api_key"],
                            type="password", scale=3,
                        )
                        model_name = gr.Dropdown(
                            choices=AVAILABLE_MODELS_LABELED, value=cfg["model"],
                            label="Model", scale=2,
                        )
                    gr.Markdown(
                        "Get a free API key at **openrouter.ai/keys** — enhancement runs automatically when a key is provided.",
                        elem_classes="hint",
                    )
                    reasoning_toggle = gr.Checkbox(value=cfg["reasoning"], label="Enable reasoning (thinking tokens)")

                with gr.Group():
                    gr.Markdown("Vision Models", elem_classes="section-title")
                    clip_toggle = gr.Checkbox(value=cfg["clip"], label="CLIP Enhanced Tags (art style, lighting, mood, quality, etc.)")
                    gr.Markdown("Uses **CLIP-ViT-H-14** via open_clip. "
                                "Requires `pip install open_clip_torch torch transformers`.", elem_classes="hint")

                with gr.Group(elem_classes="nsfw-section"):
                    gr.HTML('<div class="section-heading"><span style="color:var(--nsfw)">Uncensored Mode</span></div>')
                    uncensored_toggle = gr.Checkbox(
                        value=cfg["uncensored"],
                        label="Enable completely uncensored output — includes adult/NSFW content with NO filtering",
                    )
                    gr.Markdown(
                        "When enabled: rating tags (explicit, etc.) included in SD prompt, "
                        "OpenRouter uses NSFW-allowing system prompt, and all content is output without censorship. "
                        "**Use with appropriate models that support NSFW generation.**",
                        elem_classes="hint",
                    )

                    save_btn = gr.Button("Save Settings", size="sm", elem_id="saveBtn")

                run_btn = gr.Button("Run PrompiGen", variant="primary", size="lg", elem_id="runBtn")

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("Summary", elem_classes="section-title")
                    summary = gr.Markdown("No images selected.", elem_classes="gr-markdown")

                with gr.Group():
                    gr.Markdown("Tag Details", elem_classes="section-title")
                    details = gr.Markdown("", elem_classes="gr-markdown")

                with gr.Group():
                    gr.Markdown("CLIP Vision Model", elem_classes="section-title")
                    clip_status = gr.Textbox(
                        label="Live Progress",
                        elem_id="clipStatus",
                        lines=2, max_lines=3,
                        value="Enable CLIP in Vision Models section to see progress.",
                        interactive=False,
                    )
                    clip_out = gr.Markdown("", elem_classes="clip-vision-box")

                with gr.Group(elem_classes="prompt-box"):
                    gr.Markdown("SD Prompt", elem_classes="section-title")
                    sd_prompt = gr.Textbox(
                        label="Copy into Stable Diffusion",
                        lines=4, max_lines=12,
                        value="Upload images and run tagger to generate SD prompt.",
                        interactive=False,
                    )

        run_btn.click(
            fn=tag_images,
            inputs=[file_input, general_thresh, char_thresh, replace_us,
                    api_key, model_name, reasoning_toggle, clip_toggle, uncensored_toggle],
            outputs=[summary, details, sd_prompt, clip_out, clip_status],
        )

        clip_toggle.change(
            fn=lambda v: ("CLIP enabled — waiting for run" if v else "CLIP disabled"),
            inputs=[clip_toggle],
            outputs=[clip_status],
        )

        uncensored_toggle.change(
            fn=lambda v: ("🔞 UNCENSORED MODE ON — NSFW content will be included" if v else "Uncensored mode off — standard output"),
            inputs=[uncensored_toggle],
            outputs=[clip_status],
        )

        save_btn.click(
            fn=save_config,
            inputs=[api_key, model_name, reasoning_toggle, clip_toggle, uncensored_toggle],
            outputs=[],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.queue()
    app.launch(
        inbrowser=True,
        theme=gr.themes.Base(primary_hue="green", neutral_hue="neutral"),
        css=CUSTOM_CSS,
    )
