#!/usr/bin/env python3
"""
Usage (CLI):
  python ai_generator.py t2i "a red dragon in a misty forest"
  python ai_generator.py i2i input.jpg "turn it into oil painting"
  python ai_generator.py t2v "ocean waves at sunset" --frames 10
  python ai_generator.py i2v input.jpg "zoom into the sky" --frames 8
  python ai_generator.py          ← interactive menu

Requirements (auto-installed on first run):
  torch  diffusers  transformers  accelerate
  Pillow  imageio  imageio-ffmpeg  safetensors  numpy
"""

import os, sys, subprocess

_DEPS = [
    "torch",
    "diffusers>=0.25",
    "transformers>=4.38",
    "accelerate",
    "Pillow",
    "imageio",
    "imageio-ffmpeg",
    "safetensors",
    "numpy",
]

def _check_imports() -> bool:
    try:
        import torch, numpy, PIL, imageio  # noqa: F401
        from diffusers import (             # noqa: F401
            StableDiffusionPipeline,
            StableDiffusionImg2ImgPipeline,
            DPMSolverMultistepScheduler,
        )
        return True
    except ImportError:
        return False

if not _check_imports():
    print("[Setup] First run – installing dependencies (this takes a few minutes) …")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade"] + _DEPS
    )
    print("[Setup] Done. Restarting …\n")
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import imageio
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)

# Tiny distilled SD — ~600 MB vs 4 GB for full SD.
# Other options: "segmind/tiny-sd" (~400 MB), "nota-ai/bk-sdm-small" (~800 MB)
DEFAULT_MODEL  = "nota-ai/bk-sdm-tiny"

DEFAULT_SIZE   = 256    # 256 or 384 recommended for low-RAM machines
DEFAULT_STEPS  = 15     # 10–20 is plenty for tiny models
DEFAULT_FRAMES = 8
DEFAULT_FPS    = 4
OUTPUT_DIR     = Path("outputs")

NEG_DEFAULT = (
    "blurry, ugly, deformed, low quality, watermark, "
    "jpeg artifacts, worst quality, bad anatomy"
)


def _detect_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        free_vram_gb = torch.cuda.mem_get_info()[0] / 1e9
        dtype = torch.float16 if free_vram_gb >= 3.0 else torch.float32
        name = torch.cuda.get_device_name(0)
        print(f"[Device] GPU – {name}  |  {free_vram_gb:.1f} GB free  |  dtype={dtype}")
        return "cuda", dtype
    # Apple Silicon
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("[Device] Apple MPS (Metal)")
        return "mps", torch.float32
    print("[Device] CPU-only  ←  expect ~2–5 min per image at 256 px")
    return "cpu", torch.float32

DEVICE, DTYPE = _detect_device()

# Load once, reuse across all tasks
_t2i_cache: dict[str, StableDiffusionPipeline]       = {}
_i2i_cache: dict[str, StableDiffusionImg2ImgPipeline] = {}


def _configure_pipe(pipe):
    """Apply memory-saving tricks that work on any hardware."""
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_tiling()          # saves RAM by tiling the decoder
    if DEVICE == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass   # xformers is optional
    return pipe


def _get_scheduler(pipe):
    return DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++",
    )


def load_t2i(model_id: str) -> StableDiffusionPipeline:
    if model_id not in _t2i_cache:
        print(f"[Model] Downloading/loading '{model_id}' for text→image …")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
            safety_checker=None,
            requires_safety_checker=False,
            low_cpu_mem_usage=True,
        )
        pipe.scheduler = _get_scheduler(pipe)
        pipe = pipe.to(DEVICE)
        _configure_pipe(pipe)
        _t2i_cache[model_id] = pipe
        print("[Model] Ready.")
    return _t2i_cache[model_id]


def load_i2i(model_id: str) -> StableDiffusionImg2ImgPipeline:
    """Reuses the txt2img weights so we don't load the model twice."""
    if model_id not in _i2i_cache:
        t2i = load_t2i(model_id)
        print(f"[Model] Building image→image pipeline (shared weights) …")
        pipe = StableDiffusionImg2ImgPipeline(**t2i.components)
        pipe.scheduler = _get_scheduler(pipe)
        _configure_pipe(pipe)
        _i2i_cache[model_id] = pipe
        print("[Model] Ready.")
    return _i2i_cache[model_id]


def _round8(n: int) -> int:
    """SD's VAE requires dimensions to be multiples of 8."""
    return max(64, (n // 8) * 8)


def _prep_size(size: int) -> tuple[int, int]:
    s = _round8(size)
    return s, s


def _make_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    return torch.Generator(device=DEVICE).manual_seed(seed)


def _save_image(img: Image.Image, prefix: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{prefix}_{int(time.time())}.png"
    img.save(path)
    print(f"[Saved] {path}")
    return path


def _save_video(frames: list[np.ndarray], prefix: str, fps: int) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    # GIF first — no ffmpeg needed
    gif_path = OUTPUT_DIR / f"{prefix}_{ts}.gif"
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=max(1, 1000 // fps),
        loop=0,
        optimize=False,
    )
    print(f"[Saved] {gif_path}  (GIF)")

    # Try MP4 if ffmpeg is around
    try:
        mp4_path = OUTPUT_DIR / f"{prefix}_{ts}.mp4"
        with imageio.get_writer(
            str(mp4_path),
            fps=fps,
            codec="libx264",
            quality=6,
            macro_block_size=8,
        ) as writer:
            for f in frames:
                writer.append_data(f)
        print(f"[Saved] {mp4_path}  (MP4)")
        return mp4_path
    except Exception as exc:
        print(f"[Video] MP4 skipped ({exc}) – GIF is available.")
        return gif_path


def _open_image(path: str, size: int) -> Image.Image:
    w, h = _prep_size(size)
    return Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS)


def text_to_image(
    prompt:   str,
    negative: str           = NEG_DEFAULT,
    steps:    int           = DEFAULT_STEPS,
    size:     int           = DEFAULT_SIZE,
    guidance: float         = 7.5,
    seed:     Optional[int] = None,
    model:    str           = DEFAULT_MODEL,
) -> Path:
    """Generate one image from a text prompt."""
    pipe  = load_t2i(model)
    w, h  = _prep_size(size)
    print(f"[T2I] {w}×{h}px | {steps} steps | prompt: '{prompt[:70]}'")

    with torch.inference_mode():
        result = pipe(
            prompt              = prompt,
            negative_prompt     = negative,
            width               = w,
            height              = h,
            num_inference_steps = steps,
            guidance_scale      = guidance,
            generator           = _make_generator(seed),
        )
    return _save_image(result.images[0], "t2i")


def image_to_image(
    input_path: str,
    prompt:     str,
    negative:   str           = NEG_DEFAULT,
    strength:   float         = 0.60,   # 0 = keep original, 1 = full rewrite
    steps:      int           = DEFAULT_STEPS,
    size:       int           = DEFAULT_SIZE,
    guidance:   float         = 7.5,
    seed:       Optional[int] = None,
    model:      str           = DEFAULT_MODEL,
) -> Path:
    """Transform an existing image guided by a text prompt."""
    pipe = load_i2i(model)
    src  = _open_image(input_path, size)
    print(
        f"[I2I] strength={strength} | {steps} steps | "
        f"prompt: '{prompt[:70]}'"
    )

    with torch.inference_mode():
        result = pipe(
            prompt              = prompt,
            negative_prompt     = negative,
            image               = src,
            strength            = strength,
            num_inference_steps = steps,
            guidance_scale      = guidance,
            generator           = _make_generator(seed),
        )
    return _save_image(result.images[0], "i2i")


def text_to_video(
    prompt:   str,
    negative: str           = NEG_DEFAULT,
    frames:   int           = DEFAULT_FRAMES,
    fps:      int           = DEFAULT_FPS,
    steps:    int           = DEFAULT_STEPS,
    size:     int           = DEFAULT_SIZE,
    guidance: float         = 7.5,
    strength: float         = 0.30,   # img2img strength between frames
    seed:     Optional[int] = None,
    model:    str           = DEFAULT_MODEL,
) -> Path:
    """
    Generate a short video from text.
    Frame 0 is a txt2img result; each subsequent frame is img2img on the previous one.
    Low strength keeps frames coherent; a slight seed shift adds subtle motion.
    """
    t2i = load_t2i(model)
    i2i = load_i2i(model)
    w, h = _prep_size(size)
    base_seed = seed if seed is not None else int(time.time()) & 0xFFFFFF
    frame_list: list[np.ndarray] = []

    print(f"[T2V] Generating {frames} frames ({w}×{h}px) …")
    print(f"  frame 1/{frames}  (txt2img seed={base_seed})")
    with torch.inference_mode():
        first = t2i(
            prompt              = prompt,
            negative_prompt     = negative,
            width               = w,
            height              = h,
            num_inference_steps = steps,
            guidance_scale      = guidance,
            generator           = _make_generator(base_seed),
        ).images[0]
    frame_list.append(np.array(first))
    current = first

    for i in range(1, frames):
        frame_seed = base_seed + i * 7   # deterministic but varied
        print(f"  frame {i+1}/{frames}  (img2img seed={frame_seed})")
        with torch.inference_mode():
            current = i2i(
                prompt              = prompt,
                negative_prompt     = negative,
                image               = current,
                strength            = strength,
                num_inference_steps = max(4, int(steps * strength + 2)),
                guidance_scale      = guidance,
                generator           = _make_generator(frame_seed),
            ).images[0]
        frame_list.append(np.array(current))

    return _save_video(frame_list, "t2v", fps)


def image_to_video(
    input_path: str,
    prompt:     str,
    negative:   str           = NEG_DEFAULT,
    frames:     int           = DEFAULT_FRAMES,
    fps:        int           = DEFAULT_FPS,
    strength:   float         = 0.35,
    steps:      int           = DEFAULT_STEPS,
    size:       int           = DEFAULT_SIZE,
    guidance:   float         = 7.5,
    seed:       Optional[int] = None,
    model:      str           = DEFAULT_MODEL,
) -> Path:
    """
    Animate a source image guided by a prompt.
    Each frame runs img2img on the previous one; strength increases gradually
    so the image slowly evolves rather than jumping.
    """
    pipe = load_i2i(model)
    base_seed = seed if seed is not None else int(time.time()) & 0xFFFFFF
    src = _open_image(input_path, size)
    frame_list: list[np.ndarray] = [np.array(src)]   # frame 0 = original
    current = src

    print(f"[I2V] Animating '{Path(input_path).name}' → {frames} frames …")

    for i in range(1, frames):
        frame_seed = base_seed + i * 7
        s = min(strength + i * 0.025, 0.80)   # ramp up to keep things interesting
        print(f"  frame {i+1}/{frames}  strength={s:.2f}  seed={frame_seed}")
        with torch.inference_mode():
            current = pipe(
                prompt              = prompt,
                negative_prompt     = negative,
                image               = current,
                strength            = s,
                num_inference_steps = max(4, int(steps * s + 2)),
                guidance_scale      = guidance,
                generator           = _make_generator(frame_seed),
            ).images[0]
        frame_list.append(np.array(current))

    return _save_video(frame_list, "i2v", fps)


def _ask(label: str, default) -> str:
    val = input(f"  {label} [{default}]: ").strip()
    return val if val else str(default)

def _ask_int(label: str, default: int) -> int:
    return int(_ask(label, default))

def _ask_float(label: str, default: float) -> float:
    return float(_ask(label, default))

def _ask_seed() -> Optional[int]:
    s = input("  Seed [random]: ").strip()
    return int(s) if s else None

def _ask_file(label: str) -> str:
    while True:
        p = input(f"  {label}: ").strip()
        if Path(p).exists():
            return p
        print(f"  ✗ File not found: '{p}'. Try again.")


MENU = """
╔══════════════════════════════════╗
║   Lightweight AI Generator       ║
╠══════════════════════════════════╣
║  1.  Text  → Image               ║
║  2.  Image → Image               ║
║  3.  Text  → Video  (GIF / MP4)  ║
║  4.  Image → Video  (GIF / MP4)  ║
║  q.  Quit                        ║
╚══════════════════════════════════╝"""


def interactive_menu():
    print(MENU)
    while True:
        choice = input("\nChoose [1/2/3/4/q]: ").strip().lower()

        if choice == "q":
            print("Bye!")
            break

        elif choice == "1":
            prompt = input("  Prompt: ").strip()
            if not prompt:
                continue
            text_to_image(
                prompt   = prompt,
                size     = _ask_int("Size (px)", DEFAULT_SIZE),
                steps    = _ask_int("Steps", DEFAULT_STEPS),
                guidance = _ask_float("Guidance scale", 7.5),
                seed     = _ask_seed(),
            )

        elif choice == "2":
            input_path = _ask_file("Input image path")
            prompt     = input("  Prompt: ").strip()
            image_to_image(
                input_path = input_path,
                prompt     = prompt,
                strength   = _ask_float("Strength 0–1 (0=keep, 1=rewrite)", 0.6),
                size       = _ask_int("Size (px)", DEFAULT_SIZE),
                steps      = _ask_int("Steps", DEFAULT_STEPS),
                seed       = _ask_seed(),
            )

        elif choice == "3":
            prompt = input("  Prompt: ").strip()
            if not prompt:
                continue
            text_to_video(
                prompt   = prompt,
                frames   = _ask_int("Frames", DEFAULT_FRAMES),
                fps      = _ask_int("FPS", DEFAULT_FPS),
                size     = _ask_int("Size (px)", DEFAULT_SIZE),
                steps    = _ask_int("Steps", DEFAULT_STEPS),
                strength = _ask_float("Inter-frame strength 0–1", 0.30),
                seed     = _ask_seed(),
            )

        elif choice == "4":
            input_path = _ask_file("Input image path")
            prompt     = input("  Animation prompt: ").strip()
            image_to_video(
                input_path = input_path,
                prompt     = prompt,
                frames     = _ask_int("Frames", DEFAULT_FRAMES),
                fps        = _ask_int("FPS", DEFAULT_FPS),
                strength   = _ask_float("Starting strength 0–1", 0.35),
                size       = _ask_int("Size (px)", DEFAULT_SIZE),
                steps      = _ask_int("Steps", DEFAULT_STEPS),
                seed       = _ask_seed(),
            )

        else:
            print("  Unknown option, try 1/2/3/4/q.")


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="ai_generator",
        description="Lightweight AI Media Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ai_generator.py t2i "a red dragon in a misty forest"
  python ai_generator.py t2i "neon city at night" --size 384 --steps 20
  python ai_generator.py i2i photo.jpg "turn it into watercolor painting"
  python ai_generator.py i2i photo.jpg "cyberpunk style" --strength 0.75
  python ai_generator.py t2v "ocean waves at sunset" --frames 10 --fps 6
  python ai_generator.py i2v photo.jpg "slowly zoom into the sky" --frames 8
  python ai_generator.py          ← interactive menu
        """,
    )
    sub = root.add_subparsers(dest="cmd")

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--model",    default=DEFAULT_MODEL, metavar="HF_ID",
                        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})")
    shared.add_argument("--size",     type=int,   default=DEFAULT_SIZE,
                        help=f"Output size in pixels (default: {DEFAULT_SIZE})")
    shared.add_argument("--steps",    type=int,   default=DEFAULT_STEPS,
                        help=f"Inference steps (default: {DEFAULT_STEPS})")
    shared.add_argument("--guidance", type=float, default=7.5,
                        help="Guidance scale (default: 7.5)")
    shared.add_argument("--seed",     type=int,   default=None,
                        help="Random seed for reproducibility")
    shared.add_argument("--neg",      default=NEG_DEFAULT, metavar="TEXT",
                        help="Negative prompt")

    p1 = sub.add_parser("t2i", parents=[shared], help="Text → Image")
    p1.add_argument("prompt")

    p2 = sub.add_parser("i2i", parents=[shared], help="Image → Image")
    p2.add_argument("input",    help="Path to input image")
    p2.add_argument("prompt")
    p2.add_argument("--strength", type=float, default=0.60,
                    help="How much to change the image (0=none, 1=full)")

    p3 = sub.add_parser("t2v", parents=[shared], help="Text → Video (GIF/MP4)")
    p3.add_argument("prompt")
    p3.add_argument("--frames",   type=int,   default=DEFAULT_FRAMES)
    p3.add_argument("--fps",      type=int,   default=DEFAULT_FPS)
    p3.add_argument("--strength", type=float, default=0.30,
                    help="Inter-frame change (lower = smoother)")

    p4 = sub.add_parser("i2v", parents=[shared], help="Image → Video (GIF/MP4)")
    p4.add_argument("input",  help="Path to source image")
    p4.add_argument("prompt", help="Direction / animation prompt")
    p4.add_argument("--frames",   type=int,   default=DEFAULT_FRAMES)
    p4.add_argument("--fps",      type=int,   default=DEFAULT_FPS)
    p4.add_argument("--strength", type=float, default=0.35,
                    help="Starting per-frame strength")

    return root


if __name__ == "__main__":
    parser = build_parser() if False else _build_parser()   # keeps linters happy
    args = parser.parse_args()

    if args.cmd is None:
        interactive_menu()

    elif args.cmd == "t2i":
        text_to_image(
            prompt   = args.prompt,
            negative = args.neg,
            steps    = args.steps,
            size     = args.size,
            guidance = args.guidance,
            seed     = args.seed,
            model    = args.model,
        )

    elif args.cmd == "i2i":
        image_to_image(
            input_path = args.input,
            prompt     = args.prompt,
            negative   = args.neg,
            strength   = args.strength,
            steps      = args.steps,
            size       = args.size,
            guidance   = args.guidance,
            seed       = args.seed,
            model      = args.model,
        )

    elif args.cmd == "t2v":
        text_to_video(
            prompt   = args.prompt,
            negative = args.neg,
            frames   = args.frames,
            fps      = args.fps,
            steps    = args.steps,
            size     = args.size,
            guidance = args.guidance,
            strength = args.strength,
            seed     = args.seed,
            model    = args.model,
        )

    elif args.cmd == "i2v":
        image_to_video(
            input_path = args.input,
            prompt     = args.prompt,
            negative   = args.neg,
            frames     = args.frames,
            fps        = args.fps,
            strength   = args.strength,
            steps      = args.steps,
            size       = args.size,
            guidance   = args.guidance,
            seed       = args.seed,
            model      = args.model,
        )