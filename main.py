#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import json
import time
import shlex
import ctypes
import queue
import logging
import subprocess
import threading
import urllib.parse
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Optional


class SystemState:
    speech_start = 0.0
    speech_end   = 0.0
    # SET while AURA is speaking or in the post-speech cooldown;
    # the STT thread discards audio captured during this window.
    is_speaking  = threading.Event()
    # Silence to enforce after TTS finishes before the mic reopens.
    # Raise to 2.0+ if echo persists (reverberant room or external speakers).
    POST_SPEECH_BUFFER: float = 1.5
    # Shared queue — set by DualInputManager so TTS can flush stale voice items.
    _input_queue: "queue.Queue | None" = None


_BASE_DIR   = Path(__file__).parent
CONFIG_PATH = _BASE_DIR / "config.json"


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.json not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG: dict = _load_config()


def _setup_logging() -> logging.Logger:
    cfg      = CONFIG.get("logging", {})
    level    = getattr(logging, cfg.get("level", "INFO"), logging.INFO)
    fmt      = cfg.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log_file = _BASE_DIR / CONFIG["paths"].get("log_file", "assistant.log")
    handlers: list[logging.Handler] = []
    if cfg.get("log_to_console", True):
        handlers.append(logging.StreamHandler(sys.stdout))
    if cfg.get("log_to_file", True):
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
    return logging.getLogger("AURA")


logger = _setup_logging()


def _import_module(alias: str, relative_path: str):
    full_path = (_BASE_DIR / relative_path).resolve()
    if not full_path.exists():
        raise FileNotFoundError(f"Module file not found: {full_path}")
    spec = importlib.util.spec_from_file_location(alias, full_path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)   # type: ignore[union-attr]
    logger.debug(f"Loaded module '{alias}' from {full_path}")
    return mod


class ConversationMemory:
    def __init__(self, max_turns: int = 10, system_prompt: str = ""):
        self.max_turns     = max_turns
        self.system_prompt = system_prompt
        self._history: list[dict] = []
        logger.info(f"ConversationMemory init: max_turns={max_turns}")

    def add(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})
        max_msgs = self.max_turns * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]

    def get_messages(self, user_text: str) -> list[dict]:
        msgs: list[dict] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.extend(self._history)
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def clear(self) -> None:
        self._history.clear()
        logger.info("ConversationMemory: cleared.")

    @property
    def turn_count(self) -> int:
        return len(self._history) // 2


class SessionStats:
    def __init__(self):
        self._start           = datetime.now()
        self.total_inputs     = 0
        self.voice_inputs     = 0
        self.keyboard_inputs  = 0
        self.intent_counts: dict[str, int] = {}
        self.errors           = 0

    def record_input(self, source: str, intent: str) -> None:
        self.total_inputs += 1
        if source == "voice":
            self.voice_inputs += 1
        else:
            self.keyboard_inputs += 1
        self.intent_counts[intent] = self.intent_counts.get(intent, 0) + 1

    def record_error(self) -> None:
        self.errors += 1

    @property
    def uptime(self) -> str:
        delta = datetime.now() - self._start
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m, s   = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def summary(self) -> str:
        intent_str = "  |  ".join(f"{k}: {v}" for k, v in sorted(self.intent_counts.items()))
        return (
            f"Uptime {self.uptime}  ·  Total {self.total_inputs}  "
            f"(🎤 {self.voice_inputs}  ⌨️  {self.keyboard_inputs})  ·  Errors {self.errors}\n"
            f"  Intents → {intent_str or 'none'}"
        )


class SpeechToTextNode:
    _ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)

    @staticmethod
    def _silent_error_handler(*_): pass

    def _suppress_alsa(self) -> None:
        try:
            asound = ctypes.cdll.LoadLibrary("libasound.so.2")
            self._alsa_cb = self._ERROR_HANDLER_FUNC(self._silent_error_handler)
            asound.snd_lib_error_set_handler(self._alsa_cb)
        except OSError:
            pass

    def __init__(self):
        try:
            import speech_recognition as sr
        except ImportError:
            raise ImportError("speech_recognition not installed.")

        self._suppress_alsa()
        self.sr = sr
        self.recognizer = sr.Recognizer()
        cfg = CONFIG["speech_to_text"]

        self.recognizer.energy_threshold = cfg["energy_threshold"]
        self.recognizer.dynamic_energy_threshold = cfg["dynamic_energy_threshold"]
        self.recognizer.dynamic_energy_adjustment_damping = cfg["dynamic_energy_adjustment_damping"]
        self.recognizer.dynamic_energy_ratio = cfg["dynamic_energy_ratio"]
        self.recognizer.pause_threshold = cfg["pause_threshold"]
        self.recognizer.phrase_threshold = cfg["phrase_threshold"]
        self.recognizer.non_speaking_duration = cfg["non_speaking_duration"]

        self.timeout = cfg["wait_for_speech_timeout"]
        self.max_phrase_duration = cfg["max_phrase_duration"]
        self.language = cfg["language"]
        self.calibration_dur = cfg["calibration_duration"]
        self.input_file = _BASE_DIR / CONFIG["paths"]["input_file"]

        di_cfg = CONFIG.get("dual_input", {})
        self._backoff_secs = di_cfg.get("stt_error_backoff_seconds", 2)
        buf = CONFIG.get("speech_to_text", {}).get("post_speech_buffer", 1.5)
        SystemState.POST_SPEECH_BUFFER = float(buf)
        logger.info(f"SpeechToTextNode ready. POST_SPEECH_BUFFER={SystemState.POST_SPEECH_BUFFER}s")

    def listen_continuous(self, q: queue.Queue, stop_event: threading.Event) -> None:
        try:
            with self.sr.Microphone() as source:
                print("  🔇  Calibrating microphone (one-time)…", end=" ", flush=True)
                self.recognizer.adjust_for_ambient_noise(source, duration=self.calibration_dur)
                print(f"done  [threshold ≈ {self.recognizer.energy_threshold:.0f}]\n  🎤  Voice listener active\n")

                while not stop_event.is_set():
                    try:
                        listen_start = time.time()
                        audio = self.recognizer.listen(source, timeout=self.timeout, phrase_time_limit=self.max_phrase_duration)
                        listen_end = time.time()

                        # Drop audio captured while AURA is speaking or during cooldown
                        if SystemState.is_speaking.is_set():
                            continue

                        # Also drop audio whose capture window started before TTS finished.
                        # We check listen_start (not listen_end) because the buffer is already
                        # contaminated from the moment recording began.
                        if SystemState.speech_end > 0 and listen_start <= (SystemState.speech_end + SystemState.POST_SPEECH_BUFFER):
                            logger.debug("STT: discarded audio overlapping TTS window")
                            continue

                        text = self.recognizer.recognize_google(audio, language=self.language)
                        if text and text.strip():
                            text = text.strip()
                            self.input_file.write_text(text, encoding="utf-8")
                            q.put(("voice", text))
                    except self.sr.WaitTimeoutError:
                        pass
                    except self.sr.UnknownValueError:
                        pass
                    except self.sr.RequestError as exc:
                        time.sleep(self._backoff_secs)
        except Exception as exc:
            logger.error(f"STT thread crashed: {exc}", exc_info=True)


class KeyboardInputNode:
    def __init__(self, input_file: Path):
        self.input_file = input_file

    def listen_continuous(self, q: queue.Queue, stop_event: threading.Event, prompt: str = "  ⌨️  Type here → ") -> None:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        while not stop_event.is_set():
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                text = line.rstrip("\n").strip()
                if text:
                    self.input_file.write_text(text, encoding="utf-8")
                    q.put(("keyboard", text))
                    sys.stdout.write(prompt)
                    sys.stdout.flush()
            except (EOFError, KeyboardInterrupt):
                break


class DualInputManager:
    def __init__(self, stt_node: SpeechToTextNode, kb_node: KeyboardInputNode):
        self._stt  = stt_node
        self._kb   = kb_node
        self._q: queue.Queue[tuple[str, str]] = queue.Queue()
        SystemState._input_queue = self._q   # expose so TTS can flush stale voice items
        self._stop = threading.Event()

        di_cfg = CONFIG.get("dual_input", {})
        self._voice_priority = di_cfg.get("voice_priority", True)
        self._input_timeout = di_cfg.get("input_timeout_seconds", None)
        kb_prompt = di_cfg.get("keyboard_prompt", "  ⌨️  Type here → ")

        self._voice_thread = threading.Thread(target=self._stt.listen_continuous, args=(self._q, self._stop), daemon=True)
        self._kb_thread = threading.Thread(target=self._kb.listen_continuous, args=(self._q, self._stop, kb_prompt), daemon=True)

        self._voice_thread.start()
        self._kb_thread.start()

    def get_input(self) -> tuple[Optional[str], Optional[str]]:
        try:
            source, text = self._q.get(timeout=self._input_timeout)
            if source == "keyboard" and self._voice_priority:
                try:
                    alt_source, alt_text = self._q.get_nowait()
                    if alt_source == "voice":
                        self._q.put(("keyboard", text))
                        return alt_source, alt_text
                    else:
                        self._q.put((alt_source, alt_text))
                except queue.Empty:
                    pass
            return source, text
        except queue.Empty:
            return None, None

    @property
    def voice_alive(self) -> bool: return self._voice_thread.is_alive()
    @property
    def keyboard_alive(self) -> bool: return self._kb_thread.is_alive()

    def stop(self) -> None:
        self._stop.set()


class TextToSpeechNode:
    _MPG123_BASE = 32768   # 32768 = 100% volume; 65536 = 200%

    def __init__(self):
        try:
            import gtts
        except ImportError:
            raise ImportError("Please install required package: pip install gTTS")

        self.output_file = _BASE_DIR / CONFIG["paths"]["output_file"]
        self._max_tts_chars = CONFIG.get("performance", {}).get("max_response_length_tts", 500)
        self.language = 'en'

        tts_cfg = CONFIG.get("text_to_speech", {})
        volume = float(tts_cfg.get("volume", 1.0))
        volume = max(0.1, min(volume, 2.0))   # clamp to safe mpg123 range
        self._mpg123_scale = int(self._MPG123_BASE * volume)
        logger.info(f"TTS volume: {volume:.1f}x  (mpg123 -f {self._mpg123_scale})")

    def speak(self, text: str, truncate: bool = True) -> None:
        text = (text or "").strip()
        if not text: return

        spoken = text
        if truncate and len(text) > self._max_tts_chars:
            spoken = text[: self._max_tts_chars].rsplit(" ", 1)[0]
            spoken += "… (response truncated for speech)"

        import os
        import subprocess
        from gtts import gTTS

        temp_file = str(_BASE_DIR / "temp_speech.mp3")

        # Mute the mic before even generating audio so the STT thread can't
        # slip anything in during the gTTS network round-trip.
        SystemState.is_speaking.set()
        SystemState.speech_start = time.time()

        try:
            tts = gTTS(text=spoken, lang=self.language, slow=False)
            tts.save(temp_file)
            subprocess.run(["mpg123", "-q", "-f", str(self._mpg123_scale), temp_file])
            if os.path.exists(temp_file):
                os.remove(temp_file)

        except FileNotFoundError:
            print("\n  ⚠️  Error: 'mpg123' is not installed. Please run: sudo apt-get install mpg123")
        except Exception as e:
            print(f"\n  ⚠️  TTS Error: {e}")
        finally:
            # Record when speech actually ended, then hold the mute for a
            # short cooldown so residual room audio doesn't get picked up.
            SystemState.speech_end = time.time()

            def _release_mic():
                time.sleep(SystemState.POST_SPEECH_BUFFER)
                SystemState.is_speaking.clear()
                # Purge any voice segments buffered while AURA was talking —
                # they'd be processed as the next user command otherwise.
                if hasattr(SystemState, "_input_queue") and SystemState._input_queue is not None:
                    flushed = 0
                    while True:
                        try:
                            src, _ = SystemState._input_queue.get_nowait()
                            if src == "voice":
                                flushed += 1
                            else:
                                SystemState._input_queue.put((src, _))   # keyboard input is fine
                                break
                        except Exception:
                            break
                    if flushed:
                        logger.debug(f"STT: flushed {flushed} voice segment(s) buffered during TTS")

            threading.Thread(target=_release_mic, daemon=True).start()


class WebSearchNode:
    def __init__(self):
        cfg = CONFIG["web_search"]
        self.endpoints = cfg.get("search_endpoints", {})
        self.google_base = cfg.get("google_base", "https://www.google.com/search?q=")
        self.open_cmd = cfg.get("open_command", "xdg-open")
        self.output_file = _BASE_DIR / CONFIG["paths"]["output_file"]

    def _open_url(self, url: str) -> None:
        try:
            subprocess.Popen([self.open_cmd, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def search(self, query: str) -> str:
        query = query.strip()
        sep = None
        if " + search " in query.lower(): sep = " + search "
        elif " and search " in query.lower(): sep = " and search "

        if sep:
            domain, term = query.lower().split(sep, 1)
            domain_clean = domain.strip()
            term_clean = term.strip()
            encoded = urllib.parse.quote_plus(term_clean)
            base_url = self.endpoints.get(domain_clean)
            if base_url:
                url = base_url + encoded
            else:
                # Fall back to a Google site search if the domain isn't in config
                url = f"https://www.google.com/search?q=site:{domain_clean}+{encoded}"
            self._open_url(url)
            result = f"Searching {domain_clean} for '{term_clean}'."

        elif "." in query and " " not in query:
            url = query if query.startswith("http") else f"https://{query}"
            self._open_url(url)
            result = f"Opening {url}."
        else:
            self._open_url(self.google_base + urllib.parse.quote_plus(query))
            result = f"Searching Google for '{query}'."

        self.output_file.write_text(result, encoding="utf-8")
        return result


class AssistantNode:
    def __init__(self):
        try:
            import ollama
            self._ollama = ollama
        except ImportError:
            raise ImportError("ollama not installed.")

        module_path = CONFIG["paths"]["modules"]["assistant"]
        try:
            mod = _import_module("aura_assistant", module_path)
            self._classify = mod.classify
            self._extract_image = mod.extract_image
        except Exception:
            self._classify = self._fallback_classify
            self._extract_image = self._fallback_extract_image

        cfg = CONFIG["assistant"]
        self.chat_model = cfg["chat_model"]
        self.coder_model = cfg["coder_model"]
        self.vision_model = cfg["vision_model"]
        self.output_file = _BASE_DIR / CONFIG["paths"]["output_file"]

        conv_cfg = CONFIG["assistant"].get("conversation", {})
        raw_prompt = conv_cfg.get(
            "system_prompt",
            "You are AURA, an Adaptive Universal Response Assistant. "
            "You were created by your user, not by Microsoft, OpenAI, or any other company. "
            "Never claim to be any other AI. Keep answers clear and direct."
        )
        system_prompt = raw_prompt.replace("[CONTEXT_INJECTION]", "").strip()
        # Always enforce the identity guard even if it was stripped from the prompt
        if "not by microsoft" not in system_prompt.lower():
            system_prompt += (
                " You were created by your user. "
                "Never claim to be developed by Microsoft, OpenAI, or any third party."
            )
        self.memory = ConversationMemory(
            max_turns=conv_cfg.get("max_history_turns", 10),
            system_prompt=system_prompt
        )

    def _fallback_classify(self, prompt: str, image_path=None) -> str:
        if image_path: return "vision"
        return "coder" if any(k in prompt.lower() for k in CONFIG["assistant"]["coder_keywords"]) else "chat"

    def _fallback_extract_image(self, text: str) -> Optional[str]:
        for word in text.split():
            if any(word.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp")):
                if os.path.exists(word): return word
        return None

    def respond(self, text: str) -> str:
        image_path = self._extract_image(text)
        task = self._classify(text, image_path)
        model = {"coder": self.coder_model, "vision": self.vision_model, "chat": self.chat_model}.get(task, self.chat_model)

        try:
            if task == "vision" and image_path:
                cleaned = text.replace(image_path, "").strip()
                messages = [{"role": "user", "content": cleaned or "What is in this image?", "images": [image_path]}]
            else:
                messages = self.memory.get_messages(text)

            response = self._ollama.chat(model=model, messages=messages)
            result = response["message"]["content"]

            if task != "vision":
                self.memory.add("user", text)
                self.memory.add("assistant", result)
        except Exception as exc:
            result = f"I encountered an error with the AI model: {exc}"

        self.output_file.write_text(result, encoding="utf-8")
        return result

    def clear_memory(self) -> None:
        self.memory.clear()


class GeneratorNode:
    _VIDEO_KW: frozenset = frozenset({"video", "animate", "animation", "gif", "clip", "movie", "motion", "moving", "loop", "morph", "transition"})
    _IMG_EXTS: tuple = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif")

    _FLAG_PATTERNS = [
        ("seed",      r"--seed\s+(\d+)",                           int),
        ("steps",     r"--steps\s+(\d+)",                          int),
        ("size",      r"--size\s+(\d+)",                           int),
        ("guidance",  r"--guidance\s+([\d.]+)",                    float),
        ("strength",  r"--strength\s+([\d.]+)",                    float),
        ("frames",    r"--frames\s+(\d+)",                         int),
        ("fps",       r"--fps\s+(\d+)",                            int),
        ("model",     r'--model\s+([^\s"\']+)',                    str),
        ("negative",  r'--neg\s+"([^"]+)"|--neg\s+\'([^\']+)\'|--neg\s+([^\-]\S+)', str),
    ]

    def __init__(self):
        self._gen_mod: Optional[object] = None
        self.output_file = _BASE_DIR / CONFIG["paths"]["output_file"]
        cfg = CONFIG["generator"]

        self.default_model   = cfg["default_model"]
        self.default_size    = cfg["default_size"]
        self.default_steps   = cfg["default_steps"]
        self.default_fps     = cfg["default_fps"]
        self.default_frames  = cfg["default_frames"]
        self.guidance_scale  = cfg["guidance_scale"]
        self.strength_i2i    = cfg.get("default_strength_i2i", 0.60)
        self.strength_t2v    = cfg.get("default_strength_t2v", 0.30)
        self.strength_i2v    = cfg.get("default_strength_i2v", 0.35)
        self.negative_prompt = cfg.get("negative_prompt", "blurry, ugly, deformed, low quality")

    def _load_generator(self):
        if self._gen_mod is None:
            module_path = CONFIG["paths"]["modules"]["generator"]
            self._gen_mod = _import_module("aura_generator", module_path)
        return self._gen_mod

    def _extract_image_path(self, text: str) -> Optional[str]:
        # shlex.split handles image paths with spaces safely
        try:
            tokens = shlex.split(text)
        except ValueError:
            tokens = text.split()
        for token in tokens:
            candidate = token.strip("'\".,;:()")
            if any(candidate.lower().endswith(ext) for ext in self._IMG_EXTS):
                if os.path.exists(candidate):
                    return candidate
        return None

    def _parse_overrides(self, text: str) -> dict:
        import re
        overrides = {}
        for name, pattern, cast in self._FLAG_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                raw = next((g for g in m.groups() if g is not None), None)
                if raw is not None:
                    try: overrides[name] = cast(raw.strip())
                    except (ValueError, TypeError): pass
        return overrides

    def _strip_flags(self, text: str) -> str:
        import re
        text = re.sub(r'--neg\s+"[^"]*"', "", text)
        text = re.sub(r"--neg\s+'[^']*'", "", text)
        text = re.sub(r"--neg\s+\S+", "", text)
        text = re.sub(r"--\w+\s+\S+", "", text)
        return text.strip()

    def _detect_mode(self, text: str, image_path: Optional[str]) -> str:
        t = text.lower().strip()
        # Mirror the CLI prefixes from Generator.py
        if t.startswith("t2i"): return "t2i"
        if t.startswith("i2i"): return "i2i"
        if t.startswith("t2v"): return "t2v"
        if t.startswith("i2v"): return "i2v"

        words = set(t.split())
        has_video = bool(words & self._VIDEO_KW)
        if image_path:
            return "i2v" if has_video else "i2i"
        return "t2v" if has_video else "t2i"

    def _clean_prompt(self, text: str, image_path: Optional[str]) -> str:
        import re
        prompt = self._strip_flags(text)

        if image_path:
            prompt = prompt.replace(image_path, "")
            prompt = prompt.replace(f'"{image_path}"', '').replace(f"'{image_path}'", '')

        lower_prompt = prompt.lower().strip()
        for prefix in ("t2i", "i2i", "t2v", "i2v"):
            if lower_prompt.startswith(prefix):
                prompt = prompt[len(prefix):].strip()
                break

        # Strip generator keywords without mangling the actual prompt text
        for kw in sorted(CONFIG["intent"]["content_generation_keywords"], key=len, reverse=True):
            prompt = re.sub(r"(?i)\b" + re.escape(kw) + r"\b", "", prompt)

        # Drop dangling articles/prepositions at the start
        prompt = re.sub(r"(?i)^(a|an|the|of|on|about)\b", "", prompt.strip())

        cleaned = prompt.strip(" .,;:-")
        return cleaned if cleaned else text

    def generate(self, raw_text: str) -> str:
        try:
            gen        = self._load_generator()
            overrides  = self._parse_overrides(raw_text)
            image_path = self._extract_image_path(raw_text)
            mode       = self._detect_mode(raw_text, image_path)
            clean      = self._clean_prompt(raw_text, image_path)

            model    = overrides.get("model",    self.default_model)
            size     = overrides.get("size",     self.default_size)
            steps    = overrides.get("steps",    self.default_steps)
            guidance = overrides.get("guidance", self.guidance_scale)
            seed     = overrides.get("seed",     None)
            negative = overrides.get("negative", self.negative_prompt)
            frames   = overrides.get("frames",   self.default_frames)
            fps      = overrides.get("fps",      self.default_fps)

            if mode == "t2i":
                path = gen.text_to_image(prompt=clean, negative=negative, steps=steps, size=size, guidance=guidance, seed=seed, model=model)
                result = f"✅ Text → Image complete!\n   Prompt: {clean[:80]}\n   Saved: {path}"
            elif mode == "i2i":
                if not image_path: return "Image-to-image needs an input image file.\nExample: i2i photo.jpg watercolor"
                strength = overrides.get("strength", self.strength_i2i)
                path = gen.image_to_image(input_path=image_path, prompt=clean, negative=negative, strength=strength, steps=steps, size=size, guidance=guidance, seed=seed, model=model)
                result = f"✅ Image → Image complete!\n   Source: {image_path}\n   Saved: {path}"
            elif mode == "t2v":
                strength = overrides.get("strength", self.strength_t2v)
                path = gen.text_to_video(prompt=clean, negative=negative, frames=frames, fps=fps, steps=steps, size=size, guidance=guidance, strength=strength, seed=seed, model=model)
                result = f"✅ Text → Video complete!\n   Prompt: {clean[:80]}\n   Saved: {path}"
            else:
                if not image_path: return "Image-to-video needs an input image file.\nExample: i2v photo.jpg zoom out slowly"
                strength = overrides.get("strength", self.strength_i2v)
                path = gen.image_to_video(input_path=image_path, prompt=clean, negative=negative, frames=frames, fps=fps, strength=strength, steps=steps, size=size, guidance=guidance, seed=seed, model=model)
                result = f"✅ Image → Video complete!\n   Source: {image_path}\n   Saved: {path}"

        except Exception as exc:
            result = f"Content generation failed: {exc}"

        self.output_file.write_text(result, encoding="utf-8")
        return result


class IntentClassifier:
    def __init__(self):
        cfg = CONFIG["intent"]
        self._web_kw    = [k.lower() for k in cfg["web_search_keywords"]]
        self._gen_kw    = [k.lower() for k in cfg["content_generation_keywords"]]
        self._chat_kw   = [k.lower() for k in cfg["llm_chat_keywords"]]
        self._exit_cmds = [k.lower() for k in cfg["exit_commands"]]
        self._mem_cmds  = [k.lower() for k in cfg.get("clear_memory_commands", ["clear memory", "reset chat"])]

    def classify(self, text: str) -> tuple[str, float]:
        import re
        t = text.lower().strip()

        if any(t == cmd or t.startswith(cmd + " ") for cmd in self._exit_cmds):
            return "exit", 1.0
        if any(t == cmd or t.startswith(cmd) for cmd in self._mem_cmds):
            return "clear_memory", 1.0

        # CLI-style prefixes always go straight to the generator
        if t.startswith(("t2i ", "i2i ", "t2v ", "i2v ", "t2i", "i2i", "t2v", "i2v")):
            return "content_generation", 1.0

        def _score(keywords: list) -> int:
            count = 0
            for kw in keywords:
                if " " in kw:
                    if kw in t: count += 1
                else:
                    if re.search(r"\b" + re.escape(kw) + r"\b", t): count += 1
            return count

        web_score  = _score(self._web_kw)
        gen_score  = _score(self._gen_kw)
        chat_score = _score(self._chat_kw)

        # Boost for natural phrasing like "create a video" or "generate an image"
        gen_pattern = r"\b(create|generate|make|draw|animate|produce|render)\s+(a\s+|an\s+|the\s+)?(video|image|picture|photo|gif|clip|movie|art)\b"
        if re.search(gen_pattern, t):
            gen_score += 5   # strong enough to beat llm_chat

        total = web_score + gen_score + chat_score
        if total == 0: return "unknown", 0.0

        best = max(web_score, gen_score, chat_score)
        confidence = best / total if total > 0 else 0.0

        if gen_score > 0 and gen_score >= web_score and gen_score >= chat_score:
            return "content_generation", confidence
        if web_score > 0 and web_score >= chat_score:
            return "web_search", confidence
        if chat_score > 0:
            return "llm_chat", confidence

        return "unknown", 0.0


class AURA:
    _BANNER = r"""
╔══════════════════════════════════════════════════════════════════════╗
║        A.U.R.A. — Adaptive Universal Response Assistant              ║
║        v{version:<10}   ·   Dual-Input  ·   Graph Mode Active        ║
╚══════════════════════════════════════════════════════════════════════╝"""

    _INTENT_ICONS: dict[str, str] = {
        "web_search"         : "🌐",
        "content_generation" : "🎨",
        "llm_chat"           : "🤖",
        "clear_memory"       : "🧹",
        "unknown"            : "❓",
        "exit"               : "👋",
    }

    def __init__(self):
        sys_cfg = CONFIG["system"]
        self.name             = sys_cfg["name"]
        self.version          = sys_cfg["version"]
        self.startup_greeting = sys_cfg["startup_greeting"]
        self.shutdown_message = sys_cfg["shutdown_message"]
        self.unknown_msg      = sys_cfg["unknown_intent_message"]
        self.input_file  = _BASE_DIR / CONFIG["paths"]["input_file"]
        self.output_file = _BASE_DIR / CONFIG["paths"]["output_file"]

        self.tts       = TextToSpeechNode()
        self.stt       = SpeechToTextNode()
        self.kb        = KeyboardInputNode(self.input_file)
        self.dual      = DualInputManager(self.stt, self.kb)
        self.web       = WebSearchNode()
        self.assistant = AssistantNode()
        self.generator = GeneratorNode()
        self.intent    = IntentClassifier()
        self.stats     = SessionStats()

    def _route(self, text: str, intent: str) -> str:
        if intent == "web_search":
            text_lower = text.lower()
            if " and search " in text_lower:
                parts = text_lower.split(" and search ", 1)
                domain = parts[0].strip()
                query = parts[1].strip()
                return self.web.search(f"{domain} and search {query}")
            else:
                return self.web.search(text)

        if intent == "content_generation":
            return self.generator.generate(text)

        if intent == "clear_memory":
            self.assistant.clear_memory()
            msg = "Conversation memory cleared. I've forgotten our previous exchange — fresh start!"
            self.output_file.write_text(msg, encoding="utf-8")
            return msg

        return self.assistant.respond(text)

    def _status_line(self) -> str:
        v_ok = "🟢" if self.dual.voice_alive else "🔴"
        k_ok = "🟢" if self.dual.keyboard_alive else "🔴"
        mem  = self.assistant.memory.turn_count
        return f"  [{v_ok} Voice  {k_ok} Keyboard]  Memory: {mem} turn{'s' if mem != 1 else ''}  ·  Uptime: {self.stats.uptime}"

    def run(self) -> None:
        print(self._BANNER.format(version=self.version))
        self.tts.speak(self.startup_greeting)

        while True:
            print("\n" + "─" * 70)
            print(self._status_line())

            try:
                source, user_text = self.dual.get_input()
                if not user_text: continue

                src_icon = "🎤" if source == "voice" else "⌨️ "
                print(f"\n  {src_icon} [{source.upper()}]  You: {user_text}")

                buffered = self.input_file.read_text(encoding="utf-8").strip() if self.input_file.exists() else user_text
                intent, confidence = self.intent.classify(buffered)
                icon = self._INTENT_ICONS.get(intent, "🔹")
                print(f"  {icon}  Intent: {intent}  [confidence: {confidence:.0%}]")
                self.stats.record_input(source, intent)

                if intent == "exit":
                    print(f"\n  {self.shutdown_message}\n\n  Session Summary:\n  {self.stats.summary()}")
                    self.tts.speak(self.shutdown_message)
                    self.dual.stop()
                    break

                response = self._route(buffered, intent)

                if self.output_file.exists():
                    on_disk = self.output_file.read_text(encoding="utf-8").strip()
                    if on_disk: response = on_disk

                print(f"\n  {self.name}:\n{response}\n")
                self.tts.speak(response)

            except KeyboardInterrupt:
                print("\n\n  [Interrupted by user]")
                print(f"\n   Session Summary:\n  {self.stats.summary()}")
                self.tts.speak(self.shutdown_message)
                self.dual.stop()
                break
            except Exception as exc:
                self.stats.record_error()
                err_msg = "I encountered an unexpected error. Please try again."
                print(f"\n  Error: {exc}")
                self.tts.speak(err_msg)


if __name__ == "__main__":
    aura = AURA()
    aura.run()