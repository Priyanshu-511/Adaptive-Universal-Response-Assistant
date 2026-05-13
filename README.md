# AURA — Adaptive Universal Response Assistant

A modular, locally-run AI assistant that combines speech, vision, web search, and image/video generation into a single pipeline.

---

## Features

- **Voice Input** — Microphone-based speech recognition via Google Speech API
- **Voice Output** — Text-to-speech with configurable voice, speed, and gender
- **Smart Routing** — Automatically picks the right model (chat, code, or vision) based on your prompt
- **Web Search** — Opens searches across Google, YouTube, Wikipedia, GitHub, Reddit, and more
- **Image & Video Generation** — Text-to-image, image-to-image, and image-to-video via diffusion models
- **Config-Driven** — All settings live in `config.json`; no code changes needed for tuning

---

## Project Structure

```
├── main.py            # Entry point — ties all modules together
├── Assistant.py       # LLM chat/code/vision routing via Ollama
├── SpeechToText.py    # Microphone → text (Google Speech Recognition)
├── TextToSpeech.py    # Text → speech (pyttsx3 / espeak-ng)
├── WebSearch.py       # Browser automation and search URL builder
├── Generator.py       # Image and video generation (Stable Diffusion)
└── config.json        # Central configuration for all modules
```

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) running locally with the following models pulled:
  - `phi4-mini:latest` (chat)
  - `qwen2.5-coder:3b` (code)
  - `llava:latest` (vision)

Install Python dependencies:

```bash
pip install speechrecognition pyttsx3 diffusers torch transformers
```

---

## Usage

Run the main assistant:

```bash
python main.py
```

Or run individual modules directly:

```bash
python Assistant.py       # Text-only chat/code/vision
python SpeechToText.py    # One-shot voice transcription
python TextToSpeech.py    # Interactive TTS configuration
python WebSearch.py       # Browser search from terminal
```

### Example Prompts

| Input | What happens |
|---|---|
| `explain neural networks` | Chat model answers |
| `write python code for DFS` | Coder model answers |
| `what is in this image /path/to/pic.jpg` | Vision model analyses the image |
| `youtube and search lofi music` | Opens YouTube search in browser |
| `generate image of a sunset over mountains` | Creates image via diffusion model |

---

## Configuration

All behaviour is controlled via `config.json`. Key sections:

- `speech_to_text` — energy threshold, pause duration, language, timeouts
- `text_to_speech` — rate, volume, voice gender, espeak variant
- `assistant` — model names, coder keywords, conversation history length
- `generator` — model, output size, steps, FPS, output directory
- `intent` — keyword lists used to route prompts to the right module

---

## System Info

- **Name:** AURA
- **Version:** 1.0.0
- **Platform:** Linux (Ubuntu recommended)