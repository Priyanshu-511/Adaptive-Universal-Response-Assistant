#!/usr/bin/env python3
"""
HUD.py  ─  Live Terminal HUD for A.U.R.A.
─────────────────────────────────────────
Drop this file next to main.py, then follow the ── INTEGRATION ──
block at the very bottom for the ~25 lines to add to main.py.

Requires:  pip install rich
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections import deque
from datetime   import datetime
from enum       import Enum, auto
from typing     import Deque, List, Optional

try:
    from rich.align   import Align
    from rich         import box
    from rich.console import Console
    from rich.layout  import Layout
    from rich.live    import Live
    from rich.panel   import Panel
    from rich.text    import Text
    RICH_OK = True
except ImportError:
    RICH_OK = False


# ══════════════════════════════════════════════════════════════════
#  States
# ══════════════════════════════════════════════════════════════════

class HUDState(Enum):
    IDLE        = auto()   # waiting for any input
    LISTENING   = auto()   # mic open, user speaking
    TYPING      = auto()   # keyboard input received
    THINKING    = auto()   # intent classification in progress
    WEB_SEARCH  = auto()   # WebSearchNode running
    GENERATING  = auto()   # GeneratorNode running
    AI_THINKING = auto()   # AssistantNode / ollama running
    SPEAKING    = auto()   # TTS playing
    ERROR       = auto()   # exception caught


# ══════════════════════════════════════════════════════════════════
#  Theme tables
# ══════════════════════════════════════════════════════════════════

_STATE_THEME: dict[HUDState, dict] = {
    HUDState.IDLE        : {"color": "grey62",         "icon": "💤", "label": "RESTING"},
    HUDState.LISTENING   : {"color": "bright_green",   "icon": "🎧", "label": "LISTENING"},
    HUDState.TYPING      : {"color": "bright_cyan",    "icon": "⌨️ ", "label": "TYPING"},
    HUDState.THINKING    : {"color": "yellow",         "icon": "⚡", "label": "PROCESSING"},
    HUDState.WEB_SEARCH  : {"color": "bright_blue",    "icon": "🌐", "label": "WEB SEARCH"},
    HUDState.GENERATING  : {"color": "bright_magenta", "icon": "🎨", "label": "GENERATING"},
    HUDState.AI_THINKING : {"color": "green",          "icon": "🤖", "label": "AI THINKING"},
    HUDState.SPEAKING    : {"color": "bright_magenta", "icon": "💬", "label": "SPEAKING"},
    HUDState.ERROR       : {"color": "bright_red",     "icon": "❌", "label": "ERROR"},
}

_INTENT_THEME: dict[str, dict] = {
    "web_search"         : {"color": "bright_blue",    "icon": "🌐", "label": "WEB SEARCH"},
    "content_generation" : {"color": "bright_magenta", "icon": "🎨", "label": "CONTENT GENERATION"},
    "llm_chat"           : {"color": "bright_green",   "icon": "🤖", "label": "LLM CHAT"},
    "unknown"            : {"color": "grey50",         "icon": "❓", "label": "UNKNOWN"},
    "exit"               : {"color": "bright_red",     "icon": "👋", "label": "EXIT"},
    "clear_memory"       : {"color": "yellow",         "icon": "🧹", "label": "CLEAR MEMORY"},
}


# ══════════════════════════════════════════════════════════════════
#  Animation source strings
# ══════════════════════════════════════════════════════════════════

_WAVE_CHARS    = "▁▂▃▄▅▆▇█▇▆▅▄▃▂▁"   # waveform
_SPINNER       = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"      # braille spinner
_RIPPLE        = " ·∘○◯○∘· "          # TTS ripple
_FILL          = "░▒▓█"               # progress fill levels

# Mouth shapes cycled while SPEAKING — gives the impression of speech rhythm.
# Each frame is exactly 5 characters wide so the face doesn't jitter.
_MOUTH_TALKING = [
    " ─── ",   # closed (consonant cluster)
    "  ◡  ",   # opening
    " ◡‿◡ ",   # mid-open smile
    " ◯═◯ ",   # wide open vowel
    "  ○  ",   # round vowel
    " ◡‿◡ ",   # closing back
    "  ‿  ",   # tiny
    " ─── ",   # brief silence
]


# ══════════════════════════════════════════════════════════════════
#  Log entry
# ══════════════════════════════════════════════════════════════════

class _Msg:
    __slots__ = ("role", "text", "source", "ts")
    def __init__(self, role: str, text: str, source: str = ""):
        self.role   = role       # "user" | "aura" | "system"
        self.text   = text
        self.source = source     # "voice" | "keyboard" | ""
        self.ts     = datetime.now().strftime("%H:%M:%S")


# ══════════════════════════════════════════════════════════════════
#  AuraHUD
# ══════════════════════════════════════════════════════════════════

class AuraHUD:
    """
    Thread-safe, fullscreen terminal HUD for AURA.

    Quick-start
    ───────────
    hud = AuraHUD(version="1.0.0")
    hud.start()

    hud.set_listening()
    hud.set_thinking("Classifying…")
    hud.set_intent("llm_chat", 0.87)
    hud.add_message("user", "hello", "voice")
    hud.set_ai_thinking("phi4-mini:latest")
    hud.add_message("aura", "Hi! How can I help?")
    hud.set_speaking("Hi! How can I help?")
    hud.set_idle()

    hud.stop()
    """

    FPS          = 12     # render frequency
    MAX_CONV     = 8      # conversation lines visible
    MAX_SYS_LOG  = 200    # internal log buffer size

    def __init__(self, version: str = "1.0.0"):
        if not RICH_OK:
            raise RuntimeError("Please install 'rich':  pip install rich")

        self._version   = version
        self._lock      = threading.Lock()
        self._stop_evt  = threading.Event()
        self._thread    : Optional[threading.Thread] = None
        self._live      : Optional[Live]             = None

        # ── Current state ──────────────────────────────────────
        self._state     : HUDState = HUDState.IDLE
        self._detail    : str      = ""          # extra line under animation
        self._intent    : str      = ""
        self._conf      : float    = 0.0
        self._gen_mode  : str      = ""          # t2i / t2v / i2i / i2v

        # ── Status values ──────────────────────────────────────
        self._voice_ok  : bool = True
        self._kb_ok     : bool = True
        self._turns     : int  = 0
        self._total     : int  = 0
        self._v_count   : int  = 0
        self._k_count   : int  = 0
        self._errors    : int  = 0
        self._start     = datetime.now()

        # ── Buffers ────────────────────────────────────────────
        self._conv   : Deque[_Msg] = deque(maxlen=self.MAX_CONV)
        self._syslog : Deque[str]  = deque(maxlen=self.MAX_SYS_LOG)

        # ── Animation ──────────────────────────────────────────
        self._tick   : int = 0
        self._rng    = random.Random(42)

        self._console = Console(highlight=False)

    # ─────────────────────────────────────────────────────────────
    #  Lifecycle
    # ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the background render thread."""
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._render_loop, daemon=True, name="AURA-HUD"
        )
        self._thread.start()

    def stop(self) -> None:
        """Gracefully stop the HUD."""
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._live:
            try:
                self._live.stop()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────
    #  Public state setters  (all thread-safe)
    # ─────────────────────────────────────────────────────────────

    def set_idle(self) -> None:
        with self._lock:
            self._state  = HUDState.IDLE
            self._detail = ""

    def set_listening(self) -> None:
        with self._lock:
            self._state  = HUDState.LISTENING
            self._detail = "Mic open — say something"

    def set_listening_if_idle(self) -> None:
        """Set LISTENING only if the HUD is currently IDLE.

        Called from the STT thread on every loop iteration so the listening
        face shows the entire time the mic is open. Guarded so it never
        overwrites an active state (THINKING / AI_THINKING / SPEAKING / etc.)
        that the main thread has set.
        """
        with self._lock:
            if self._state == HUDState.IDLE:
                self._state  = HUDState.LISTENING
                self._detail = "Mic open — say something"

    def set_transcribing(self, detail: str = "Transcribing your speech…") -> None:
        """Audio captured, now waiting on the Google API.

        Routes to THINKING so the existing PROCESSING palette/animation is
        reused — but with a different detail message so the user knows
        what's happening.
        """
        with self._lock:
            self._state  = HUDState.THINKING
            self._detail = detail

    def set_typing(self) -> None:
        with self._lock:
            self._state  = HUDState.TYPING
            self._detail = "Reading keyboard input…"

    def set_thinking(self, detail: str = "Classifying intent…") -> None:
        with self._lock:
            self._state  = HUDState.THINKING
            self._detail = detail
    def set_web_searching(self, query: str = "") -> None:
        with self._lock:
            self._state  = HUDState.WEB_SEARCH
            self._detail = f"Query: {query[:55]}" if query else "Opening browser…"

    def set_generating(self, mode: str = "") -> None:
        with self._lock:
            self._state    = HUDState.GENERATING
            self._gen_mode = mode.upper()
            self._detail   = f"Mode: {mode.upper()}" if mode else "Generating content…"

    def set_ai_thinking(self, model: str = "") -> None:
        with self._lock:
            self._state  = HUDState.AI_THINKING
            self._detail = f"Model: {model}" if model else "Generating response…"

    def set_speaking(self, text: str = "") -> None:
        with self._lock:
            self._state  = HUDState.SPEAKING
            preview      = text[:70] + ("…" if len(text) > 70 else "")
            self._detail = f'"{preview}"'

    def set_error(self, msg: str = "") -> None:
        with self._lock:
            self._state  = HUDState.ERROR
            self._detail = msg[:80]

    def set_intent(self, intent: str, confidence: float = 0.0) -> None:
        with self._lock:
            self._intent = intent
            self._conf   = confidence

    # ─────────────────────────────────────────────────────────────
    #  Content & stats
    # ─────────────────────────────────────────────────────────────

    def add_message(self, role: str, text: str, source: str = "") -> None:
        with self._lock:
            self._conv.append(_Msg(role, text, source))

    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            self._syslog.append(f"[grey42]{ts}[/]  {msg}")

    def update_status(
        self, *,
        voice_alive: bool = True,
        kb_alive   : bool = True,
        turns      : int  = 0,
        total      : int  = 0,
        voice      : int  = 0,
        keyboard   : int  = 0,
        errors     : int  = 0,
    ) -> None:
        with self._lock:
            self._voice_ok = voice_alive
            self._kb_ok    = kb_alive
            self._turns    = turns
            self._total    = total
            self._v_count  = voice
            self._k_count  = keyboard
            self._errors   = errors

    # ─────────────────────────────────────────────────────────────
    #  Render loop
    # ─────────────────────────────────────────────────────────────

    def _render_loop(self) -> None:
        with Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=self.FPS,
            screen=True,           # fullscreen — keeps terminal clean
            transient=False,
        ) as live:
            self._live = live
            while not self._stop_evt.is_set():
                self._tick += 1
                live.update(self._build_layout())
                time.sleep(1 / self.FPS)

    # ─────────────────────────────────────────────────────────────
    #  Layout assembly
    # ─────────────────────────────────────────────────────────────

    def _build_layout(self) -> Layout:
        with self._lock:
            state    = self._state
            detail   = self._detail
            intent   = self._intent
            conf     = self._conf
            conv     = list(self._conv)
            voice_ok = self._voice_ok
            kb_ok    = self._kb_ok
            turns    = self._turns
            total    = self._total
            v_cnt    = self._v_count
            k_cnt    = self._k_count
            errors   = self._errors
            tick     = self._tick

        root = Layout()
        root.split_column(
            Layout(name="header", size=3),
            Layout(name="middle", size=9),
            Layout(name="conv",   size=10),
            Layout(name="footer", size=1),
        )
        root["middle"].split_row(
            Layout(name="state", ratio=3),
            Layout(name="stats", ratio=2),
        )

        root["header"].update(self._header(voice_ok, kb_ok))
        root["state" ].update(self._state_panel(state, detail, tick))
        root["stats" ].update(self._stats_panel(turns, total, v_cnt, k_cnt, errors))
        root["conv"  ].update(self._conv_panel(conv, intent, conf, tick))
        root["footer"].update(self._footer())
        return root

    # ─────────────────────────────────────────────────────────────
    #  Panel renderers
    # ─────────────────────────────────────────────────────────────

    def _header(self, voice_ok: bool, kb_ok: bool) -> Panel:
        v = "[bright_green]● VOICE[/]" if voice_ok else "[bright_red]● VOICE[/]"
        k = "[bright_cyan]● KEYS[/]"  if kb_ok    else "[bright_red]● KEYS[/]"
        t = Text.assemble(
            ("  A·U·R·A  ", "bold bright_white"),
            (f"v{self._version}  ", "grey50"),
            ("│  ", "grey30"),
            Text.from_markup(f"{v}  {k}"),
            ("  │  ⏱ ", "grey30"),
            (self._uptime(), "bright_yellow"),
            ("  │  Adaptive Universal Response Assistant  ", "grey42"),
        )
        return Panel(Align.center(t), style="grey19 on grey3", padding=(0, 0), box=box.SIMPLE)

    # ── State / animation panel ───────────────────────────────

    _FACE_STATES = (HUDState.IDLE, HUDState.LISTENING, HUDState.SPEAKING)

    def _state_panel(self, state: HUDState, detail: str, tick: int) -> Panel:
        th    = _STATE_THEME[state]
        color = th["color"]
        icon  = th["icon"]
        label = th["label"]

        body = Text()

        if state in self._FACE_STATES:
            # Face-driven layout: the face itself communicates the state, so we
            # skip the icon/label header line (the panel title shows it instead)
            # and use the full 7 content lines for the face + waveform + detail.
            face = self._face(state, tick)
            body.append(face + "\n", style=color)
            if detail:
                d = detail if len(detail) <= 56 else detail[:53] + "…"
                body.append(f"  {d}", style=f"italic {color}")
        else:
            # Compact layout for transitional/processing states
            anim = self._animation(state, tick)
            body.append(f"\n  {icon}  ", style=f"bold {color}")
            body.append(f"{label}\n", style=f"bold {color}")
            body.append(f"\n  {anim}\n", style=color)
            if detail:
                body.append(f"\n  {detail}\n", style=f"italic {color}")

        return Panel(
            body,
            title=f"[bold {color}]  {icon}  {label}  [/]",
            border_style=color,
            box=box.ROUNDED,
            padding=(0, 1),
        )

    # ── Talking-face composer ─────────────────────────────────

    def _face(self, state: HUDState, tick: int) -> str:
        """Multi-line ASCII face for IDLE / LISTENING / SPEAKING."""

        if state == HUDState.SPEAKING:
            # Mouth advances ~every other tick → at FPS=12 that's ~6 shapes/sec,
            # close to natural syllable rate.
            mouth = _MOUTH_TALKING[(tick // 2) % len(_MOUTH_TALKING)]
            # Blink ~once every 4s (48 ticks at 12 FPS), held for 2 ticks.
            blink = (tick % 48) < 2
            eyes  = "─       ─" if blink else "◕       ◕"
            # Waveform is energetic when the mouth is open (vowel/round shapes).
            is_open = any(c in mouth for c in "◯○◡‿═")
            wave    = self._speech_wave(tick, energetic=is_open)
            return (
                "    ╭─────────────╮\n"
                f"    │  {eyes}  │\n"
                "    │      ·      │\n"
                f"    │    {mouth}    │\n"
                "    ╰─────────────╯\n"
                f"      {wave}"
            )

        if state == HUDState.LISTENING:
            # Attentive open eyes, closed mouth — the bars show your incoming voice.
            blink = (tick % 60) < 2
            eyes  = "─       ─" if blink else "◔       ◔"
            wave  = self._speech_wave(tick, energetic=True)
            return (
                "    ╭─────────────╮\n"
                f"    │  {eyes}  │\n"
                "    │      ·      │\n"
                "    │    ─────    │\n"
                "    ╰─────────────╯\n"
                f"      {wave}"
            )

        # IDLE — restful, occasional slow blink.
        # Eyes spend ~10s as ◠ (resting) and briefly open to ◡ as a "wake check".
        is_awake = (tick % 144) < 8
        eyes     = "◡       ◡" if is_awake else "◠       ◠"
        return (
            "    ╭─────────────╮\n"
            f"    │  {eyes}  │\n"
            "    │      ·      │\n"
            "    │     ───     │\n"
            "    ╰─────────────╯\n"
            "      ·  ·  ·  ·  ·"
        )

    def _speech_wave(self, tick: int, energetic: bool = True) -> str:
        """Random-ish 13-char waveform; calmer bars when not energetic."""
        rng = random.Random(tick)
        if energetic:
            bars = "▂▃▄▅▆▇█▆▅▄▃"
        else:
            bars = "▁▁▂▂▁▁"
        return "".join(bars[rng.randint(0, len(bars) - 1)] for _ in range(13))

    def _animation(self, state: HUDState, tick: int) -> str:
        # NOTE: IDLE / LISTENING / SPEAKING are rendered by _face() instead.
        # This method only handles transitional / processing states.
        W = 36  # bar width in characters

        if state in (HUDState.THINKING,):
            # Spinner + filling bar
            sp     = _SPINNER[tick % len(_SPINNER)]
            filled = (tick * 3) % (W + 1)
            bar    = "▓" * filled + "░" * (W - filled)
            return f"{sp}  {bar}"

        if state == HUDState.AI_THINKING:
            # Streaming dots — different phase per tick
            sp      = _SPINNER[tick % len(_SPINNER)]
            offset  = tick % 4
            dots    = ("·" * offset).ljust(W, " ")[:W]
            segment = W // 3
            bar     = "█" * min(segment, tick % (W + 1)) + "░" * (W - min(segment, tick % (W + 1)))
            return f"{sp}  {bar}"

        if state == HUDState.WEB_SEARCH:
            # ◉ dot traveling left-to-right
            pos  = tick % W
            bar  = [" "] * W
            bar[pos] = "◉"
            if pos > 0:
                bar[pos - 1] = "○"
            if pos > 1:
                bar[pos - 2] = "·"
            return "".join(bar)

        if state == HUDState.GENERATING:
            # Growing fill bar (restarts each cycle)
            filled = tick % (W + 1)
            return "█" * filled + "░" * (W - filled)

        if state == HUDState.TYPING:
            # Blinking cursor
            cursor = "█" if (tick // 5) % 2 == 0 else " "
            return f"  _ _ _ _ _ {cursor}  "

        if state == HUDState.ERROR:
            # Flashing warn bar
            return ("━" * W) if (tick // 6) % 2 == 0 else (" " * W)

        # IDLE — slow sinusoidal breathing
        phase = (math.sin(tick * 0.12) + 1) / 2
        n     = int(phase * (W // 2))
        half  = "·" * n
        full  = half + " " * (W // 2 - n)
        return full + full[::-1]

    # ── Stats panel ───────────────────────────────────────────

    def _stats_panel(self, turns: int, total: int, v_cnt: int, k_cnt: int, errors: int) -> Panel:
        t = Text()
        t.append("\n")
        self._stat_row(t, "⏱ ", "Uptime  ", self._uptime(), "bright_yellow")
        self._stat_row(t, "🔁 ", "Turns   ", str(turns),    "bright_white")
        self._stat_row(t, "📊 ", "Total   ", str(total),    "bright_white")
        self._stat_row(t, "🎤 ", "Voice   ", str(v_cnt),    "bright_green")
        self._stat_row(t, "⌨️  ", "Keys    ", str(k_cnt),   "bright_cyan")
        err_color = "bright_red" if errors > 0 else "bright_green"
        self._stat_row(t, "⚠️  ", "Errors  ", str(errors),  err_color)
        return Panel(
            t,
            title="[bold white] SESSION [/]",
            border_style="grey42",
            box=box.ROUNDED,
            padding=(0, 0),
        )

    @staticmethod
    def _stat_row(t: Text, icon: str, label: str, value: str, val_color: str) -> None:
        t.append(f"  {icon}", style="")
        t.append(f" {label}", style="grey50")
        t.append(f"{value}\n", style=f"bold {val_color}")

    # ── Conversation panel ────────────────────────────────────

    def _conv_panel(self, conv: List[_Msg], intent: str, conf: float, tick: int) -> Panel:
        text = Text()

        # ── Intent bar ────────────────────────────────────────
        if intent:
            meta   = _INTENT_THEME.get(intent, {"color": "grey50", "icon": "🔹", "label": intent})
            ic     = meta["color"]
            ii     = meta["icon"]
            lbl    = meta["label"]
            filled = int(conf * 22)
            bar    = "█" * filled + "░" * (22 - filled)
            text.append(f"\n  {ii}  ", style=f"bold {ic}")
            text.append(lbl, style=f"bold {ic}")
            text.append(f"   {bar}  ", style=ic)
            text.append(f"{conf:.0%}\n", style=f"bold {ic}")
            text.append("  " + "─" * 58 + "\n", style="grey30")
        else:
            text.append("\n")

        # ── Messages ──────────────────────────────────────────
        if not conv:
            text.append("  [waiting for first message]\n", style="grey42 italic")
        else:
            for msg in conv:
                if msg.role == "user":
                    src  = "🎤" if msg.source == "voice" else "⌨️ "
                    text.append(f"  {src} ", style="bright_cyan")
                    text.append("You: ", style="bold bright_cyan")
                    body = msg.text if len(msg.text) <= 68 else msg.text[:65] + "…"
                    text.append(f"{body}\n", style="bright_white")
                elif msg.role == "aura":
                    text.append("  🤖 ", style="bright_green")
                    text.append("AURA: ", style="bold bright_green")
                    body = msg.text if len(msg.text) <= 68 else msg.text[:65] + "…"
                    text.append(f"{body}\n", style="white")
                else:
                    text.append(f"  ℹ️  {msg.text}\n", style="grey42 italic")

        return Panel(
            text,
            title="[bold white] CONVERSATION [/]",
            border_style="grey35",
            box=box.ROUNDED,
            padding=(0, 0),
        )

    # ── Footer ────────────────────────────────────────────────

    def _footer(self) -> Text:
        t = Text()
        t.append("  [voice] Speak   [keys] Type   [exit/quit] Stop  ", style="grey42")
        t.append(f"  {datetime.now().strftime('%H:%M:%S')}", style="grey50")
        return t

    # ─────────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────────

    def _uptime(self) -> str:
        delta = datetime.now() - self._start
        h, r  = divmod(int(delta.total_seconds()), 3600)
        m, s  = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


# ══════════════════════════════════════════════════════════════════
#
#  ── INTEGRATION GUIDE ──────────────────────────────────────────
#
#  Add the following to main.py in the places indicated.
#  Total addition: ~25 lines spread across 6 locations.
#
# ──────────────────────────────────────────────────────────────────
#
#  [1] Near the top of main.py, after the existing imports:
#
#       from HUD import AuraHUD, HUDState
#       _HUD: AuraHUD | None = None        # global singleton
#
# ──────────────────────────────────────────────────────────────────
#
#  [2] In TextToSpeechNode.speak(), right after the line
#      "SystemState.is_speaking.set()" :
#
#       if _HUD: _HUD.set_speaking(text)
#
#      And in the _release_mic() inner function, at the very end:
#
#       if _HUD: _HUD.set_idle()
#
# ──────────────────────────────────────────────────────────────────
#
#  [3] In AURA.__init__(), after all node construction is done:
#
#       global _HUD
#       _HUD = AuraHUD(version=self.version)
#
# ──────────────────────────────────────────────────────────────────
#
#  [4] In AURA.run(), replace the existing "while True:" block
#      with the version below (adds ~12 lines, no logic changes):
#
#       _HUD.start()
#       _HUD.log("AURA online")
#       _HUD.add_message("system", self.startup_greeting)
#
#       while True:
#           _HUD.update_status(
#               voice_alive=self.dual.voice_alive,
#               kb_alive=self.dual.keyboard_alive,
#               turns=self.assistant.memory.turn_count,
#               total=self.stats.total_inputs,
#               voice=self.stats.voice_inputs,
#               keyboard=self.stats.keyboard_inputs,
#               errors=self.stats.errors,
#           )
#
#           try:
#               source, user_text = self.dual.get_input()
#               if not user_text: continue
#
#               # ── NEW ──
#               if source == "voice":
#                   _HUD.set_thinking("Heard: " + user_text[:50])
#               else:
#                   _HUD.set_typing()
#               _HUD.add_message("user", user_text, source)
#               # ────────
#
#               buffered = ...  # (same as before)
#               intent, confidence = self.intent.classify(buffered)
#
#               # ── NEW ──
#               _HUD.set_intent(intent, confidence)
#               _HUD.set_thinking(f"Routing → {intent}")
#               # ────────
#
#               self.stats.record_input(source, intent)
#
#               if intent == "exit":
#                   _HUD.stop()           # ← NEW
#                   ...shutdown as before...
#                   break
#
#               # ── NEW: pre-routing state ──
#               if intent == "web_search":
#                   _HUD.set_web_searching(user_text)
#               elif intent == "content_generation":
#                   _HUD.set_generating()
#               else:
#                   _HUD.set_ai_thinking(
#                       self.assistant.chat_model
#                       if hasattr(self.assistant, "chat_model") else ""
#                   )
#               # ───────────────────────────
#
#               response = self._route(buffered, intent)
#
#               # ── NEW ──
#               _HUD.add_message("aura", response)
#               # ────────
#
#               self.tts.speak(response)   # set_speaking called inside speak()
#
#           except KeyboardInterrupt:
#               _HUD.stop()               # ← NEW
#               ...shutdown as before...
#               break
#           except Exception as exc:
#               _HUD.set_error(str(exc))  # ← NEW
#               ...
#
# ══════════════════════════════════════════════════════════════════