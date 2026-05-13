import pyttsx3
import platform

engine = pyttsx3.init()
all_voices = engine.getProperty('voices')
OS = platform.system()  # 'Linux', 'Windows', 'Darwin'

#  Windows / macOS  -> real female voice objects exist
#  Linux (espeak-ng) -> female = +f variant on any voice
FEMALE_KEYWORDS = {
    "female", "zira", "hazel", "susan", "eva", "victoria",
    "samantha", "karen", "moira", "tessa", "fiona", "veena",
    "catherine", "alice", "nora", "sara", "helena", "laura"
}

# espeak-ng female pitch variants (f1=deepest … f5=highest)
ESPEAK_FEMALE_VARIANTS = {
    "f1": "+f1  (deep female)",
    "f2": "+f2  (low female)",
    "f3": "+f3  (natural female)  ← recommended",
    "f4": "+f4  (high female)",
    "f5": "+f5  (very high female)",
}

def detect_gender_native(voice):
    """For Windows/macOS where female voices are real objects."""
    if hasattr(voice, 'gender') and voice.gender:
        return voice.gender.capitalize()
    name_id = (voice.name + voice.id).lower()
    return "Female" if any(kw in name_id for kw in FEMALE_KEYWORDS) else "Male"


#  UI HELPERS
def pick_int(prompt, lo, hi, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            val = int(raw)
            if lo <= val <= hi:
                return val
            print(f"  Enter a number between {lo} and {hi}.")
        except ValueError:
            print("  Invalid input. Enter a number.")

def pick_float(prompt, lo, hi, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            val = float(raw)
            if lo <= val <= hi:
                return val
            print(f"  Enter a value between {lo} and {hi}.")
        except ValueError:
            print("  Invalid input. Enter a decimal number.")


#  HEADER
print("=" * 54)
print("           Text  ->  Speech  Converter")
print(f"           Engine : {'espeak-ng (Linux)' if OS == 'Linux' else OS}")
print("=" * 54)


# GENDER CHOICE

print("\n  Select voice gender:\n")
print("  [1]  Male")
print("  [2]  Female")
print("  [3]  All voices")

while True:
    g = input("\nEnter choice (1 / 2 / 3): ").strip()
    if g in ("1", "2", "3"):
        break
    print("  Please enter 1, 2, or 3.")


#  LINUX PATH  (espeak-ng variants)
if OS == "Linux" and g == "2":
    # pick base language voice
    print(f"\n{'=' * 54}")
    print("  Step 2a — Pick a base language voice")
    print("=" * 54)
    for i, v in enumerate(all_voices):
        print(f"  [{i}]  {v.name}")
    print("=" * 54)

    base_idx = pick_int(f"\nSelect voice (0 to {len(all_voices)-1}): ", 0, len(all_voices)-1)
    base_voice = all_voices[base_idx]

    # pick female variant
    print(f"\n{'=' * 54}")
    print("  Step 2b — Pick a female pitch variant")
    print("  (espeak-ng encodes female as pitch variants)")
    print("=" * 54)
    variant_keys = list(ESPEAK_FEMALE_VARIANTS.keys())
    for i, key in enumerate(variant_keys):
        print(f"  [{i}]  {ESPEAK_FEMALE_VARIANTS[key]}")
    print("=" * 54)

    var_idx = pick_int(f"\nSelect variant (0 to {len(variant_keys)-1}) [default 2]: ",
                       0, len(variant_keys)-1, default=2)
    chosen_variant = variant_keys[var_idx]

    # Build the espeak-ng female voice ID: e.g.  "english+f3"
    female_voice_id = base_voice.id + f"+{chosen_variant}"
    engine.setProperty('voice', female_voice_id)

    print(f"\n  Base voice : {base_voice.name}")
    print(f"  Variant    : +{chosen_variant}  (Female)")
    print(f"  Voice ID   : {female_voice_id}")


#  WINDOWS / macOS PATH  (real voice objects)
else:
    tagged = [{"voice": v, "gender": detect_gender_native(v)} for v in all_voices]

    if g == "1":
        filtered = [t for t in tagged if t["gender"] == "Male"]
        label = "Male Voices"
    elif g == "2":
        filtered = [t for t in tagged if t["gender"] == "Female"]
        label = "Female Voices"
    else:
        filtered = tagged
        label = "All Voices"

    # Fallback if filter yields nothing
    if not filtered:
        print(f"\n  No {label} found — showing all voices instead.\n")
        filtered = tagged
        label = "All Voices"

    print(f"\n{'=' * 54}")
    print(f"  {label}")
    print("=" * 54)
    for i, t in enumerate(filtered):
        print(f"  [{i}]  {t['voice'].name}  |  {t['gender']}")
    print("=" * 54)

    choice = pick_int(f"\nSelect voice (0 to {len(filtered)-1}): ", 0, len(filtered)-1)
    selected = filtered[choice]
    engine.setProperty('voice', selected["voice"].id)
    print(f"\n  Voice  : {selected['voice'].name}")
    print(f"  Gender : {selected['gender']}")


# SPEED & VOLUME
print()
rate   = pick_int  ("Speech rate  (80–300 wpm)  [default 150]: ", 80, 300, default=150)
volume = pick_float("Volume       (0.0–1.0)     [default 1.0]: ", 0.0, 1.0, default=1.0)
engine.setProperty('rate', rate)
engine.setProperty('volume', volume)


# SPEAK LOOP

print("\n" + "=" * 54)
print("  Type text to speak.  Type 'quit' to exit.")
print("=" * 54)

while True:
    my_text = input("\nEnter text: ").strip()

    if my_text.lower() == "quit":
        print("Exiting. Goodbye!")
        break

    if not my_text:
        print("  Text cannot be empty. Please type something.")
        continue

    print(f'\n  Speaking: "{my_text}"\n')
    engine.say(my_text)
    engine.runAndWait()