import speech_recognition as sr
import os
import ctypes

# Suppress ALSA & JACK logs (safely)

_ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
    None,
    ctypes.c_char_p, ctypes.c_int,
    ctypes.c_char_p, ctypes.c_int,
    ctypes.c_char_p
)

def _silent_error_handler(filename, line, function, err, fmt):
    pass

_c_handler = _ERROR_HANDLER_FUNC(_silent_error_handler)

try:
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(_c_handler)
except OSError:
    pass

class _SuppressStderr:
    def __enter__(self):
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        self._old     = os.dup(2)
        os.dup2(self._devnull, 2)
        os.close(self._devnull)
        return self

    def __exit__(self, *_):
        os.dup2(self._old, 2)
        os.close(self._old)

# Listening duration config

WAIT_FOR_SPEECH_TIMEOUT = 15    # seconds to wait for speech to START
                                # set None to wait forever

MAX_PHRASE_DURATION     = 120    # seconds to keep recording AFTER speech starts
                                # set None for no hard cap (records until silence)

# Recognizer sensitivity tuning

recognizer = sr.Recognizer()

recognizer.energy_threshold             = 200    # lower → hears quieter sounds
recognizer.dynamic_energy_threshold     = True   # auto-adapts to room noise
recognizer.dynamic_energy_adjustment_damping = 0.10
recognizer.dynamic_energy_ratio         = 1.2

# KEY: pause_threshold - how many seconds of silence ends the recording.
# Increase this so brief pauses mid-sentence don't cut you off.
recognizer.pause_threshold              = 3.0    # waits 3s of silence after sentence ends    # 2s between words
recognizer.phrase_threshold             = 0.1
recognizer.non_speaking_duration        = 0.3    # silence padding kept at edges

# Listen

try:
    with _SuppressStderr():
        mic = sr.Microphone()

    with mic as source:
        print("Calibrating for ambient noise... please wait (1 second).")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print(f"Energy threshold set to: {recognizer.energy_threshold:.1f}")
        print(f"Will wait up to {WAIT_FOR_SPEECH_TIMEOUT}s for you to start speaking.")
        print(f"Max recording length: {MAX_PHRASE_DURATION}s after speech begins.")
        print(f"Stops {recognizer.pause_threshold}s after you finish speaking.\n")

        print("Listening... Speak now! (even softly)\n")
        audio = recognizer.listen(
            source,
            timeout         = WAIT_FOR_SPEECH_TIMEOUT,
            phrase_time_limit = MAX_PHRASE_DURATION
        )

    print("Processing...")
    text = recognizer.recognize_google(audio, language="en-IN")
    print(f"\nYou said: {text}")

except sr.WaitTimeoutError:
    print(f"No speech detected in {WAIT_FOR_SPEECH_TIMEOUT}s. Please try again.")
except sr.UnknownValueError:
    print("Could not understand the audio. Please try again.")
except sr.RequestError as e:
    print(f"Could not reach Google Speech Recognition service: {e}")
except KeyboardInterrupt:
    print("\nStopped by user.")