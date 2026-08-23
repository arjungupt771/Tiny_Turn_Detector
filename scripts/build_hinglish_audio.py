"""
Task 8 (completion pass) -- attach real (synthetic) audio to the Hinglish
evaluation manifest.

Previously: `data/hinglish_eval/manifest.csv` had 41 hand-written, text-only
Hinglish/code-switch/filler/pause/ambiguous-ending examples with
has_audio=False for every row (no TTS engine was available in the sandbox).

This pass installs `espeak-ng` via apt (archive.ubuntu.com is on the network
allowlist -- no huggingface.co or other blocked host needed) and uses it to
synthesize a .wav clip for every manifest row, replacing conceptual "..."
pause markers with SSML <break> tags whose duration matches the row's
pause_length_conceptual (short/medium/long), and resampling to 16 kHz mono
to match the project's stated preprocessing convention.

Explicit limitation (documented, not hidden): espeak-ng is a formant/rule-
based synthesizer, not a neural TTS system, and has no dedicated Hinglish or
Hindi-accented-English voice in this environment -- it is speaking the
romanized text with an English voice. The prosody is much more robotic and
uniform than real human Hinglish speech, particularly for code-switched
Hindi words. This is still labeled `synthetic_hinglish=True` (never
presented as real human recordings), and it does NOT change the other
limitation: the model still cannot be evaluated on this audio in this
sandbox, because the frozen Whisper-tiny encoder's pretrained weights
cannot be downloaded here (no route to huggingface.co). `model_evaluated`
stays False for every row.
"""
import csv
import io
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

MANIFEST_PATH = Path("data/hinglish_eval/manifest.csv")
AUDIO_DIR = Path("data/hinglish_eval/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16000
ESPEAK_VOICE = "en"

PAUSE_MS = {
    "short": 300,
    "medium": 700,
    "long": 1500,
}


def build_ssml(text: str, pause_length: str) -> str:
    """Replace literal '...' pause markers with SSML <break> tags sized to
    the row's conceptual pause length. Rows with pause_length_conceptual ==
    'none' just get espeak-ng's natural punctuation-driven pausing."""
    if pause_length in PAUSE_MS and "..." in text:
        ms = PAUSE_MS[pause_length]
        text = text.replace("...", f'<break time="{ms}ms"/>')
    # Escape raw ampersands for SSML validity; text has none currently but
    # this keeps the generator robust if future rows introduce one.
    text = text.replace("&", "and")
    return f"<speak>{text}</speak>"


def synthesize(ssml: str, out_path: Path) -> float:
    """Run espeak-ng, capture stdout wav bytes, resample to TARGET_SR mono,
    write final wav, return duration in seconds."""
    raw_path = out_path.with_suffix(".raw.wav")
    subprocess.run(
        ["espeak-ng", "-m", "-v", ESPEAK_VOICE, "-s", "150", ssml, "-w", str(raw_path)],
        check=True, capture_output=True,
    )
    data, sr = sf.read(raw_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        gcd = np.gcd(sr, TARGET_SR)
        data = resample_poly(data, TARGET_SR // gcd, sr // gcd)
    data = data.astype(np.float32)
    sf.write(out_path, data, TARGET_SR, subtype="PCM_16")
    raw_path.unlink(missing_ok=True)
    return len(data) / TARGET_SR


def main():
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys())
    for col in ("audio_path", "audio_duration_sec", "tts_engine"):
        if col not in fieldnames:
            fieldnames.append(col)

    n_ok = 0
    for row in rows:
        out_path = AUDIO_DIR / f"{row['id']}.wav"
        ssml = build_ssml(row["text"], row["pause_length_conceptual"])
        try:
            duration = synthesize(ssml, out_path)
            row["has_audio"] = "True"
            row["audio_path"] = str(out_path)
            row["audio_duration_sec"] = f"{duration:.3f}"
            row["tts_engine"] = "espeak-ng (formant synthesis, English voice, romanized Hinglish text)"
            row["model_evaluated"] = "False"  # still true -- see module docstring
            n_ok += 1
        except subprocess.CalledProcessError as e:
            row["has_audio"] = "False"
            row["audio_path"] = ""
            row["audio_duration_sec"] = ""
            row["tts_engine"] = f"FAILED: {e}"

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Synthesized audio for {n_ok}/{len(rows)} rows -> {AUDIO_DIR}/")
    durations = [float(r["audio_duration_sec"]) for r in rows if r["audio_duration_sec"]]
    print(f"Duration range: {min(durations):.2f}s - {max(durations):.2f}s, mean {np.mean(durations):.2f}s")


if __name__ == "__main__":
    main()
