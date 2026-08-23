from pathlib import Path

import librosa
import soundfile as sf


AUDIO_DIR = Path("data/samples")


def analyze_audio(path):

    audio, sample_rate = librosa.load(
        path,
        sr=None,
        mono=True,
    )

    duration = len(audio) / sample_rate

    print(f"\nFile: {path.name}")
    print(f"Sample rate: {sample_rate}")
    print(f"Samples: {len(audio):,}")
    print(f"Duration: {duration:.3f} sec")
    print(f"Min amplitude: {audio.min():.4f}")
    print(f"Max amplitude: {audio.max():.4f}")
    print(f"Mean amplitude: {audio.mean():.4f}")
    print(f"RMS energy: {librosa.feature.rms(y=audio).mean():.6f}")


def main():

    files = sorted(AUDIO_DIR.glob("*.wav"))

    if not files:
        print("No WAV files found.")
        return

    print(f"Found {len(files)} audio files.")

    for path in files:
        analyze_audio(path)


if __name__ == "__main__":
    main()