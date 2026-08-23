from pathlib import Path

import soundfile as sf
from datasets import load_dataset


DATASET_NAME = "pipecat-ai/smart-turn-data-v3.2-train"

OUTPUT_DIR = Path("data/samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_audio(sample, filename):
    """
    Extract audio from a Hugging Face AudioDecoder
    and save it as a WAV file.
    """

    audio = sample["audio"]

    print(f"\nAudio object type: {type(audio)}")

    # Inspect the decoder object
    print("Audio object:", audio)

    # Decode audio
    decoded = audio.get_all_samples()

    print("Decoded audio type:", type(decoded))

    waveform = decoded.data
    sample_rate = decoded.sample_rate

    print("Waveform shape:", waveform.shape)
    print("Sample rate:", sample_rate)

    # Convert torch tensor -> numpy
    waveform = waveform.numpy()

    # Hugging Face audio can be [channels, samples]
    if waveform.ndim == 2:
        waveform = waveform.T

    output_path = OUTPUT_DIR / filename

    sf.write(
        output_path,
        waveform,
        sample_rate,
    )

    print(f"Saved: {output_path}")


def main():

    print("Loading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    print("Dataset loaded.")

    found = {
        "end": False,
        "continue": False,
        "midfiller_continue": False,
        "midfiller_end": False,
        "endfiller_end": False,
    }

    for sample in dataset:

        endpoint = sample["endpoint_bool"]
        midfiller = sample["midfiller"]
        endfiller = sample["endfiller"]

        # ----------------------------------
        # Normal END
        # ----------------------------------

        if (
            endpoint is True
            and midfiller is not True
            and endfiller is not True
            and not found["end"]
        ):
            print("\nFound END example")

            print("Language:", sample["language"])
            print("Endpoint:", endpoint)
            print("Midfiller:", midfiller)
            print("Endfiller:", endfiller)
            print("Synthetic:", sample["synthetic"])
            print("Text:", sample["spoken_text"])

            save_audio(sample, "end.wav")

            found["end"] = True

        # ----------------------------------
        # Normal CONTINUE
        # ----------------------------------

        elif (
            endpoint is False
            and midfiller is not True
            and endfiller is not True
            and not found["continue"]
        ):
            print("\nFound CONTINUE example")

            print("Language:", sample["language"])
            print("Endpoint:", endpoint)
            print("Midfiller:", midfiller)
            print("Endfiller:", endfiller)
            print("Synthetic:", sample["synthetic"])
            print("Text:", sample["spoken_text"])

            save_audio(sample, "continue.wav")

            found["continue"] = True

        # ----------------------------------
        # Midfiller + CONTINUE
        # ----------------------------------

        elif (
            endpoint is False
            and midfiller is True
            and not found["midfiller_continue"]
        ):
            print("\nFound MIDFILLER + CONTINUE example")

            print("Language:", sample["language"])
            print("Endpoint:", endpoint)
            print("Midfiller:", midfiller)
            print("Endfiller:", endfiller)
            print("Synthetic:", sample["synthetic"])
            print("Text:", sample["spoken_text"])

            save_audio(
                sample,
                "midfiller_continue.wav",
            )

            found["midfiller_continue"] = True

        # ----------------------------------
        # Midfiller + END
        # ----------------------------------

        elif (
            endpoint is True
            and midfiller is True
            and not found["midfiller_end"]
        ):
            print("\nFound MIDFILLER + END example")

            print("Language:", sample["language"])
            print("Endpoint:", endpoint)
            print("Midfiller:", midfiller)
            print("Endfiller:", endfiller)
            print("Synthetic:", sample["synthetic"])
            print("Text:", sample["spoken_text"])

            save_audio(
                sample,
                "midfiller_end.wav",
            )

            found["midfiller_end"] = True

        # ----------------------------------
        # Endfiller + END
        # ----------------------------------

        elif (
            endpoint is True
            and endfiller is True
            and not found["endfiller_end"]
        ):
            print("\nFound ENDFILLER + END example")

            print("Language:", sample["language"])
            print("Endpoint:", endpoint)
            print("Midfiller:", midfiller)
            print("Endfiller:", endfiller)
            print("Synthetic:", sample["synthetic"])
            print("Text:", sample["spoken_text"])

            save_audio(
                sample,
                "endfiller_end.wav",
            )

            found["endfiller_end"] = True

        # ----------------------------------
        # Stop when all examples found
        # ----------------------------------

        if all(found.values()):
            break

    print("\n" + "=" * 50)
    print("SAMPLING COMPLETE")
    print("=" * 50)

    for name, status in found.items():
        print(f"{name:25s}: {'FOUND' if status else 'NOT FOUND'}")


if __name__ == "__main__":
    main()