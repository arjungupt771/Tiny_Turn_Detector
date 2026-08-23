from collections import Counter
from datasets import load_dataset


DATASET_NAME = "pipecat-ai/smart-turn-data-v3.2-train"
MAX_SAMPLES = 20_000


def main():
    print("Loading dataset in streaming mode...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    print("Dataset loaded.")

    # -----------------------------
    # Counters
    # -----------------------------

    total = 0

    endpoint_counts = Counter()
    language_counts = Counter()
    midfiller_counts = Counter()
    endfiller_counts = Counter()
    synthetic_counts = Counter()
    source_counts = Counter()

    # Cross-statistics
    endpoint_by_language = Counter()
    endpoint_by_midfiller = Counter()
    endpoint_by_endfiller = Counter()
    endpoint_by_synthetic = Counter()

    print(f"\nScanning first {MAX_SAMPLES:,} samples...")

    # -----------------------------
    # Iterate through dataset
    # -----------------------------

    for i, sample in enumerate(dataset):

        if i >= MAX_SAMPLES:
            break

        total += 1

        # Extract metadata
        endpoint = sample["endpoint_bool"]
        language = sample["language"]
        midfiller = sample["midfiller"]
        endfiller = sample["endfiller"]
        synthetic = sample["synthetic"]
        source = sample["dataset"]

        # -----------------------------
        # Basic distributions
        # -----------------------------

        endpoint_counts[endpoint] += 1
        language_counts[language] += 1
        midfiller_counts[midfiller] += 1
        endfiller_counts[endfiller] += 1
        synthetic_counts[synthetic] += 1
        source_counts[source] += 1

        # -----------------------------
        # Cross statistics
        # -----------------------------

        endpoint_by_language[(language, endpoint)] += 1
        endpoint_by_midfiller[(midfiller, endpoint)] += 1
        endpoint_by_endfiller[(endfiller, endpoint)] += 1
        endpoint_by_synthetic[(synthetic, endpoint)] += 1

        # Progress
        if total % 1_000 == 0:
            print(f"Processed {total:,} samples...")

    # -----------------------------
    # Results
    # -----------------------------

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"\nSamples analyzed: {total:,}")

    # -----------------------------
    # Endpoint distribution
    # -----------------------------

    print("\n--- Endpoint distribution ---")

    for label, count in endpoint_counts.items():

        percentage = count / total * 100

        name = "END" if label else "CONTINUE"

        print(
            f"{name:10s}: "
            f"{count:10,} "
            f"({percentage:.2f}%)"
        )

    # -----------------------------
    # Languages
    # -----------------------------

    print("\n--- Languages ---")

    print(f"Number of languages: {len(language_counts)}")

    for language, count in language_counts.most_common():

        percentage = count / total * 100

        print(
            f"{language:10s}: "
            f"{count:10,} "
            f"({percentage:.2f}%)"
        )

    # -----------------------------
    # Mid fillers
    # -----------------------------

    print("\n--- Mid fillers ---")

    for value, count in midfiller_counts.items():

        percentage = count / total * 100

        print(
            f"{str(value):10s}: "
            f"{count:10,} "
            f"({percentage:.2f}%)"
        )

    # -----------------------------
    # End fillers
    # -----------------------------

    print("\n--- End fillers ---")

    for value, count in endfiller_counts.items():

        percentage = count / total * 100

        print(
            f"{str(value):10s}: "
            f"{count:10,} "
            f"({percentage:.2f}%)"
        )

    # -----------------------------
    # Synthetic
    # -----------------------------

    print("\n--- Synthetic ---")

    for value, count in synthetic_counts.items():

        percentage = count / total * 100

        print(
            f"{str(value):10s}: "
            f"{count:10,} "
            f"({percentage:.2f}%)"
        )

    # -----------------------------
    # Source datasets
    # -----------------------------

    print("\n--- Source datasets ---")

    for source, count in source_counts.most_common():

        percentage = count / total * 100

        print(
            f"{source:20s}: "
            f"{count:10,} "
            f"({percentage:.2f}%)"
        )

    # -----------------------------
    # Endpoint by midfiller
    # -----------------------------

    print("\n--- Endpoint by midfiller ---")

    for midfiller, _ in midfiller_counts.items():

        end_count = endpoint_by_midfiller[(midfiller, True)]
        continue_count = endpoint_by_midfiller[(midfiller, False)]

        total_group = end_count + continue_count

        if total_group == 0:
            continue

        end_percentage = end_count / total_group * 100

        print(
            f"midfiller={midfiller}: "
            f"END={end_count:,}, "
            f"CONTINUE={continue_count:,}, "
            f"END%={end_percentage:.2f}%"
        )

    # -----------------------------
    # Endpoint by endfiller
    # -----------------------------

    print("\n--- Endpoint by endfiller ---")

    for endfiller, _ in endfiller_counts.items():

        end_count = endpoint_by_endfiller[(endfiller, True)]
        continue_count = endpoint_by_endfiller[(endfiller, False)]

        total_group = end_count + continue_count

        if total_group == 0:
            continue

        end_percentage = end_count / total_group * 100

        print(
            f"endfiller={endfiller}: "
            f"END={end_count:,}, "
            f"CONTINUE={continue_count:,}, "
            f"END%={end_percentage:.2f}%"
        )

    # -----------------------------
    # Endpoint by synthetic
    # -----------------------------

    print("\n--- Endpoint by synthetic ---")

    for synthetic, _ in synthetic_counts.items():

        end_count = endpoint_by_synthetic[(synthetic, True)]
        continue_count = endpoint_by_synthetic[(synthetic, False)]

        total_group = end_count + continue_count

        if total_group == 0:
            continue

        end_percentage = end_count / total_group * 100

        print(
            f"synthetic={synthetic}: "
            f"END={end_count:,}, "
            f"CONTINUE={continue_count:,}, "
            f"END%={end_percentage:.2f}%"
        )

    # -----------------------------
    # Endpoint by language
    # -----------------------------

    print("\n--- Endpoint by language ---")

    for language, count in language_counts.most_common():

        end_count = endpoint_by_language[(language, True)]
        continue_count = endpoint_by_language[(language, False)]

        total_group = end_count + continue_count

        if total_group == 0:
            continue

        end_percentage = end_count / total_group * 100

        print(
            f"{language:10s}: "
            f"END={end_count:,}, "
            f"CONTINUE={continue_count:,}, "
            f"END%={end_percentage:.2f}%"
        )


if __name__ == "__main__":
    main()