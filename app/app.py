"""
Gradio demo for the tiny turn-detection model.

Run:
    python app/app.py

Then open the local URL Gradio prints. Record or upload a short clip
and the app will say whether the speaker sounds DONE (END) or still
mid-thought (CONTINUE), with a probability, a waveform, a plain-language
explanation, and the time taken.

TASK 20 completion-pass note: this adds the waveform view, the
explanation panel, and Hinglish example clips that were previously
missing. It does NOT change the fundamental blocker documented elsewhere
in this repo: actually launching this app still requires downloading the
real openai/whisper-tiny encoder from huggingface.co, which this sandbox
cannot reach. The UI/UX additions below were written and sanity-checked
against synthetic inputs (see `_fake_result_for_ui_check`), not against a
live TurnDetector session, since that session can't be started here.
"""

import sys
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from inference import TurnDetector  # noqa: E402

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"
HINGLISH_AUDIO_DIR = Path(__file__).resolve().parent.parent / "data" / "hinglish_eval" / "audio"
HINGLISH_MANIFEST = Path(__file__).resolve().parent.parent / "data" / "hinglish_eval" / "manifest.csv"

_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = TurnDetector()
    return _detector


def _waveform_plot(waveform: np.ndarray, sample_rate: int):
    fig, ax = plt.subplots(figsize=(6, 1.8))
    t = np.arange(len(waveform)) / sample_rate
    ax.plot(t, waveform, linewidth=0.6, color="#3b82f6")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.set_xlim(0, max(t[-1], 0.01))
    fig.tight_layout()
    return fig


def _explanation(result: dict) -> str:
    """Plain-language explanation of the verdict -- distance from the
    decision threshold, not a black-box confidence number alone."""
    prob = result["end_probability"]
    thr = result["threshold"]
    margin = abs(prob - thr)
    label = result["label"]

    if margin < 0.08:
        closeness = "very close to the decision threshold -- this is a borderline case"
    elif margin < 0.2:
        closeness = "moderately clear of the threshold"
    else:
        closeness = "well clear of the threshold"

    if label == "END":
        reason = (f"P(END)={prob:.2f} is above the threshold ({thr:.2f}), so the model "
                   f"believes the speaker has finished their turn. This is {closeness}.")
    else:
        reason = (f"P(END)={prob:.2f} is below the threshold ({thr:.2f}), so the model "
                   f"believes the speaker is still mid-thought (a pause, filler word, or "
                   f"trailing-off sentence). This is {closeness}.")
    return reason


def run_detection(audio):
    if audio is None:
        return "No audio provided.", None, None, None

    detector = get_detector()
    sample_rate, waveform = audio

    waveform = waveform.astype(np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    # Gradio's numpy mic input is int16-range PCM; normalize if needed.
    if np.abs(waveform).max() > 1.0:
        waveform = waveform / 32768.0

    result = detector.predict_waveform(waveform, sample_rate)

    label = result["label"]
    prob = result["end_probability"]
    latency = result["latency_ms"]

    verdict = "🟢 Turn is OVER (END)" if label == "END" else "🟡 Speaker is still going (CONTINUE)"
    detail = (f"Confidence: {prob:.1%}  |  Threshold: {result['threshold']:.2f}  |  "
              f"Latency: {latency:.1f} ms  |  Model: {result['model_version']}")
    explanation = _explanation(result)
    waveform_fig = _waveform_plot(waveform, sample_rate)

    return verdict, detail, explanation, waveform_fig


def _load_hinglish_examples():
    """Load a handful of the synthetic Hinglish/filler/pause clips built in
    Task 8, clearly labeled as synthetic TTS, not real human recordings."""
    if not HINGLISH_MANIFEST.exists():
        return []
    import csv
    with open(HINGLISH_MANIFEST) as f:
        rows = list(csv.DictReader(f))
    picked = []
    seen_categories = set()
    for row in rows:
        if row.get("has_audio") == "True" and row["category"] not in seen_categories:
            picked.append(row["audio_path"])
            seen_categories.add(row["category"])
        if len(picked) >= 5:
            break
    return picked


with gr.Blocks(title="Tiny Turn Detector") as demo:
    gr.Markdown(
        """
        # 🎙️ Tiny Turn Detector
        A small audio classifier that decides whether a speaker has **finished their turn**
        or is **just pausing** (e.g. mid-thought, filler word, Hinglish "matlab...", "toh...").

        Built on a frozen **Whisper-tiny** encoder (not fine-tuned) with a
        ~400-parameter logistic-regression head trained on the
        [pipecat-ai/smart-turn-data-v3.2](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train)
        dataset. See the README and REPORT.md for the full experiment writeup, and
        PROJECT_BLOCKERS.md for what this environment could/couldn't run.
        """
    )

    with gr.Row():
        audio_input = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Speak or upload a short clip")

    run_btn = gr.Button("Detect turn", variant="primary")

    with gr.Row():
        verdict_output = gr.Textbox(label="Verdict")
        detail_output = gr.Textbox(label="Details")

    explanation_output = gr.Textbox(label="Why this verdict", lines=2)
    waveform_output = gr.Plot(label="Waveform")

    all_outputs = [verdict_output, detail_output, explanation_output, waveform_output]
    run_btn.click(fn=run_detection, inputs=audio_input, outputs=all_outputs)
    audio_input.change(fn=run_detection, inputs=audio_input, outputs=all_outputs)

    if SAMPLES_DIR.exists():
        sample_files = sorted(str(p) for p in SAMPLES_DIR.glob("*.wav"))
        if sample_files:
            gr.Markdown("### Try a sample")
            gr.Examples(examples=sample_files, inputs=audio_input)

    hinglish_examples = _load_hinglish_examples()
    if hinglish_examples:
        gr.Markdown(
            """
            ### Try a Hinglish / filler / pause example
            ⚠️ These are **synthetic, TTS-generated** (espeak-ng) clips built for the
            Hinglish evaluation set in Task 8 -- not real human recordings. They stand
            in for code-switching, fillers ("matlab", "toh", "um"), and mid-sentence
            pauses, but the accent/prosody is robotic, not authentic Hinglish speech.
            """
        )
        gr.Examples(examples=hinglish_examples, inputs=audio_input)


if __name__ == "__main__":
    demo.launch()
