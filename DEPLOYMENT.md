# Deployment — Hugging Face Space

**Status: NOT deployed.** No live Space exists. This documents exactly why and
the exact steps to deploy for real once the blocker is gone.

## Why it wasn't deployed from this environment

This project was completed in a sandboxed container whose network egress is
restricted to a fixed allowlist (pypi.org, npmjs.com, github.com, crates.io,
and a few package-registry mirrors — see the container's network config).
`huggingface.co` is **not** on that list:

```
$ curl -I https://huggingface.co
HTTP/2 403
x-deny-reason: host_not_allowed
```

Creating or pushing to a Hugging Face Space requires the HF Hub API
(`huggingface_hub` calls `huggingface.co`/`hf.co`) and an HF access token,
neither of which is reachable/available here. Deploying is therefore a task
for an environment with normal internet access and HF credentials, not
something that can be faked or simulated honestly from this sandbox.

## What's already deployment-ready in this repo

- `app/app.py` — the Gradio demo, already runnable locally via `python app/app.py`
  (needs `torch`/`transformers` installed and network access to download
  `openai/whisper-tiny` on first run).
- `requirements.txt` — already lists everything the Space needs (torch,
  transformers, gradio, scikit-learn, onnxruntime, etc.).
- `models/turn_detector_whisper.joblib` — the trained classifier head to ship
  alongside the Space.
- `checkpoints/turn_classifier_head_fp32.onnx` / `..._int8.onnx` — exported
  ONNX versions of the classifier head (see `reports/onnx_export_verification.json`
  and `reports/int8_quantization.json` for verification numbers), useful if the
  Space should run the head via onnxruntime instead of sklearn directly.

## Exact steps to deploy, once network + HF credentials are available

```bash
# 1. Install the HF Hub CLI (already in most environments with `pip install huggingface_hub`)
pip install huggingface_hub

# 2. Log in (requires an HF account + access token from https://huggingface.co/settings/tokens)
huggingface-cli login

# 3. Create the Space (Gradio SDK)
huggingface-cli repo create smart-turn-hinglish --type space --space_sdk gradio

# 4. Clone it locally and copy in the app
git clone https://huggingface.co/spaces/<your-username>/smart-turn-hinglish
cd smart-turn-hinglish
cp ../Audio_ml_model/app/app.py .
cp ../Audio_ml_model/requirements.txt .
cp -r ../Audio_ml_model/src .
mkdir -p models && cp ../Audio_ml_model/models/turn_detector_whisper.joblib models/
cp -r ../Audio_ml_model/data/samples data/samples  # optional, for example clips

# 5. Push
git add .
git commit -m "Deploy turn detector demo"
git push

# 6. Test the deployed Space
# Open https://huggingface.co/spaces/<your-username>/smart-turn-hinglish
# and run the sample clips in data/samples/ plus a live recording.
```

## What "testing the deployed application" would additionally verify

Once live, confirm: (a) the Space builds without dependency errors, (b) a
recorded/uploaded clip returns a verdict + probability + latency within a
few seconds (cold start on free-tier CPU Spaces can be 10-30s for the first
request while Whisper-tiny loads), (c) the displayed latency number matches
roughly what `reports/latency_benchmark.json` would show for a full,
network-enabled environment — not just the classifier-head-only number
recorded in this sandbox (see `PROJECT_BLOCKERS.md`).
