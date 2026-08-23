"""
Task 8 — Build a real Hinglish evaluation set.

HONEST SCOPE, READ BEFORE USING THIS FILE:
This produces a TEXT-ONLY manifest of synthetic Hinglish/code-switched
example utterances with hand-labeled ground truth (END/CONTINUE) and
filler/pause/ambiguity metadata. It does NOT contain audio, and no
model was evaluated against it, because:

1. Generating audio (real recordings or TTS) requires either manual
   collection or a TTS service, neither of which is available in this
   sandbox (no network route to huggingface.co, no TTS API configured).
2. Running the model against audio requires the Whisper-tiny encoder,
   which also can't be downloaded here.

This is explicitly a "clearly-labeled SYNTHETIC Hinglish evaluation"
per the brief's own fallback instruction, taken one step further: it
is TEXT-only synthetic, not audio-synthetic, and this file says so
everywhere the data is referenced. `reports/hinglish_eval_results.csv`
does NOT exist because there is no model output to report -- fabricating
one would violate the "do not fabricate Hinglish results" rule.

To complete this task for real: (a) generate TTS audio for each row
below (e.g. via a TTS API in an environment with network access), or
collect real human Hinglish recordings and re-label against them, then
(b) run `src/inference.py` against the resulting audio manifest once
the Whisper-tiny encoder is available, and (c) merge results into
`reports/targeted_metrics.csv`.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "hinglish_eval" / "manifest.csv"

# (text, label, category, filler_position, filler_words, pause_length, note)
# label: 1 = END (turn genuinely over), 0 = CONTINUE (speaker not done)
# filler_position: none | mid | end
# pause_length: none | short | medium | long  (conceptual only -- no audio timing exists)
ROWS = [
    # --- Clean code-switching, clear END ---
    ("Haan I'll send it tomorrow.", 1, "code_switch", "none", "", "none", "clean END, no filler"),
    ("Theek hai, that works for me.", 1, "code_switch", "none", "", "none", "clean END"),
    ("Ok cool, main baad mein call karta hoon.", 1, "code_switch", "none", "", "none", "clean END"),
    ("Done, ye final hai.", 1, "code_switch", "none", "", "none", "clean END"),
    ("Sounds good, kal milte hain.", 1, "code_switch", "none", "", "none", "clean END"),
    # --- Clean code-switching, clear CONTINUE ---
    ("Haan I'll send it tomorrow but pehle mujhe", 0, "code_switch", "none", "", "none", "cut mid-sentence, clearly continuing"),
    ("Wait ek minute, mujhe check karna hai ki", 0, "code_switch", "mid", "wait", "none", "trails into dependent clause"),
    ("Toh phir hum kal kar lenge, lekin agar", 0, "code_switch", "none", "", "none", "conditional clause left open"),
    ("Main basically ye keh raha tha ki jab humne", 0, "code_switch", "mid", "basically", "none", "clearly unfinished"),
    ("Actually mujhe lagta hai ki shayad", 0, "code_switch", "mid", "actually", "none", "trails off before conclusion"),
    # --- Mid-filler, END ---
    ("Um, so I think that's basically it.", 1, "filler", "mid", "um,basically", "short", "filler mid, ends cleanly"),
    ("Matlab, ye hi tha jo main bolna chahta tha.", 1, "filler", "mid", "matlab", "none", "filler mid, END"),
    ("Toh, bas itna hi tha.", 1, "filler", "mid", "toh", "none", "filler mid, END"),
    ("Acha, so that's the plan then.", 1, "filler", "mid", "acha", "none", "filler mid, END"),
    ("Uh, ye sab kaafi hai for now.", 1, "filler", "mid", "uh", "short", "filler mid, END"),
    # --- Mid-filler, CONTINUE (the hard case: filler right before real end) ---
    ("Um, so I think that we should also consider", 0, "filler", "mid", "um", "none", "filler mid, but sentence continues"),
    ("Matlab, jo maine kaha tha wo actually", 0, "filler", "mid", "matlab,actually", "none", "double filler, still continuing"),
    ("Hmm, ye theek hai but humein bhi dekhna hoga ki", 0, "filler", "mid", "hmm", "short", "filler then subordinate clause"),
    ("Like, I was thinking maybe we could also", 0, "filler", "mid", "like", "none", "filler then continues"),
    ("Basically what happened was ki jab hum pahunche toh", 0, "filler", "mid", "basically", "none", "filler then narrative continues"),
    # --- End-filler, ambiguous by construction (filler immediately before true end) ---
    ("That's the update, matlab.", 1, "filler", "end", "matlab", "none", "end-filler right before genuine END"),
    ("Ye sab kar denge, basically.", 1, "filler", "end", "basically", "none", "end-filler, genuine END"),
    ("I think we're done here, toh.", 1, "filler", "end", "toh", "none", "end-filler, genuine END -- easy to mis-flag as continuing"),
    ("Haan bas, acha.", 1, "filler", "end", "acha", "short", "end-filler, genuine END"),
    ("So that's it, like.", 1, "filler", "end", "like", "none", "end-filler, genuine END"),
    # --- Pauses of varying conceptual length (no real timing without audio) ---
    ("Main... ... ...socha ki hum kal milenge.", 1, "pause", "none", "", "long", "long mid-utterance pause then completes"),
    ("Wait... ... ek second.", 0, "pause", "mid", "wait", "medium", "medium pause, clearly still speaking"),
    ("Haan toh... ...theek hai, done.", 1, "pause", "mid", "toh", "short", "short pause then END"),
    ("Mujhe lagta hai ki... ... ... shayad hum change kar sakte hain, lekin", 0, "pause", "mid", "", "long", "long pause then continues into new clause"),
    ("Ok so...done.", 1, "pause", "none", "", "short", "short pause then abrupt END"),
    # --- Ambiguous endings: sounds complete but continues ---
    ("I think we're good", 0, "ambiguous", "none", "", "none", "sounds complete, but no terminal marker and speaker continues"),
    ("That should work for", 0, "ambiguous", "none", "", "none", "trails into dangling preposition"),
    ("Humne wo kaam kar diya", 0, "ambiguous", "none", "", "none", "sounds like a complete Hindi clause but continues to elaborate"),
    ("So the plan is basically to just", 0, "ambiguous", "mid", "basically", "none", "sounds like it's building to a point, cuts off"),
    ("Yeah I think that's", 0, "ambiguous", "none", "", "none", "classic false-END trap -- sounds finished, isn't"),
    # --- Ambiguous endings: sounds unfinished but is actually done ---
    ("And that's basically the whole thing, actually.", 1, "ambiguous", "end", "basically,actually", "none", "double filler makes it feel unfinished, but it's a genuine END"),
    ("So toh bas, done, that's all, matlab.", 1, "ambiguous", "end", "toh,matlab", "none", "filler-heavy, easy to mis-flag as continuing, is genuine END"),
    # --- No filler, plain END / CONTINUE controls ---
    ("The meeting starts at 5 PM tomorrow.", 1, "no_filler", "none", "", "none", "plain declarative END"),
    ("Mujhe kal subah jaana hai station.", 1, "no_filler", "none", "", "none", "plain declarative END"),
    ("I was thinking that we should go to the", 0, "no_filler", "none", "", "none", "plain unfinished CONTINUE"),
    ("Jab main ghar pahuncha toh dekha ki", 0, "no_filler", "none", "", "none", "plain unfinished CONTINUE"),
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "text", "label", "category", "filler_position",
            "filler_words", "pause_length_conceptual", "note",
            "synthetic_hinglish", "has_audio", "model_evaluated",
        ])
        for i, (text, label, category, fpos, fwords, plen, note) in enumerate(ROWS):
            w.writerow([
                f"hinglish_text_{i:03d}", text, label, category, fpos, fwords,
                plen, note, True, False, False,
            ])
    print(f"Wrote {len(ROWS)} text-only Hinglish rows to {OUT}")
    print("has_audio=False and model_evaluated=False for every row -- see module docstring.")


if __name__ == "__main__":
    main()
