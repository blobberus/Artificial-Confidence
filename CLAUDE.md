# CLAUDE.md — LLM Calibration Study

Project spec distilled from the source artifact "Does It Know What It Doesn't Know"
(https://claude.ai/artifact/Kj3jRaGDi96t3ounRT7Bvd). This file is meant to be self-sufficient: the
user should be able to say "do step b3", "protocol §6", or "the S2 comparison" and I know exactly what
that means without re-reading the artifact. If this file and the user's instructions disagree, the user wins.

## 1. Goal

**Question:** Does a small open language model know what it doesn't know?

Measure whether a locally run open model's *stated confidence* (the softmax probability it assigns to
its chosen answer on a multiple-choice question) matches how often it is actually right — then show
how **unstable** that measurement is. The deliverable is one public repo (the GitHub repo
`blobberus/Artificial-Confidence`; the artifact calls it `llm-calibration-study`) that ~15 research labs
each have a reason to care about. The README is the real deliverable: people will spend ~90 seconds on it.

- **Models (both 4-bit nf4, loaded from Hugging Face safetensors — never GGUF, never both in VRAM at once):**
  - `stock`: `Qwen/Qwen3-8B` — https://huggingface.co/Qwen/Qwen3-8B
  - `uncensored`: `huihui-ai/Huihui-Qwen3-8B-abliterated-v2` — https://huggingface.co/huihui-ai/Huihui-Qwen3-8B-abliterated-v2
    (base `Qwen/Qwen3-8B`, same architecture/param count; **abliterated** = weights edited to remove refusal
    behaviour, not trained — describe it accurately in the README rather than as a "fine-tune").
  - Run one variant at a time (load → score → free GPU); the comparison is made afterwards from `scores.csv`.
  - Tokenizers of the two repos were checked and are identical (same letter token ids and chat-template output).
- **Hook:** GPT-4's technical report showed the pre-trained model was well calibrated and RLHF made it
  worse. Comparing stock instruct vs. uncensored fine-tune asks: *does removing alignment tuning move
  calibration back?* Nobody else will have this comparison.
- **Compute:** RTX 4070 (12 GB), all local. **No training / fine-tuning at all** — read a model, don't change it.
- **Effort budget:** ~30 h, build window 16–28 Sep 2026.
- **The 28 September rule:** on Mon 28 Sep 2026 the project is finished, whatever its state. After that,
  no analysis improvements — only outreach (read two papers per lab, write two sentences). Resist scope
  creep; "you want to keep improving it past 28 Sep" is rated the *very-high-likelihood* actual risk.

The user's statistics background (scoring rules, estimators, bootstrap) is the point of the project.
The PyTorch is ~20 lines: load model → forward pass → softmax over four token logits → CSV.

## 2. Scope — hold this line

**In scope**
- One model family, two variants: stock instruct + uncensored fine-tune
- 500–1000 four-option questions from ONE multiple-choice benchmark (default: ARC-Easy test, first 800 usable)
- Three readouts of the model's probability (letter token, sequence likelihood, length-normalised sequence likelihood)
- ECE, Brier, Murphy decomposition — all with bootstrap intervals
- Comparisons: bin count, option order, readout, model variant (see §6 — the artifact says "three comparisons"
  in the scope list but its protocol table has four)
- Three figures + a README readable in 90 seconds

**Explicitly out of scope (do not build, do not suggest building)**
- Any training or fine-tuning
- A second benchmark (one is enough)
- Free-text calibration
- Recalibration methods (temperature / Platt scaling) — name as "obvious next step I didn't take"
- Conformal prediction (a conversation for a lab, not a build)
- Leaderboard numbers / beating anything

## 3. Definition of done (protocol §10)

All true on 28 Sep:
1. Notebook runs top-to-bottom from a clean kernel with no manual intervention
2. Every reported number has a bootstrap interval next to it
3. Three figures rendered at 200 dpi and embedded in the README
4. "What I could not resolve" names at least three of the four struggles with the user's *actual* numbers
5. One non-ML person read the README cold and could say what was measured
6. The user has stopped

Artifacts: `scores.csv`, `fig1_reliability.png`, `fig2_stability.png`, `fig3_comparison.png`, `README.md`.

## 4. The build — 16 steps, IDs the user may cite

Time budgets in brackets. Stage dates are the planned windows.

### Stage 1 — Get probabilities out of a model (Wed 16 – Fri 18 Sep, 8 h)
The only genuinely unfamiliar part; everything after is statistics on a CSV.
- **a1** [2h] Load the model in 4-bit, run one forward pass. Confirm it fits in VRAM and the logits tensor prints.
- **a2** [1h] Load 500–1000 four-option questions. Filter to exactly 4 choices labelled A–D (3/5-option and 1–4-labelled rows silently corrupt results).
- **a3** [4h] Write the scoring function: question + four options → four probabilities. Budget an afternoon for tokenizer problems.
- **a4** [1h] Run the full set, write `scores.csv`. Sanity check: accuracy clearly above 25%; at chance = broken prompt format, not broken model.

### Stage 2 — The calibration statistics (Sat 19 – Tue 22 Sep, 9 h)
- **b1** [2h] Reliability diagram: 10 bins, observed accuracy vs. mean confidence, diagonal, **show bin counts**.
- **b2** [2h] ECE and Brier, then Murphy decomposition (reliability / resolution / uncertainty).
- **b3** [3h] Bootstrap interval (1000 resamples of the question set) on **every** reported number.
- **b4** [2h] Write two plain sentences saying what the calibration *is* (overconfident everywhere, or only in top bins?). If the sentence can't be written, the analysis isn't done.

### Stage 3 — Make it yours: is the number stable? (Wed 23 – Sat 26 Sep, 9 h)
- **c1** [1h] Recompute ECE at 5/10/15/20 bins vs. the bootstrap interval → struggle **S2**.
- **c2** [2h] Permute answer options (answer key moves with them), rerun; measure accuracy/calibration delta → struggle **S3**.
- **c3** [3h] Compare readouts: answer-letter token vs. whole-option-string likelihood, both curves on one axis → struggle **S1**. Called "the most interesting figure".
- **c4** [3h] Run the uncensored variant through the *identical* pipeline (same questions, readout, bins, seed) → struggle **S4**. The comparison that is the user's alone.

### Stage 4 — Ship it (Sun 27 – Mon 28 Sep, 5 h)
- **d1** [2h] Write the README (question, method, three embedded PNGs, findings). Must render without cloning.
- **d2** [1h] Write "What I could not resolve" — one short paragraph per struggle. **This is the content of all fifteen outreach emails; write it more carefully than the results.**
- **d3** [1h] Restart-and-run-all, pin versions, push. Record GPU + quantisation settings.
- **d4** [1h] One non-ML friend reads the README cold. (Only the user can do this — remind them.)

The artifact also has a separate "outreach tracker" (campaign schedule, the fifteen labs, email drafts)
that counts these same 16 steps. It is **not** part of this project's build and is not reproduced here.

## 5. The four struggles (S1–S4) — the "what I could not resolve" content

Each is a genuine open issue, and each lab gets the one closest to its work. Arrive with a measured
thing and an unsettled question, not a tidy result.

| ID | Struggle | Comparison that produces it | Labs it's led with |
|----|----------|-----------------------------|--------------------|
| **S1** | **Where does the model's belief live?** Letter-token vs. full-option likelihood vs. length-normalised likelihood give three different calibration curves from one model; no settled answer which is "the" belief. | Readout (c3) | Veitch · Holtzman · TTIC · Ferguson · 3DL |
| **S2** | **ECE depends on binning.** 5 vs. 20 bins moves it; with a bootstrap interval the interval often covers the whole range — a biased estimator reported as a model property. | Bin sensitivity (c1) | Willett · SIGMA · Yuxin Chen |
| **S3** | **Shuffling options changes the answer.** Permuting A/B/C/D moves accuracy and calibration, so one model has several calibration curves depending on an information-free formatting choice. | Option order (c2) | CHAI · Globus · SAND |
| **S4** | **Two changes confounded.** Uncensored fine-tune and 4-bit quantisation both move calibration; separating them needs a 2×2 {stock, uncensored} × {fp16, 4-bit} design; fp16 8B won't fit in 12 GB. State the confound, name the design, say compute ran out. | Model variant (c4) | Secure Learning · LMCache · Ce Zhang · Tian Li |

## 6. Protocol — specifics (sections numbered as in the artifact's "The protocol" tab)

The artifact's code is "a correct sketch, not a tested package". Expect 2–3 fixes from `transformers` /
`bitsandbytes` version drift. **When something breaks, log what broke** (kept in `notes/breakage-log.md` —
my addition, not in the artifact) — the user says that list is interesting to a systems lab.

### §0 Environment
- Python 3.11, CUDA 12.x, RTX 4070 (12 GB). Existing `.venv/` is in the repo folder (gitignored).
- `pip install torch --index-url https://download.pytorch.org/whl/cu124`, then
  `transformers accelerate bitsandbytes datasets numpy pandas matplotlib`.
- Verify: `torch.cuda.get_device_name(0)`, `torch.cuda.is_available()`; check bitsandbytes built against the right CUDA.
- **Use `transformers`, not Ollama/llama.cpp** — the project needs raw full-vocab logits at a chosen position.
- **Windows note (my adaptation):** the artifact's commands are bash (`source .venv/bin/activate`,
  `pip freeze | grep`). This machine is Windows 11 — activate with `.venv\Scripts\Activate.ps1` (PowerShell) and
  use `Select-String` or a Python one-liner for the requirements filter. `bitsandbytes` on Windows needs a
  recent version (≥0.43) with Windows wheels; if it fails, that goes in the breakage log.
- Pin at ship time to `requirements.txt`: torch, transformers, accelerate, bitsandbytes, datasets, numpy, pandas, matplotlib.

### §1 Load the model (4-bit)
`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
bnb_4bit_use_double_quant=True)`; `AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
device_map="cuda:0")`; `.eval()`. `MODEL_ID = "Qwen/Qwen3-8B"`. Expect ~5–6 GB at nf4; if it won't fit,
drop to a 4B model (nothing downstream depends on size). Print `model.get_memory_footprint()`.

### §2 Dataset
`load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")`. `usable(row)`: exactly 4 choices, labels
== `["A","B","C","D"]`, `answerKey` in A–D. Take first 800 usable; **print the count**. MMLU
(`cais/mmlu`, a few subjects) is the acceptable alternative — pick one, never both.

### §3 Readout A — answer-letter token (the core of the project)
- `letter_ids = [tok.encode(" " + L, add_special_tokens=False)[-1] for L in "ABCD"]` — **leading space matters**;
  `assert len(set(letter_ids)) == 4`; print `tok.convert_ids_to_tokens(letter_ids)` and eyeball it.
- `build_prompt`: user message = "Answer the multiple-choice question with a single letter.\n\nQuestion: …\nA. …\nB. …\nC. …\nD. …",
  passed through `tok.apply_chat_template(..., tokenize=False, add_generation_prompt=True, enable_thinking=False)`
  then `+ "Answer:"`.
- `readout_letter`: one forward pass under `torch.no_grad()`, take `logits[0, -1]`, index the four letter ids,
  `.float()`, softmax → 4 probabilities.
- **Failure modes, in order of frequency:** (1) skipped the chat template → accuracy ≈ 25%;
  (2) Qwen3 emits a `<think>` block (old transformers lacks `enable_thinking`) → one letter gets ~all the mass on every
  question; fix by upgrading transformers or appending the template's empty-think marker manually;
  (3) missing leading-space tokens → arbitrary-looking probabilities.
- **Before running 800 questions, run 10**: print the prompt and the four probabilities and check by hand.
- **Modelling choice to disclose (one README sentence, part of S1):** the 4-way softmax renormalises away any
  mass the model put on other tokens (e.g. "I don't know").

### §4 Readouts B and C — sequence likelihood
`continuation_logprob(prompt, continuation)` = sum of log P over continuation tokens only (slice the per-token
log-probs at `prompt_len - 1`). `readout_sequence(question, options, normalise)`: score `" " + option_text` for each
option, softmax over the four scores. **B** = `normalise=False` (raw sum; biased toward short options).
**C** = `normalise=True` (mean per token; over-corrects). Neither is "right" — that is the point.

### §5 Data schema — `data/scores.csv` (write once, never restructure)
One row per (question, readout, model, perm):

| Column | Type | Meaning |
|---|---|---|
| `qid` | str | Benchmark item id |
| `model` | str | `stock` or `uncensored` |
| `readout` | str | `letter`, `seq`, `seq_norm` |
| `perm` | int | 0 = original order, 1–3 = cyclic shifts |
| `p_a … p_d` | float | Full four-way distribution (keep it) |
| `pred` | str | argmax letter |
| `conf` | float | max probability — what calibration is about |
| `truth` | str | Correct letter **after** permutation |
| `correct` | int | 1 if `pred == truth` |

**Permutation bug everyone writes:** shuffle options and the answer key together. One function returns
`(permuted_options, new_correct_letter)`; never permute in two places, or accuracy collapses to chance.

### §6 Statistics
- **ECE** = Σ_b (n_b/N)·|acc(b) − conf(b)|. Support `n_bins` and `scheme` ∈ {`width` (equal-width over [0,1], default),
  `mass` (equal-mass via quantiles)}. Bin index via `np.digitize` on inner edges, clipped to `[0, n_bins-1]`; skip empty bins.
- **Brier** BS = (1/N) Σ (p_i − y_i)², with p = top-choice confidence, y = 1 if top choice correct.
  **Murphy:** BS = reliability − resolution + uncertainty, where
  reliability = Σ_b (n_b/N)(conf(b) − acc(b))² (lower better);
  resolution = Σ_b (n_b/N)(acc(b) − ȳ)² (higher better);
  uncertainty = ȳ(1 − ȳ) (property of the data, not the model).
  Purpose: separates "miscalibrated" from "merely uninformative" — different failures, different fixes.
- **Bootstrap:** `bootstrap_ci(conf, correct, stat, n_boot=1000, alpha=0.05, seed=0)`; resample with
  `rng.integers(0, n, n)`; percentile interval at [α/2, 1−α/2]; return `(point, lo, hi)`.
  **Resample *questions*, not rows** — a question's rows across readouts/permutations are not independent;
  carry all of a question's rows together or intervals will be too narrow.
- Use **seed 0** everywhere unless there's a reason not to; keep the same seed across stock/uncensored.

### §7 The comparisons

| Comparison | Struggle | Procedure | Report |
|---|---|---|---|
| Bin sensitivity | S2 | Same scores; `ece(..., n_bins=k)` for k ∈ {5,10,15,20}, both binning schemes | Eight numbers with intervals, as a small figure; does spread exceed the interval? |
| Option order | S3 | Original + three cyclic shifts, key moved with options; full rerun each | Accuracy and ECE per permutation + range across the four |
| Readout | S1 | Same questions/model, three readout functions | Three reliability curves on one axis; where do they diverge most? |
| Model variant | S4 | stock vs. uncensored, everything else identical (seed, 4-bit config) | Two reliability curves + both ECEs with intervals |

**The confound (S4), stated properly for the README:** stock@4-bit vs. uncensored@4-bit — both differ from
unquantised stock, so a difference could be the fine-tune, its interaction with quantisation, or noise. Clean
design is 2×2 {stock, uncensored} × {fp16, 4-bit}; fp16 8B doesn't fit in 12 GB. State it accurately, name the
design, say compute ran out — "the most credible sentence in the README".

### §8 Figures
- **fig1_reliability.png** — confidence (x) vs. observed accuracy (y), dashed diagonal, 10 bins, bin counts as
  a thin histogram or marker size ∝ n (a 9-point bin must not look as authoritative as a 200-point one).
- **fig2_stability.png** — bin count (x) vs. ECE (y), one line per binning scheme, shaded bootstrap band.
- **fig3_comparison.png** — 2–3 reliability curves on one axis: the three readouts *or* stock vs. uncensored,
  whichever shows the larger effect (the other goes in the README as a secondary plot).
- Axis labels with units; **one colour per series, consistent across all three figures**; `dpi=200`; no rainbow colormap.

### §9 Repository layout (at the repo root of this repo)
```
README.md               # the actual deliverable
requirements.txt        # pinned
src/
  score.py              # model loading + the three readouts
  calibrate.py          # ece, brier, murphy, bootstrap
  figures.py
data/
  scores.csv            # commit if < ~50 MB; it is the evidence
figures/
  fig1_reliability.png
  fig2_stability.png
  fig3_comparison.png
notebooks/
  walkthrough.ipynb     # restart-and-run-all before every push
```
Existing files: `experiment.ipynb` is a hello-world placeholder (it can be replaced by / moved to
`notebooks/walkthrough.ipynb`); `.venv/` and `.gitignore` already exist.

### §10 README skeleton
`# Does a small model know what it doesn't know?` → **One line** (calibration study of `<model>` on `<n>` MC
questions, 4-bit, single RTX 4070) → **What I measured** (two sentences + fig1) → **Method** (model, quantisation,
dataset, filtering, readout, binning; short) → **Findings** (three bullets *with numbers and intervals*, not prose)
→ **What I could not resolve** (S1/S2/S3/S4, one paragraph each — write this best) → **Reproducing** (GPU, versions, one command).

## 7. Risks and pre-agreed responses

| Risk | Likelihood | Response |
|---|---|---|
| Tokenizer indexing eats two days | High | Budgeted in a3; verify on 10 questions by hand before 800 |
| Accuracy near chance | Medium | Print one complete prompt and read it as a human — almost always the chat template |
| Uncensored variant won't load / behaves oddly | Medium | **Drop c4.** Project still works with the other comparisons; lose one email angle |
| Out of VRAM | Low | Drop to a 4B model |
| Wanting to keep improving past 28 Sep | Very high | Don't. Ship on the 28th |

## 8. How I should work on this project

- **Referencing:** step IDs `a1`–`d4`, struggle IDs `S1`–`S4`, protocol sections `§0`–`§10`, and "readout A/B/C"
  mean exactly what is defined above. Map any of them to the right piece of work without asking.
- **Order:** follow the stages in sequence. Don't start stage 3 comparisons before `scores.csv` exists and passes the
  >25%-accuracy sanity check.
- **Validate before scaling:** always run ~10 questions and eyeball prompt + probabilities before a full run.
- **Never hardcode a result** into the README or figures; everything derives from `scores.csv` via `src/`.
- **Report numbers honestly**: if accuracy is at chance, a bootstrap interval is wide, or the S2 spread *doesn't*
  exceed the interval, say so — the README should reflect what was actually found, not the story the spec anticipates.
- **Hold the scope line (§2).** If asked for something out of scope, say it's out of scope and offer to note it in
  "what I didn't do" instead. After 28 Sep, don't improve analysis.
- **Don't commit or push** unless the user asks. Don't commit model weights or HF caches; `data/scores.csv` is
  intended to be committed.
- **Things only the user can do:** get the non-ML read (d4), and anything on the outreach side.
- **Known open items in the spec:** "three comparisons" (scope) vs. four (protocol table);
  spec dates (16–28 Sep 2026) may already be partly elapsed — ask what stage the user is actually at rather than assuming.
