# Handoff: shrinking the S2S model (256 → 128 / 64)

Goal: shrink the C2K (then P2K) model from width 256 to 128 or 64 for faster
inference, with an iterative "cut a little, distill to convergence, repeat"
pipeline adapted from Kyutai's Pocket TTS small models
(<https://kyutai.org/blog/2026-10-20-pocket-tts-small/>). Delete this file once
the work is done.

## Already done

- `train.py`: `Model(p2k, dim=DIM)` now takes the width; `MyDataset` builds
  `G2p()` only for P2K. Verified: `Model(dim=256)` has exactly the parameter
  names and shapes of the shipped `src/e2k/models/model-c2k.npz`, loading it
  reproduces `geogaddi -> ジオガディ`.
- Parameter counts: 256: 2.22M, 192: 1.26M, 128: 571k, 64: 150k.
- Inference was already sped up (float32 throughout, K/V cache) in commit
  `d49a113`; numpy C2K is ~3.2 ms/word at 256.

Nothing of the pipeline itself is written yet.

## What to take from the blog, and what not

The blog's recipe for width (its "Shrinking gradually" section):

1. A sequence of models of decreasing size, each **cut from the previous one**
   and **retrained by distillation from the teacher to convergence** before
   the next cut (Minitron + Zhu & Gupta).
2. Width cuts rotate the residual stream onto its principal components and
   keep the largest (SliceGPT); heads/FFN units are kept by largest weights.

Its ablations are the important part for us:

| init of the final small model | UTMOS |
|---|---|
| gradual cuts, retrain each step (recipe) | 4.357 |
| one single big cut | 3.703 |
| teacher weights cut straight to size | 3.678 |
| random init | 3.683 |
| channel *selection* instead of PCA rotation | 4.33 (stalls, then ends level) |

So: the gradual schedule matters a lot, the PCA rotation barely. Below some
size the steps have to get smaller.

Not applicable to e2k: guidance-in-target distillation, the narrow
backbone→head interface, decoder distillation, reward fine-tuning, embedding
factorisation (vocabularies are 32–87 rows).

### Why PCA rotation only applies to some spaces here

SliceGPT needs a space that is only read and written through linear maps.
GRU hidden states go through element-wise gates (`(1-z)*n + z*h`,
`r * (W_hh h)`) and `enc_out` through `tanh`, so rotating them changes the
function: there, keep whole units instead. The spaces of this model:

| space | produced by | read by | cut |
|---|---|---|---|
| E: `e_emb` | `e_emb.weight` cols | `encoder.weight_ih_l0{,_reverse}` cols | PCA rotation |
| Hf / Hb: encoder hidden fwd / bwd | encoder GRU gate rows | own `weight_hh` cols, `encoder_fc.0.weight` cols `[:D]` / `[D:]` | unit selection |
| C: `enc_out` (after tanh) | `encoder_fc.0` rows | attn K and V `in_proj` cols | unit selection |
| K: `k_emb` | `k_emb.weight` cols | `pre_decoder.weight_ih_l0` cols | PCA rotation |
| P: `dec_out` | pre_decoder gate rows | own `weight_hh` cols, attn Q `in_proj` cols, `post_decoder.weight_ih_l0` cols `[:D]` | unit selection |
| QK: per-head query/key dims | Q, K `in_proj` rows | each other (dot product) | per-head selection |
| V: per-head value dims | V `in_proj` rows | `attn.out_proj.weight` cols | per-head PCA rotation |
| A: `attn_out` | `attn.out_proj` rows + bias | `post_decoder.weight_ih_l0` cols `[D:]` | PCA rotation |
| H2: post_decoder hidden | post_decoder gate rows | own `weight_hh` cols, `fc.weight` cols | unit selection |

Keep one width `d` for every space (and `d % 4 == 0`). The packed
`nn.MultiheadAttention` weights need P, C, A, QK and V all equal, and both the
numpy (`src/e2k/inference.py`) and Rust (`e2k-rs`) loaders infer every size
from the weight shapes with 4 heads hard-coded, so a uniformly shrunk npz
loads with **no inference code change**.

## The cut, concretely

Give each space a basis `B` of shape `[D, d]` with orthonormal columns: PCA
eigenvectors, or `eye(D)[:, idx]` for a selection. Then every weight that
**writes** the space is left-multiplied by `B.T` (biases too), and every
weight that **reads** it is right-multiplied by `B`. For GRUs, apply the
hidden basis to all three gate blocks (PyTorch gate order `r, z, n`, rows
`[0:D] [D:2D] [2D:3D]`), i.e. with `G(B) = block_diag(B, B, B)`:

```
W_ih' = G(B_h).T @ W_ih @ B_in        b' = G(B_h).T @ b
W_hh' = G(B_h).T @ W_hh @ B_h
encoder_fc.W' = B_C.T @ W @ block_diag(B_Hf, B_Hb)
post_decoder.W_ih' = G(B_H2).T @ W_ih @ block_diag(B_P, B_A)
Wq' = s * B_QK.T @ Wq @ B_P,  Wk' = B_QK.T @ Wk @ B_C,  Wv' = B_V.T @ Wv @ B_C
out_proj.W' = B_A.T @ W_o @ B_V,  fc.W' = fc.W @ B_H2,  e_emb' = e_emb @ B_E
```

- `s = sqrt(dh_new / dh_old)`: PyTorch scales logits by `1/sqrt(head_dim)`,
  so rescale the query to keep the attention logits unchanged.
- QK and V bases are block diagonal per head (keep `d/4` dims in each of the
  4 heads, so head boundaries stay aligned).
- PCA: eigenvectors of the **uncentered** second moment `E[x xᵀ]` over
  non-padding positions (biases are carried through, so no centering).
- Unit score for a selection: activation energy `E[x_j²]` times the squared
  norm of the weight columns reading unit `j`. For QK, `E[q_j²]·E[k_j²]`.
- Gather the statistics with a teacher-forced forward pass in eval mode on a
  few hundred training batches. Mask padding: source-side tensors with
  `src_mask`, decoder-side ones with `tgt_mask[:, :-1]`.
- Compute in float64, then cast back.

**Sanity check, run it before every pipeline run:** with rotation enabled
and `d = D`, every basis is orthogonal and every selection is the identity,
so the "cut" model must reproduce the teacher's logits (to ~1e-5). That one
check catches nearly every indexing or transpose mistake.

Also keep a `--no-rotate` switch (selection everywhere) to reproduce the
blog's selection-vs-PCA ablation.

## Distillation and "converged"

- Loss on teacher-forced logits over non-padding targets:
  `0.5 * KL(teacher || student) + 0.5 * CE(data)`. The KD term matters more
  than usual here: `MyDataset` picks one of several valid katakana spellings,
  so hard targets are noisy, and the teacher's distribution covers all of them.
- Always distill from the **original 256 teacher**, not the previous stage
  (as in the blog).
- Adam, lr ~5e-4 after a cut, `ReduceLROnPlateau(factor=0.5, patience=1)` on
  validation loss, early stop after ~3 epochs without improvement, cap at
  ~40 epochs, and restore the best state. Log loss per epoch to confirm the
  plateau is real.
- Schedule: 256 → 192 → 128 → 96 → 64 (then 48 / 32 if you want the Pareto
  curve). If a step loses much more than its neighbours, split it.
- Save each stage (`.pth` + exported `.npz`) and make the script resumable,
  skipping stages whose checkpoint exists.

## Data, teacher and evaluation protocol

- Data: `katakana_dict.jsonl` from the **0.2.0** GitHub release (31,338
  entries); `extract.py` sources are kaikki.org and edrdg.org. It may not be
  the exact data the shipped model was trained on.
- **Train a fresh 256 teacher** with the `train.py` recipe (Adam 1e-3,
  `ExponentialLR(0.8)`, 10 epochs, batch 64) on a fixed, seeded
  train/val/test split (e.g. 90/5/5 with its own `torch.Generator`). The shipped
  model's split can't be reproduced, so most test words would be in its
  training set, and a student distilled from it would inherit that leak.
- Metrics on the test split, greedy decoding: exact match against **any**
  reference spelling, plus character error rate against the closest one. Read
  the references from `dataset.data[idx]["kata"]`, not via `set_return_full`
  (see gotchas).
- Speed: export each stage to npz and time `e2k.inference.S2S` on the test
  words (single thread).

### Baselines to report alongside the chain

At 128 and 64, otherwise the result can't be judged:

1. from scratch, CE only (`train.py` recipe at that width);
2. from scratch, same distillation loop;
3. a single cut 256 → 64, then distillation (the blog's "one go").

The model is tiny and trains fast, so if (1) or (2) is as good as the
gradual chain, the simpler recipe wins.

Expected results table: width, params, exact match, CER, ms/word, plus the
teacher row.

## Gotchas found in the training code

- `export.py` line 19 has a stray `r` after
  `parser.add_argument("--fp32", action="store_true")`: a syntax error, the
  script doesn't run. It also does `from accent import AccentPredictor`, so
  it only works from inside `e2k_train/`.
- `eval.py` makes its own `random_split([0.1, 0.9])`, unrelated to the 95/5
  split in `train.py`, so most of its "test" words were trained on.
- `MyDataset.__getitem__` caches the first sampled spelling per index, so the
  random choice is frozen after the first access; and once an index is
  cached, `set_return_full(True)` still returns the cached single tensor.
- `Model.inference` is unbatched and stops after 16 tokens; it runs the
  attention without a mask, which is fine unbatched.
- `nn.MultiheadAttention(dropout=0.1)`: collect statistics and compare logits
  in `eval()` mode.
- Importing `g2p_en` downloads NLTK data (`averaged_perceptron_tagger`,
  `cmudict`) at import time. `train.py` still imports it at module level, so
  have that data available even for C2K.

## Suggested shape

A new `e2k_train/shrink.py` with: `activations(model, batch)` (teacher-forced
forward returning the intermediates above), `moments(model, batches)`,
`shrink(model, d, moments, rotate=True) -> Model`, the equivalence check,
`distill(student, teacher, ...)`, `evaluate(model, test_idx)`, `export_npz`,
and a CLI: `--data --p2k --schedule 192,128,96,64 --out vendor/shrink
--no-rotate --baselines`. Write per-stage metrics to
`vendor/shrink/results.json`.

Do C2K first; P2K is the same with `--p2k` once g2p_en's NLTK data is in
place.
