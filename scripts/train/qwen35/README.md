# Qwen3.5 native function-calling SFT

This directory is the Qwen-specific counterpart to the existing OLMoCore SFT
path. It does not modify or call the Dolci converter and it does not reinterpret
historical Dolci JSON as Qwen data.

## Frozen boundary between shared and model-specific data

Shared across OLMo and Qwen:

- canonical source records and stable arm membership;
- semantic edits and assistant-message loss decisions;
- raw NumPy token/mask/boundary artifact shape;
- fixed-sequence packing policy and accounting fields.

Model-specific:

- native chat/tool rendering;
- token IDs and assistant-token masks;
- model/trainer implementation and packed-attention kernels.

The Qwen converter accepts canonical rows with `messages`, `tools`, `dataset`,
and `sample_id`. OpenAI JSON-string call arguments are parsed into dictionaries
only in an in-memory rendering copy because the native Qwen template iterates
the argument mapping. Call IDs, result messages, message boundaries, tools, and
the source row remain intact.

## 1. Convert a canonical fixture or arm

```bash
PYTHONPATH=. python scripts/data/convert_sft_data_for_qwen35.py \
  --input-jsonl /path/to/canonical.jsonl \
  --output-dir /path/to/qwen35_numpy \
  --tokenizer-name-or-path Qwen/Qwen3.5-0.8B-Base \
  --tokenizer-revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --max-seq-length 32768
```

The converter writes raw (headerless) arrays compatible with the existing
OLMo artifact convention:

```text
token_ids_part_0000.npy
labels_mask_part_0000.npy
token_ids_part_0000.csv.gz
tokenizer/
documents.jsonl.gz
statistics.json
manifest.json
```

`manifest.json` pins the tokenizer revision and chat-template hash, declares
the token dtype and length of every part, and hashes every core artifact. With
`verify_hashes=True`, the loader verifies the document-identity index directly
in addition to token, mask, and boundary files. The distributed trainer
requires this policy, performs the complete identity-bearing NumPy/document
file pass once on global rank 0, then synchronizes ranks. Every rank validates
the saved tokenizer semantically before model loading. The conversion is built
in a sibling `.incomplete` directory and renamed only after success. Existing
output is never replaced without `--overwrite`.

Qwen3.5's native template does not expose Transformers generation spans. The
converter inserts Jinja `generation` blocks into an in-memory template copy for
mask bookkeeping and asserts per record that the instrumented and untouched
native renderings have identical token IDs. Assistant headers and the native
empty `<think>...</think>` prefix are masked. Assistant answer/tool-call tokens
and `<|im_end|>` are trained. An optional assistant-loss manifest supports
message-level M08-style exclusions without rewriting messages.

## 2. Validate the NumPy reader and renderer

Run tests inside the separately pinned Qwen3.5 environment, not the
repository's ordinary released-Transformers `uv run` environment:

```bash
PYTHONPATH=. /path/to/qwen35-venv/bin/python -m pytest -q tests/test_qwen35*.py
```

The tests cover schema preservation, argument normalization, template
instrumentation, raw-memmap validation, document boundaries, position/sequence
resets, cross-document label masking, and isolated non-loss padding.

Run the frozen fixture through the real pinned tokenizer in the reviewed
Qwen3.5 environment:

```bash
PYTHONPATH=. python scripts/train/qwen35/validate_qwen35_reference.py \
  --fixture-jsonl tests/fixtures/qwen35_fc/fixture.jsonl \
  --report-output /path/to/qwen35_fixture_reference_report.json
```

Add `--model-parity` on a CUDA node to compare the official
`Qwen3_5ForConditionalGeneration` against converted text-only logits exactly on
independent sequences, then compare those text-only independent logits against
one packed sequence. This gate requires Flash Attention 2,
flash-linear-attention (`cu_seq_lens_q`), and causal-conv1d (`seq_idx`).

## 3. Build a frozen exposure schedule

Packing and scheduling are separate. The NumPy loader deterministically packs
atomic conversations while preserving every boundary. The schedule then says
which physical packs are consumed, in exactly what order, under one nominal
training seed. The schedule hashes each stable pack identity, never repeats a
real pack, and records exact fixed tokens, real tokens, effective assistant
targets, padding, document counts, and isolated-attention quadratic mass.

For the four-GPU, two-microbatch G2 proof, 80 real packs produce ten exact
global groups of eight:

```bash
PYTHONPATH=. python scripts/data/build_qwen35_pack_schedule.py \
  --numpy-data-dir /path/to/C00/qwen35_numpy \
  --output /path/to/C00/g2_seed3407_80packs.json \
  --sequence-length 32768 \
  --schedule-seed 3407 \
  --global-packs-per-update 8 \
  --real-pack-limit 80 \
  --verify-data-hashes
```

The default is fail-closed when the real-pack count is not divisible by the
global group. This avoids silently adding an unusually expensive full-32K
all-masked sequence. Synthetic final-group padding exists only as an explicit
adversarial/qualification option. Scientific schedules should instead choose a
preregistered divisible real-pack limit and, when needed, a common effective
assistant-target budget and tolerance across arms.

The nominal training seed and data seed must equal the schedule seed. Qwen3.5
0.8B has no dropout and starts from a fixed checkpoint, so the important
stochastic intervention is the hashed exposure order. The remaining seed still
controls any runtime RNG use and is checkpointed on every distributed rank.

## 4. Correct text-only model and selective loss path

The production trainer does not retain an unused vision tower. It directly
loads the official multimodal checkpoint into `Qwen3_5ForCausalLM` with the
checkpoint's pinned `text_config`, requires an empty loading-info error set,
proves tied input/output embeddings, and writes a tensor-by-tensor source-key,
shape, dtype, and SHA-256 conversion ledger.

Only Liger's fused linear cross entropy is patched. Liger RMSNorm, SwiGLU,
rotary, and ordinary cross-entropy substitutions remain disabled. For every
supervised next-token target at label position `t+1`, the collator projects only
hidden row `t`; user/system/tool/padding output rows are never materialized
through the 248,320-way output matrix. A zero-target rank receives one ignored
sentinel row so its output head remains in the DDP graph while contributing
exactly zero loss.

The precision contract is:

- FP32 trainable parameters;
- FP32 gradients;
- FP32 AdamW first and second moments;
- BF16 autocast for forward/backward compute.

Transformers counts the already shifted selected labels across every gradient
accumulation microbatch and DDP rank. Liger computes a local loss sum divided by
that global assistant-target count; Trainer multiplies by world size before
DDP's gradient average. The resulting update is the global mean per supervised
assistant target even when ranks have unequal target counts.

The generic Transformers FLOP counter is deliberately disabled. It does not
model Qwen3.5's hybrid GDN/full-attention layers, document-isolated quadratic
mass, or selective output rows.

## 5. Required qualification programs

The following gates test independent failure modes:

- `validate_qwen35_reference.py`: native fixture rendering, untouched
  `Qwen3_5ForConditionalGeneration` versus converted `Qwen3_5ForCausalLM`, and
  independent-versus-packed model logits;
- `validate_qwen35_selective_loss.py`: real CUDA Liger loss and gradient parity
  for dense, sparse, one-target, unequal-divisor, and all-ignored-sentinel
  cases, plus end-to-end dense-versus-patched loss and every-parameter-gradient
  parity through a tiny real `Qwen3_5ForCausalLM` forward;
- `validate_qwen35_ddp_loss_normalization.py`: multi-rank Liger/DDP parity with
  deliberately unequal per-rank target counts and a zero-target rank;
- `validate_qwen35_accelerate_schedule_sharding.py`: sequential distributed
  sharding with `even_batches=False`, proving every schedule index is observed
  exactly once;
- `compare_qwen35_checkpoints.py`: tensor-semantic comparison of continuous and
  stop/resume model, optimizer, scheduler, RNG, and trainer state;
- `validate_qwen35_generation_parser.py`: strict native-tool parsing and exact
  single-versus-left-padded-batch greedy generation invariance.

CPU tests independently compare selective versus dense losses and gradients,
tamper with schedule hashes and entries, validate conversion on a tiny real
Qwen3.5 checkpoint, verify FP32 optimizer states, recompute the 0.8B FLOP
formula, and adversarially mutate checkpoints and G2 artifacts.

## 6. Exact run reporting

`qwen35_exact_metrics.jsonl` is the authoritative stream. Every synchronized
window records:

- exact schedule indices and pack UIDs;
- fixed 32K tokens, real non-padding tokens, effective assistant targets,
  padding, documents, and synthetic packs;
- the exact global loss divisor and reconstructed global target-mean loss;
- synchronized wall time and explicitly labeled host-enqueue diagnostics;
- fixed-token, real-token, and assistant-target rates as distinct quantities;
- peak allocated and reserved CUDA bytes;
- versioned Qwen3.5 hybrid FLOP components and analytic model MFU under the
  labeled nominal 312-TFLOP/s A100 dense-BF16 denominator.

For causal full attention, the useful pair count is computed exactly as
`sum_i L_i(L_i+1)/2 = (sum_i L_i^2 + fixed_tokens)/2`, including isolated
padding. Using `sum_i L_i^2` would incorrectly count the masked upper triangle.
GDN recurrence FLOPs remain explicitly labeled an analytic approximation.

Host enqueue durations are never presented as device kernel durations. Detailed
device-region and roofline work still requires CUDA events and profiler/DCGM
artifacts. Short G2 jobs synchronize every step; production profiling should
use longer explicitly recorded windows so measurement barriers do not distort
steady-state overlap.

## 7. Unsubmitted Leonardo G2 sequence

The staged wrappers are:

- `leonardo_0_8b_smoke.sbatch`: one GPU, conversion/reference and selective
  Liger qualification, two updates, and checkpoint validation;
- `leonardo_g2_b_distributed.sbatch`: four GPUs, sharding and unequal-target
  DDP qualification, then stop at checkpoint 5 of the common ten-step recipe;
- `leonardo_g2_c_continuous.sbatch`: an uninterrupted ten-step reference;
- `leonardo_g2_c_resume.sbatch`: resume accepted checkpoint 5 through step 10,
  run packed/generation/parser gates, and require zero-tolerance semantic
  equality with the continuous checkpoint 10. RNG comparison keeps Torch's
  restricted weights-only loader enabled and allowlists only the exact NumPy
  MT19937 globals present in HF RNG checkpoints.

All wrappers use global fixed-token batch 262,144, learning rate `2e-5`, cosine
decay, 3% warmup, AdamW betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`,
and gradient clipping `1.0`. These are qualification/calibration settings, not
yet evidence that the recipe is scientifically optimal.

Every wrapper hard-codes personal Slurm account `aifac_f02_434`, refuses paths
outside the personal AIFAC root, verifies code/model/schedule identities, and
does not submit itself. Qualification jobs are launched only by separately
recorded submission wrappers; no wrapper automatically starts a later gate or
scientific run.

## Current validation status

The earlier 2026-07-15 Cray result remains historical evidence for native
rendering and packed-boundary kernels, not evidence for this corrected trainer.
The 128-example fixture still reproduces 563,233 rendered tokens and 147,669
effective assistant targets, with two deliberate truncations. A read-only CPU
audit of the real pinned checkpoint matched all 321 text target state tensors,
found an empty loading-error set, and proved 752,393,024 FP32 tied-embedding
text parameters.

R17 subsequently passed H0 and H1 but failed H2 under its matched
Liger-versus-dense-selected reference. An independent validator confirmed that
failure, so Liger is abandoned for this campaign; it was not threshold-rescored
and H3 did not run. R18 is the preregistered correctness-first successor. Its
local pure-Torch objective, manifest resolver, report validator, and
adversarial tests are implementation evidence only. R18 CUDA H2, multi-rank
NCCL, full-32K memory, continuous/resume, topology, and reporting gates remain
pending. No scientific training or BFCL/tau2 evaluation is authorized.

## 8. Frozen Leonardo hardware-qualification protocol

The current H2 successor contract is the SHA-256-bound replacement overlay
`qwen35_hardware_qualification_r18.json`. It resolves only over the immutable
byte-exact R17 overlay and its R16/R15 ancestry. R16 and R17 remain immutable
failed H2 protocols; R18 does not reinterpret either result. The resolved R18
contract is the source of truth for the personal Slurm account, eligible
C00-only scope, model/runtime pins, 32K/global-batch recipe, HBM headroom,
topology decision, reporting overhead, retry ceiling, and ordered H0--H9 gates.
It explicitly forbids Liger execution. A qualification run additionally binds
the overlay hash to one clean, byte-manifested Git commit. No wrapper may turn
qualification into C01--C11 training or BFCL/tau2 evaluation.

The additional gates are:

- H0: exact account/path/source/runtime/A100/ECC/MIG inventory, including one
  report per node and unique GPU UUIDs for multi-node jobs;
- H1: official conditional-model versus text-CausalLM conversion identity and
  standalone-versus-packed GPU parity under the predeclared packed-logit
  envelope;
- H2: pure-PyTorch non-reentrant checkpointed selected-row chunks at all four
  preregistered sizes `{128,256,512,1024}`. The gating reference executes the
  identical chunks and scalar-add order with ordinary autograd, so loss,
  gradients, clipping, Adam moments, cumulative displacement, complete state,
  and held-out behavior must be raw-byte exact. Ten boundary cases, the real
  `1025 x 1024 -> 248320` head geometry, three 256-step tied-output
  trajectories, the actual patched Qwen forward, graph-connected all-masked
  ranks, recomputation counts, and saved-tensor exclusion are mandatory.
  Chunked-versus-unchunked and unchunked-versus-full-ignore contrasts remain
  complete finite diagnostics but cannot rescue or fail an otherwise exact
  checkpoint transformation;
- H3: batched single-process, accumulated single-process, and four-rank
  DDP+GA full hybrid-Qwen loss/gradient/clipping/update/moment parity, with one
  entire DDP rank carrying zero targets; plus exact Accelerate sharding;
- H4: a real one-GPU 32K trajectory with allocator history, peak HBM,
  profiler trace, dense-logit exclusion, required component detection, and a
  fail-closed human-reviewed mapping for every observed GPU kernel;
- H5: a bounded four-GPU production-path trajectory with exact exposure,
  parameter updates, full checkpoint validation, and observed NCCL activity;
- H6: uninterrupted versus stop-at-5/resume-to-10 equality at zero tolerance
  for model, optimizer, scheduler, and per-rank RNG state;
- H7: correctness-preserving four-GPU/GA2 versus two-node eight-GPU/GA1 timing
  on the same 104 packs and LR sequence, plus a separate two-node NCCL trace;
- H8: fine versus coarse reporting on an otherwise identical trajectory,
  requiring bit-exact final state, identical exposure/accounting, and no more
  than the frozen incremental synchronization-overhead threshold;
- H9: an independent hash-closure audit over every input report. It selects a
  qualified topology but explicitly does not authorize scientific training.

R16/R17 wrappers are historical and must not be reused as R18 launchers. R18's
H2 producer is `validate_qwen35_chunked_loss_r18.py`; its separately callable
validator is `validate_qwen35_h2_report_r18.py`. Successor H3--H9 wrappers must
be added as new R18 programs after their preceding gate passes, rather than by
mutating historical scripts. Every output directory remains immutable and
every result writer is fail closed and atomic.
