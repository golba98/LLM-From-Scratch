# Stage 1 programming-request specialist

This pipeline encodes text with frozen Google EmbeddingGemma 2 and trains small
classification heads. It is independent of Codexa's decoder-only `LanguageModel`,
tokenizer, base training, conversational SFT, and model exports.

**No suitable human-labeled programming-intent dataset is currently available.**
The checked-in implementation provides infrastructure and mechanical tests.
It does not include a trained intent classifier, task accuracy, or fabricated
labeled programming examples. Synthetic tensors and opaque schema fixtures in
tests verify software behavior only.

## Environment

Use the separate `.venv-specialist` environment. The existing `.venv`, base
requirements, and generative runtime are untouched. On a clean checkout:

```bash
uv venv --python .venv/bin/python .venv-specialist
uv pip install --python .venv-specialist/bin/python -r requirements-specialist.lock
.venv-specialist/bin/python scripts/inference/specialist.py --help
.venv-specialist/bin/python -m pytest -q tests/inference/test_specialist.py
```

The lock targets Linux x86-64 and Python 3.14.7. Direct dependencies are listed in
`requirements-specialist.txt`: PyTorch 2.13.0, Transformers 5.19.0, Sentence
Transformers 6.1.0, Hugging Face Hub 1.33.0, tokenizers 0.23.2, NumPy 2.5.1,
PyYAML 6.0.3, safetensors 0.8.0, scikit-learn 1.9.1, SciPy 1.18.1, and pytest 9.1.1.
Pillow 12.3.0 and TorchVision 0.28.0 are required by the upstream multimodal
processor loader even for text-only inputs; no vision/audio neural tower is loaded.
Sentence Transformers brings the metric/splitting dependencies transitively.
Transformers 4.57.6 in the existing environment lacks EmbeddingGemma 2 support.

Installing the specialist environment requires substantial space for the CUDA
PyTorch dependencies. Initial encoder loading downloads the model into the
normal Hugging Face cache; subsequent runs can use those cached files. No
repository checkpoint or generative tokenizer is used.

## Architecture and numerical behavior

The encoder is `google/embeddinggemma-2`, pinned to commit
`914f7f89142e33e77833254d9c9b90c3cef7303b`. Sentence Transformers loads it with
`vision_config=None` and `audio_config=None`. Initialization verifies that those
modality configurations and towers are absent and every parameter is frozen.
It preserves the published mean-pooling/projection/normalization modules.

Every text is encoded with `prompt_name="Classification"`, using exactly
`task: classification | query: ` as the task prefix. Output is a finite,
L2-normalized FP32 tensor of shape `[batch, 768]`. BF16 is the default encoder
precision on native-BF16 CUDA devices; CPU uses FP32. FP16 is rejected.
These settings follow the [Google model card](https://huggingface.co/google/embeddinggemma-2).

The default custom head is:

```text
normalized embedding [768]
    → Linear(768, 256)
    → GELU
    → Dropout(0.1)
    → Linear(256, number_of_labels)
    → logits
```

The baseline is `Linear(768, number_of_labels)`. Both heads run in FP32 and
use the same embeddings and split assignments. Configuration lives in
`configs/specialist.yaml`; hidden size, dropout, labels, class weights, label
smoothing, optimizer settings, batch sizes, epochs, precision, device, and
seed are validated. Cross-entropy is the supported Stage 1 loss.

Inputs are stripped at their outer edges, preserving internal formatting.
The default limit is 2,048 tokens including the classification prefix and
tokenizer special tokens. Overlong text is rejected instead of silently
truncated. The configurable limit cannot exceed the published 8,192-token
budget. Long requests need a deliberate preprocessing policy in Stage 2.

## Annotation policy and schema

Annotate the main requested outcome, rather than keywords or the programming
language. The initial label order is fixed in the default configuration:

| Label | Main requested outcome |
| --- | --- |
| Debugging | Diagnose or correct faulty behavior |
| Refactoring | Restructure existing code while preserving intended behavior |
| Testing | Create or improve tests or verification |
| Implementation | Add new functionality |
| Explanation | Understand code, behavior, or a concept |

Supporting actions do not add labels. A request to correct a defect can also
ask for a supporting explanation; annotate the dominant requested outcome.
If neither outcome dominates, mark the record `needs_review`. Review does not
mean choosing a category by fixed priority. Non-programming or clearly unrelated
requests are `out_of_domain`, held outside the five-class supervised task.

The dataset is UTF-8 JSONL with one object per line. No blank lines are accepted.

| Field | Requirement |
| --- | --- |
| `id` | Unique nonempty string |
| `text` | Nonempty request text |
| `source` | Nonempty provenance identifier |
| `label` | One configured label for supervised records; absent/null for review/OOD |
| `group_id` | Optional globally unique conversation/task-family identifier |
| `status` | `labeled` (default), `needs_review`, or `out_of_domain` |

Unknown fields, duplicate IDs, malformed JSON, invalid labels, and conflicting
duplicate annotations fail with file/line context. Exact duplicate stripped
texts with matching annotation and group are deduplicated; the lexicographically
smallest ID is retained. Review/OOD groups cannot overlap supervised groups.
Use `group_id` to keep paraphrases, related tasks, or conversation turns out of
different partitions. Near-duplicate detection is not automatic in Stage 1.

Preparation uses seeded stratified group folds to create approximately
80% train, 10% validation, and 10% test partitions. It requires at least ten
independent groups per class and coverage of every class in all three
partitions. This is a technical minimum, not a claim of sufficient data quality.
Larger or mixed-label groups can require more data and produce uneven ratios.
Records are sorted by ID before splitting. Challenge records are kept separate.

Provenance metadata does not replace permission to use a data source. Existing
conversational datasets are not automatically relabeled or treated as gold
intent annotations. Collect representative reviewed requests before training.

## Commands

Run each command's `--help` before starting its pipeline. From the repository root:

```bash
# Real encoder only; no classification training or ground-truth labels.
.venv-specialist/bin/python scripts/inference/specialist.py smoke-encoder \
  --config configs/specialist.yaml

# Requires the real labeled JSONL file supplied through annotation.
.venv-specialist/bin/python scripts/inference/specialist.py prepare \
  --config configs/specialist.yaml \
  --dataset data/raw/specialist/programming_requests.jsonl \
  --output-dir data/processed/specialist/programming-v1

# Both heads, with an automatic visible Kitty viewer for each head.
.venv-specialist/bin/python scripts/inference/specialist.py train \
  --config configs/specialist.yaml \
  --prepared-dir data/processed/specialist/programming-v1 \
  --run-name programming-v1 --model both

# Validation-selected checkpoints evaluated against the same held-out test set.
.venv-specialist/bin/python scripts/inference/specialist.py evaluate \
  --prepared-dir data/processed/specialist/programming-v1 \
  --run-name programming-v1

# Requires the head checkpoint produced by the labeled-data training command.
.venv-specialist/bin/python scripts/inference/specialist.py infer \
  --checkpoint checkpoints/specialist/programming-v1/mlp/best.pt \
  --text 'Explain what this function does.'

# Optional additional reviewed ambiguity/OOD records, with no invented gold label.
.venv-specialist/bin/python scripts/inference/specialist.py challenge \
  --checkpoint checkpoints/specialist/programming-v1/mlp/best.pt \
  --dataset data/raw/specialist/challenge.jsonl
```

Prepared bundles contain records, split assignments, encoder identity,
dataset/content checksums, FP32 embeddings, and prefixed token counts. Embedding
cache identity includes input contents, model revision, prefix, token limit,
precision, device type, and relevant installed versions. Incompatible or
corrupt caches are rejected; reusable matching caches avoid repeated encoding.
An existing prepared output directory is never overwritten.

Run names reserve separate directories below `logs/specialist/` and
`checkpoints/specialist/`. Existing run names are rejected. Checkpoints are
written atomically, checksummed, and loaded with `weights_only=True`. They
contain head weights, class order/configuration, optimizer and PyTorch/shuffle
RNG state, stopping state, history, and dataset/encoder identities. The
`latest.pt` and `best.pt` files belong only to the newly allocated run. Existing
generative artifacts are never written by this pipeline. No cleanup is implemented.

Defaults are AdamW with learning rate 0.001 and weight decay 0.01, batch size
64, gradient clipping at 1.0, and at most 50 epochs. Early stopping uses validation
loss with patience five and minimum improvement 0.0001. The true lowest-loss
checkpoint is saved even if an improvement is smaller than the patience threshold.
Test performance never selects or promotes a checkpoint.

Training refuses to start without a graphical session and visible 100x22 Kitty
viewer. It waits for viewer attachment and stops if that viewer closes during
training. The straight progress bar, percentage, measured-step ETA, step count,
token throughput, training/validation losses, cumulative token count, and
checkpoint path are shown automatically. During head training, token counts
represent cached prefixed requests presented to the head, repeatedly across
epochs; the encoder is neither running nor updating. ETA excludes validation,
checkpoint I/O, and early-stopping uncertainty.

## Reports, acceptance, and limitations

Evaluation reports contain accuracy, macro-F1, per-class precision/recall/F1
and support, confusion matrices (actual rows, predicted columns), train/validation
loss histories, test loss, misclassified IDs/scores, challenge predictions,
process-local peak CUDA memory, and synchronized warmed head latency. Each
evaluation writes a new report instead of replacing an earlier one. Reports
compare the MLP with the linear baseline without assuming the MLP is better.

Cached-head evaluation does not claim encoder or end-to-end latency. The real
encoder smoke report separately measures encoder loading, warmed batch-size-one
latency, and encoder CUDA memory. Text inference measures its actual batch
encoder/head/end-to-end duration, explicitly including host transfers and not
claiming a warmed benchmark. CUDA memory figures concern this PyTorch process,
not GPU-wide consumption by unrelated applications.

Inference returns a predicted label and uncalibrated softmax scores. A forced
five-class output does not establish confidence, resolve ambiguity, or detect
OOD requests reliably. CPU FP32 inference is supported when the training cache
used GPU BF16; its runtime metadata is returned and cross-device numerical
equivalence is not promised. Deterministic execution is requested, but exact
reproduction is limited to matching hardware, dependencies, and input processing.

Mechanical tests inject a tiny encoder and use temporary synthetic bundles;
they do not download models, depend on ignored datasets, or report test-fixture
metrics as real task results. A real encoder smoke must separately succeed.
The original environment's verified baseline is 58 passed / 3 failed, involving
stale 921M parameter assumptions and missing position-embedding fixture fields.
Those failures remain independent of specialist acceptance.

Stage 1 has no resume CLI, annotation UI, calibrated rejection, near-duplicate
detector, automatic chunking, or trained-head export. Checkpoints and RNG state
are retained for subsequent recovery tooling. If a run fails, preserve its
artifacts and use a fresh run name for any replacement; do not delete checkpoints
until the repository's export-and-validation storage policy has been satisfied.

## Implementation inventory and verified Stage 1 acceptance

Created:

- `src/specialist/`: `__init__.py`, `config.py`, `artifacts.py`, `encoder.py`,
  `models.py`, `data.py`, `training.py`, `evaluation.py`, and `viewer.py`.
- `scripts/inference/specialist.py`, `configs/specialist.yaml`, `requirements-specialist.txt`,
  `requirements-specialist.lock`, `tests/test_specialist.py`, and this document.

Modified: `.gitignore` to ignore the isolated environment, plus appended dated
entries in `documentation/training/SESSION_DECISIONS.md` and
`documentation/training/100M_PROGRESS_LOG.md`. Existing unrelated modifications
and untracked files were preserved. Nothing was committed or pushed.

Verification on 2026-10-08:

- All 39 specialist tests passed inside `.venv-specialist`.
- All 65 installed specialist packages passed dependency checks and exactly
  matched the lock. Matching native PyTorch/CUDA/Triton packages were copied into
  the separate environment to avoid redundant network downloads. Representative
  native binary hashes matched, with separate file inodes; no original package
  directory is shared or changed.
- The original environment's complete suite reported 92 passed, 5 specialist
  dependency-related skips, and the same three pre-existing generative failures.
- A real visible 100x22 Kitty viewer attached during a non-training smoke check.
- The real pinned encoder loaded 271,002,624 text-only parameters on an RTX 4080
  using BF16, with zero trainable weights and normalized deterministic `[2, 768]`
  embeddings. The final five-call warmed encoder benchmark measured 15.2 ms
  median batch-size-one latency and 580,295,680 bytes (553.4 MiB) peak allocated
  CUDA memory. A real FP32 CPU smoke also passed, measuring 46.0 ms median
  encoder latency. These are actual encoder measurements, not classifier results.
- No labeled task training or generative training was launched. Task accuracy,
  macro-F1, intent confusion matrices, and baseline-versus-MLP task comparisons
  remain unavailable until reviewed intent data is supplied.

Recommended Stage 2 work:

- Collect diverse reviewed intent labels and independent ambiguity/OOD probes.
- Measure annotator agreement and challenge-set behavior before changing labels.
- Calibrate confidence and evaluate an abstention threshold using held-out data.
- Compare multi-label classification, near-duplicate grouping, and long-input policies.
- Add verified resume/export support and measure CPU versus GPU deployment costs.
- Test smaller embedding dimensions or larger heads only against the linear baseline.


## Stage 2 retrieval reuse

`FrozenEncoder.encode(texts, task="SearchQuery"|"Document")` now supports explicit retrieval prompts verified against the model registry. The default remains `Classification`; classification metadata and caches keep their original identity. Retrieval uses a separate scoped memory module and worker, documented in [CONVERSATIONAL_MEMORY.md](CONVERSATIONAL_MEMORY.md). No classification head training occurred in Stage 2.
