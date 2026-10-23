<h1 align="center">LLM From Scratch</h1>

<p align="center"><strong>PyTorch Codexa v1</strong></p>

<p align="center">
  An independently developed language-model engineering project.<br>
  From tokenization and Transformer architecture to training, inference, and memory.
</p>

<p align="center">
  <a href="#architecture">Architecture</a> &nbsp;·&nbsp;
  <a href="#components">Components</a> &nbsp;·&nbsp;
  <a href="#getting-started">Getting started</a> &nbsp;·&nbsp;
  <a href="#project-status">Project status</a>
</p>

---

## Architecture

Eight Python packages cover the main parts of building and running a language
model. Each package has a defined responsibility; model formats and tokenizer
conventions connect them.

<p align="center"><code>Text → Tokenization → Data preparation → Training → Inference</code></p>

The architecture package defines the Transformer and checkpoint interfaces used
by training and inference. Conversation memory adds storage and retrieval;
specialist models use a separate embedding encoder. The integration package
provides shared evaluation utilities.

## Components

| Component | Focus |
| :--- | :--- |
| **[LLM-Architecture](LLM-Architecture)** | Decoder-only Transformer, runtime policies, and checkpoint readers |
| **[LLM-Data](LLM-Data)** | Text preparation, conversation mixtures, and binary token processing |
| **[LLM-From-Scratch](LLM-From-Scratch)** | Shared evaluation and integration utilities |
| **[LLM-Inference](LLM-Inference)** | Generation, native chat, context composition, and model export |
| **[LLM-Memory](LLM-Memory)** | Scoped SQLite conversation storage and exact cosine retrieval |
| **[LLM-Specialist](LLM-Specialist)** | Frozen embedding encoder and trainable classifier heads |
| **[LLM-Tokenizer](LLM-Tokenizer)** | Byte-level BPE and shared conversation serialization |
| **[LLM-Training](LLM-Training)** | Base and conversational training, checkpoint recovery, and monitoring |

<details>
<summary><strong>Repository layout</strong></summary>

```text
LLM-From-Scratch/
├── LLM-Architecture/
├── LLM-Data/
├── LLM-From-Scratch/
├── LLM-Inference/
├── LLM-Memory/
├── LLM-Specialist/
├── LLM-Tokenizer/
├── LLM-Training/
└── README.md
```

The repository presents the approved source as one project while preserving
separate package boundaries. Each component declares its dependencies and entry
points in `pyproject.toml`.

</details>

## Getting started

**Python 3.11 or later** is required. From the repository root, make the local
packages available on the Python path:

```bash
for component in LLM-*/src; do
  export PYTHONPATH="$PWD/$component${PYTHONPATH:+:$PYTHONPATH}"
done
```

Try a small integration utility using only the Python standard library:

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
from codexa_workspace.hashing import fingerprint

print(fingerprint({"example": 1}))
PY
```

Install a component's declared dependencies before running its model or data
tools. The project packages are supplied here as local source. Model execution
also requires compatible checkpoint and tokenizer files.

## Project status

Model architecture, tokenization, data processing, training, inference, and
conversation memory are implemented, with synthetic tests in the component
directories. Specialist systems and evaluation methods remain experimental.

**Weights, trained tokenizers, and datasets are not included.** This repository
provides source code; it does not publish benchmark claims or large-scale training
results.
