"""Export the trained Codexa checkpoint through LM Studio's GPT-2 GGUF path.

Codexa's tensor layout is deliberately close to GPT-2's GGUF layout: fused QKV,
learned positions, and 4x feed-forward width. The export keeps those tensors
and uses the supported GPT-2 runtime metadata so LM Studio can load the model.
"""

from __future__ import annotations

from llm_architecture.checkpointing import read_native_checkpoint

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch


def _tokenizer_files(tokenizer_path: Path, destination: Path) -> None:
    payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    model = payload["model"]
    vocab = model["vocab"]
    merges = model["merges"]
    (destination / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    merge_lines = ["#version: 0.2"]
    merge_lines.extend(" ".join(pair) for pair in merges)
    (destination / "merges.txt").write_text("\n".join(merge_lines) + "\n", encoding="utf-8")
    tokenizer_config = {
        "model_max_length": 2048,
        "tokenizer_class": "GPT2Tokenizer",
        "bos_token": "<bos>",
        "eos_token": "<|end|>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"],
        "chat_template": "{% for message in messages %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}",
    }
    (destination / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": "<|end|>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def export(checkpoint: Path, tokenizer: Path, output: Path, converter: Path) -> Path:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    payload = read_native_checkpoint(checkpoint, map_location="cpu")
    state = payload["model_state_dict"]
    raw_config = payload["config"]
    config = raw_config.get("model", raw_config)
    hidden = int(config["hidden_size"])
    inner = int(config["intermediate_size"])
    layers = int(config["num_layers"])

    mapped: dict[str, torch.Tensor] = {
        "transformer.wte.weight": state["token_embeddings.weight"],
        "transformer.wpe.weight": state["position_embeddings.weight"],
        "transformer.ln_f.weight": state["final_norm.weight"],
        "transformer.ln_f.bias": torch.zeros(hidden),
        "lm_head.weight": state["lm_head.weight"],
    }
    for index in range(layers):
        prefix = f"blocks.{index}"
        target = f"transformer.h.{index}"
        mapped[f"{target}.ln_1.weight"] = state[f"{prefix}.attention_norm.weight"]
        mapped[f"{target}.ln_1.bias"] = torch.zeros(hidden)
        mapped[f"{target}.attn.c_attn.weight"] = state[f"{prefix}.attention.qkv_projection.weight"].T.contiguous()
        mapped[f"{target}.attn.c_attn.bias"] = torch.zeros(3 * hidden)
        mapped[f"{target}.attn.c_proj.weight"] = state[f"{prefix}.attention.output_projection.weight"].T.contiguous()
        mapped[f"{target}.attn.c_proj.bias"] = torch.zeros(hidden)
        mapped[f"{target}.ln_2.weight"] = state[f"{prefix}.ffn_norm.weight"]
        mapped[f"{target}.ln_2.bias"] = torch.zeros(hidden)
        # GPT-2 has one 4x projection. Use Codexa's learned up projection;
        # the original gated path cannot be represented by GPT-2 metadata.
        mapped[f"{target}.mlp.c_fc.weight"] = state[f"{prefix}.feed_forward.up_projection.weight"].T.contiguous()
        mapped[f"{target}.mlp.c_fc.bias"] = torch.zeros(inner)
        mapped[f"{target}.mlp.c_proj.weight"] = state[f"{prefix}.feed_forward.down_projection.weight"].T.contiguous()
        mapped[f"{target}.mlp.c_proj.bias"] = torch.zeros(hidden)

    torch.save(mapped, output / "pytorch_model.bin", _use_new_zipfile_serialization=True)
    model_config = {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "n_ctx": int(config["context_length"]),
        "n_positions": int(config["context_length"]),
        "n_embd": hidden,
        "n_layer": layers,
        "n_head": int(config["num_heads"]),
        "n_inner": inner,
        "vocab_size": int(config["vocab_size"]),
        "layer_norm_epsilon": 1e-6,
        "activation_function": "gelu_new",
        "tie_word_embeddings": True,
    }
    (output / "config.json").write_text(json.dumps(model_config, indent=2) + "\n", encoding="utf-8")
    _tokenizer_files(tokenizer, output)
    artifact_name = output.name.replace("-hf-gpt2", "")
    gguf_path = output.parent / f"{artifact_name}-lmstudio-f16.gguf"
    if gguf_path.exists():
        gguf_path.unlink()
    command = [
        sys.executable, str(converter), "--outfile", str(gguf_path),
        "--outtype", "f16", "--model-name", artifact_name,
        str(output),
    ]
    subprocess.run(command, check=True)
    return gguf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.checkpoint, args.tokenizer, args.output, args.converter)
    print(result)


if __name__ == "__main__":
    main()
