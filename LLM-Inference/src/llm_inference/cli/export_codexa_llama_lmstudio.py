"""Export Codexa weights to a loadable Llama GGUF adapter for LM Studio."""

from __future__ import annotations

from llm_architecture.checkpointing import read_native_checkpoint

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from llm_inference.cli.export_codexa_lmstudio import _tokenizer_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    payload = read_native_checkpoint(args.checkpoint, map_location="cpu")
    state = payload["model_state_dict"]
    raw_config = payload["config"]
    config = raw_config.get("model", raw_config)
    hidden = int(config["hidden_size"])
    heads = int(config["num_heads"])
    inner = int(config["intermediate_size"])
    layers = int(config["num_layers"])
    mapped = {
        "model.embed_tokens.weight": state["token_embeddings.weight"],
        "model.norm.weight": state["final_norm.weight"],
        "lm_head.weight": state["lm_head.weight"],
    }
    for index in range(layers):
        src = f"blocks.{index}"
        dst = f"model.layers.{index}"
        q, k, v = state[f"{src}.attention.qkv_projection.weight"].chunk(3, dim=0)
        mapped[f"{dst}.input_layernorm.weight"] = state[f"{src}.attention_norm.weight"]
        mapped[f"{dst}.self_attn.q_proj.weight"] = q
        mapped[f"{dst}.self_attn.k_proj.weight"] = k
        mapped[f"{dst}.self_attn.v_proj.weight"] = v
        mapped[f"{dst}.self_attn.o_proj.weight"] = state[f"{src}.attention.output_projection.weight"]
        mapped[f"{dst}.post_attention_layernorm.weight"] = state[f"{src}.ffn_norm.weight"]
        mapped[f"{dst}.mlp.gate_proj.weight"] = state[f"{src}.feed_forward.gate_projection.weight"]
        mapped[f"{dst}.mlp.up_proj.weight"] = state[f"{src}.feed_forward.up_projection.weight"]
        mapped[f"{dst}.mlp.down_proj.weight"] = state[f"{src}.feed_forward.down_projection.weight"]
    torch.save(mapped, args.output / "pytorch_model.bin", _use_new_zipfile_serialization=True)
    model_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": int(config["vocab_size"]),
        "hidden_size": hidden,
        "intermediate_size": inner,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": heads,
        "max_position_embeddings": int(config["context_length"]),
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
    }
    (args.output / "config.json").write_text(json.dumps(model_config, indent=2) + "\n", encoding="utf-8")
    _tokenizer_files(args.tokenizer, args.output)
    artifact_name = args.output.name.replace("-hf-llama", "")
    gguf_path = args.output.parent / f"{artifact_name}-lmstudio-f16.gguf"
    if gguf_path.exists():
        gguf_path.unlink()
    subprocess.run([
        sys.executable, str(args.converter), "--outfile", str(gguf_path),
        "--outtype", "f16", "--model-name", artifact_name, str(args.output),
    ], check=True)
    print(gguf_path)


if __name__ == "__main__":
    main()
