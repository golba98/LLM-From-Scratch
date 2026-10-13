# EmbeddingGemma representations and Codexa conditioning

## Evidence

The pinned text-only EmbeddingGemma 2 loads 271,002,624 frozen parameters. A real CPU FP32 probe returned `[1, 15, 768]` projected token states, with 24 intermediate `[1, 15, 512]` tensors followed by the projected final state. No parameters were trainable. The reproducible result is `logs/memory/token-state-probe-v1.json`; use `scripts/diagnostics/probe_embedding_states.py` to repeat it.

The installed Transformers implementation constructs bidirectional full/sliding masks and projects each final 512-dimensional text state to 768 dimensions before Sentence Transformers pools and normalizes it. `output_hidden_states=True` exposes token states; the final tensor is projected, so do not assume it has width 512. A hook on the final text normalization could expose the final unprojected tensor. See [Transformers EmbeddingGemma2 documentation](https://huggingface.co/docs/transformers/model_doc/embedding_gemma2).

Pooled, normalized 768-dimensional retrieval vectors compress a whole passage. They are useful for selection and cannot substitute for an autoregressive vocabulary head or token-level response training. EmbeddingGemma's tokenizer has a 262,144-entry vocabulary; Codexa's verified tokenizer has 16,384 entries. Their token IDs and segmentation are incompatible.

## Architecture comparison

| Architecture | Compatibility and training | Compute and likely benefit |
| --- | --- | --- |
| A: Retrieval-augmented Codexa | Implemented. Existing model/checkpoint/tokenizer; frozen encoder selects text that Codexa tokenizes itself. No new decoder weights. | Both models fit together in measured inference. Recovers otherwise omitted history, but does not repair fluent generation or knowledge by itself. |
| B: Encoder-conditioned Codexa | Accessible token states condition new cross-attention modules. Codexa still generates with its own tokenizer. New modules require supervised alignment and a versioned checkpoint/inference format. | Potential richer conditioning than pooled vectors. Adapter-only training with frozen encoder/decoder is plausible on 16 GB, but has not been profiled or validated. Full decoder training leaves less activation headroom. |
| C: New encoder-decoder | EmbeddingGemma remains the encoder; a decoder is initialized separately or reused with newly added cross-attention. A random decoder still needs substantial language training. | Greatest migration and training risk. A pretrained decoder could supply fluency, but would change the existing custom-model goal and is not evaluated here. |

For B, keep the encoder frozen and in evaluation mode initially. Encode only the user request and available earlier context, never the target assistant answer. Train the decoder with teacher-forced assistant-response next-token cross-entropy and explicit end-of-turn supervision. Independent tokenizers can share the underlying text; cross-attention aligns sequences without one-to-one token matching. Mask padded encoder states. Use `no_grad()` for frozen encoder execution while retaining tensors usable by trainable projections.

A projection between widths is required somewhere. It can be a separate 512→1536 or 768→1536 map, or the cross-attention key/value projections themselves. A pooled embedding alone would be a lossy conditioning bottleneck. A small first research variant could add zero-gated cross-attention every fourth decoder layer, freezing original weights initially. This is a proposal, not implemented or benchmarked.

Six cross-attention blocks with 1536-wide queries/outputs and 512-wide encoder inputs require approximately 37.75M projection weights, before normalization/gates. With 768-wide inputs the figure is 42.47M. A separate 512→1536 projection adds 786,432 weights. These are arithmetic architecture estimates; they are not memory measurements. Backpropagation still requires decoder activations above adapters, even when original decoder weights are frozen. Checkpointing, fused attention and short encoder sequences would require actual preflight.

Measured A inference used at most 7,647 MiB total GPU memory in the preliminary combined-model workload, including the desktop. Existing full-decoder 8-bit-AdamW BF16 SFT uses about 12.2 GiB reserved in the Stage 2 pilot. Adding cross-attention training and encoder workspaces cannot be assumed to fit simply because the weights fit. Ordinary FP32 full-parameter AdamW for roughly 1.2B joint parameters would need around 19 GB for weights, gradients and moments alone, before activations. Frozen components/adapters or 8-bit optimizers change that calculation, but only a new measured preflight can decide B/C feasibility.

## Recommendation

Continue with A and assess the actual generative bottleneck first. The saved repair experiment repeated a small set of anchors and generalized poorly; its validation loss flattening did not predict reliable unseen instructions. The Stage 2 pilot tests diverse local conversational examples from the verified base, using held-out source/root groups and separately authored behavioral prompts. Do not infer training duration or conversational capability from pooled embeddings or one validation-loss improvement.

The base received approximately 655.36M tokens for 934.36M parameters, about 0.70 tokens per parameter. This is evidence for investigating undertraining, not proof that more pretraining alone solves instruction following. The compute-optimal training literature emphasizes scaling training tokens with model size; it does not specify a guaranteed target for this corpus or GPU. [Hoffmann et al., Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)

Cross-attention retrofitting has research precedent: RETRO combines a frozen retriever with a differentiable encoder and chunked cross-attention, including experiments retrofitting pretrained transformers. Those large-corpus results do not establish that a frozen EmbeddingGemma adapter will improve this undertrained decoder. [Borgeaud et al., Improving language models by retrieving from trillions of tokens](https://arxiv.org/abs/2112.04426)

Before B, require a tiny alignment/overfit diagnostic, verified attention masks, no target leakage, token-level checkpoint equivalence with zero gates, measured VRAM at realistic lengths, and a held-out improvement over A under equivalent training budgets. Estimate necessary training only after measuring throughput and learning curves. No credible hour/token estimate for B or C exists from the current measurements. C should remain a later research option rather than this project's next implementation step.
