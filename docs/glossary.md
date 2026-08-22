# Glossary

## Autoregressive generation

Producing text by repeatedly predicting one next token and feeding that token back as input.

## Batch

A collection of independent input sequences processed together. `v0.1-single` fixes batch size to
one.

## Causal mask

An attention mask preventing a token from reading positions to its right. It preserves the
next-token-prediction constraint during multi-token prefill.

## Checkpoint

Serialized model parameters and accompanying configuration/tokenizer artifacts. The supported
checkpoints use safetensors for weights.

## Cold start

Elapsed time from generation request entry through completion, including artifact resolution and
model loading.

## Decode

The generation phase after prefill. Each forward call consumes one previously generated token,
appends one K/V position per layer, and produces the next logits.

## Dtype

The numeric representation of tensor elements. F32 uses four bytes; F16 and BF16 use two. v0.1
generation is validated for CPU/F32.

## EOS

End-of-sequence token. Selecting it stops generation; it is not included in the completion.

## GQA

Grouped-query attention. Multiple query heads share each key/value head, reducing KV projection
and cache size relative to multi-head attention.

## Greedy sampling

Deterministic selection of the highest-logit token using `argmax`. Despite the common term
“sampling,” no randomness is involved.

## Head dimension

The feature width of one attention head: `hidden_size / num_attention_heads`.

## Hidden state

The per-token vector carried through transformer blocks. Its width is the model hidden size `H`.

## KV cache

Per-layer storage of attention keys and values for positions already processed. It prevents
recomputation of the entire prefix on each decode step.

## Logits

Unnormalized scores over the vocabulary. The largest logit is the greedy next token.

## MHA

Multi-head attention, where every query head has its own key and value head (`K = Q`).

## MQA

Multi-query attention, where all query heads share one key and one value head (`K = 1`).

## Placement

The planner's comparison between logical persistent bytes and an optional declared per-rank
budget. It is not OS memory enforcement.

## Prefill

The first model forward over every prompt token. It populates the KV cache and returns logits for
the first generated token.

## Rank

One participant in distributed execution. The baseline has a single participant, rank 0, and
records that identity in plans, events, and reports.

## RMSNorm

Root-mean-square normalization. It rescales hidden vectors using their RMS magnitude and learned
weights without subtracting a mean.

## RoPE

Rotary position embeddings. Position-dependent rotations are applied to queries and keys so their
dot products encode relative position.

## Safetensors

A tensor serialization format whose metadata exposes tensor names, shapes, dtypes, and byte
offsets without executing model code.

## SwiGLU

The gated Llama MLP: `down(SiLU(gate(x)) × up(x))`.

## Token

An integer ID produced by a tokenizer from text. Tokens may represent words, fragments,
punctuation, whitespace, or control markers.

## Tokenizer

The model-specific conversion between text and token IDs. Chat-template rendering occurs before
encoding.

## TTFT

Time to first token. In this implementation it includes tokenization and post-load work through
prefill and first-token selection, but excludes artifact resolution and model loading.

## Weight tying

Reusing the token embedding matrix as the LM-head matrix. A tied model does not store a separate
`lm_head.weight`.
