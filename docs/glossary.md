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

## Communicator

The tensor-level interface owned by one rank. It converts Candle tensors into owned packets before
calling a transport and reconstructs received packets as new CPU tensors.

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

## Message tag

A caller-selected integer that distinguishes point-to-point messages between the same rank pair.
`recv` matches both the source rank and tag; packets with other tags remain pending.

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

One participant in distributed execution. Generation remains on rank 0. In the v0.2 communication
path, each rank is hosted by one worker thread and owns one logical CPU device.

## Ring exchange

A point-to-point pattern in which rank `r` sends to the next rank and receives from the previous
rank, with both peer calculations wrapping around at `world_size`.

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

## Transport

The boundary that moves owned, tagged tensor packets between ranks. v0.2 provides an in-memory
channel implementation; TCP is deferred to a later checkpoint.

## TTFT

Time to first token. In this implementation it includes tokenization and post-load work through
prefill and first-token selection, but excludes artifact resolution and model loading.

## Weight tying

Reusing the token embedding matrix as the LM-head matrix. A tied model does not store a separate
`lm_head.weight`.

## World size

The number of ranks participating in one communication world. Valid global ranks are the
contiguous integers from zero through `world_size - 1`.
