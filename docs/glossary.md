# Glossary

## All-gather

A collective in which every rank contributes a distinct shard and every rank receives the
rank-ordered concatenation. Tensor-parallel generation uses it to reconstruct vocabulary logits.

## All-reduce

A collective that elementwise-reduces equal tensors and returns the complete result to every
rank. The native centralized implementation reduces on rank 0 then broadcasts; the ring
implementation combines reduce-scatter and all-gather.

## Collective

An operation in which every rank in a communication group participates in the same sequence.
The v0.5 native backend derives broadcast, reduce, all-gather, reduce-scatter, all-to-all, and
all-reduce solely from point-to-point send/receive.

## Activation

The intermediate residual hidden-state tensor crossing from one pipeline stage to the next.
Prefill transfers `[1,S,H]`; cached decode transfers `[1,1,H]`.

## Autoregressive generation

Producing text by repeatedly predicting one next token and feeding that token back as input.

## Batch

A collection of independent input sequences processed together. `v0.1-single` fixes batch size to
one.

## Barrier

A reusable synchronization point that releases a generation only after every rank has arrived.
The TCP implementation uses rank 0 to collect arrivals and send releases.

## Cgroup

The Linux kernel mechanism Docker uses to enforce CPU quotas and memory limits. Each rank reads
its effective cgroup values and compares them with the launch plan.

## Container

The isolated Linux environment hosting exactly one rank process from v0.3 onward. Containers
share one trusted Docker Engine and private bridge network.

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

## Control packet

A bounded owned non-tensor payload carried by protocol v2. Pipeline execution uses typed control
packets for final-stage token feedback and rank-0 continue/stop decisions.

## Dtype

The numeric representation of tensor elements. F32 uses four bytes; F16 and BF16 use two.
Generation through v0.5 is validated for CPU/F32.

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
For `dlir pipeline`, the same comparison uses the memory limit that Docker then enforces on that
rank; it still estimates persistent model/cache state rather than peak process RSS.

## Pipeline parallelism (PP)

Partitioning an ordered model into stages and passing residual activations between them. In v0.4,
`PP=world_size`, stages own contiguous layer ranges, and execution is sequential without
microbatch overlap.

## Pipeline stage

One rank's local model slice: an assigned non-empty transformer-layer range, a local KV cache, and
optionally token embeddings on rank 0 or final normalization/LM head on the last rank.

## Prefill

The first model forward over every prompt token. It populates the KV cache and returns logits for
the first generated token.

## Rank

One participant in distributed execution. In the v0.2 communication path, each rank is hosted by
one worker thread. From v0.3 onward each physical rank owns one process inside one Docker
container. In v0.4 every rank executes one pipeline stage. In v0.5 every tensor rank executes all
layers on a local shard, while rank 0 additionally owns token decisions and completion emission.

## Rendezvous

The startup phase in which ranks register their identity and advertised listener address with
rank 0 and receive the same ordered peer table.

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

## TCP

A reliable ordered byte stream. `TcpTransport` adds explicit message framing because TCP itself
does not preserve application message boundaries.

## Transport

The boundary that moves owned tagged messages between ranks. v0.2 provides in-memory tensor
channels; v0.3 adds full-mesh TCP; protocol v2 in v0.4 adds a separate bounded control-frame kind
without weakening source/tag matching.

## Tensor parallelism (TP)

Sharding matrix dimensions and attention heads so every rank executes every transformer layer on
different slices. Partial row-parallel outputs are summed with all-reduce. In v0.5 TP is strict,
equal, CPU/F32, and identical to the Docker/TCP world size.

## TUI

The optional terminal user interface that reduces the same distributed event stream retained by
the report. It is observational only and has no cluster-management capability.

## TTFT

Time to first token. Single-rank and pipeline reports exclude artifact/model loading. The pipeline
measurement begins when rank 0 starts prefill after the startup barrier and ends when its first
non-EOS token is available.

## Weight tying

Reusing the token embedding matrix as the LM-head matrix. A tied model does not store a separate
`lm_head.weight`. When embeddings and the output head belong to different pipeline processes, the
same checkpoint tensor is materialized independently on both ranks and reported as duplication.

## World size

The number of ranks participating in one communication world. Valid global ranks are the
contiguous integers from zero through `world_size - 1`.
