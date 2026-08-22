# Tensor-parallel Llama and GQA

## Partition invariant

Every tensor rank executes every layer, but owns equal slices of large matrices:

| Component | Checkpoint axis | Local result |
| --- | ---: | --- |
| embedding and LM head | vocabulary rows (`0`) | `V/TP` rows |
| Q/K/V, gate, up | output rows (`0`) | local heads/features |
| attention output, down | input columns (`1`) | partial full-width output |
| RMSNorm | replicated | full `H` vector |

[`ParallelContext`](../crates/runtime/src/parallel.rs) records TP/PP/EP identities.
[`TensorParallelLlama`](../crates/runtime/src/model/tensor_parallel.rs) composes vocabulary-parallel
embedding, sharded GQA attention, sharded SwiGLU, replicated norms, and a vocabulary-parallel head.
Real checkpoints use Candle's mmap-backed sharded builder, so a rank converts only its assigned
slices to F32; offline fixtures use the same graph after slicing full in-memory tensors.

The attention path maps `[1,S,H]` to local queries `[1,Q/TP,S,D]` and compact keys/values
`[1,K/TP,S,D]`. Only K and V are cached, with persistent shape `[1,K/TP,C,D]` per tensor and
layer. Local attention produces `[1,S,H/TP]`; the input-column-sharded output projection produces
a partial `[1,S,H]`, and all-reduce reconstructs the residual update. The MLP repeats this pattern:
gate/up produce `I/TP`, local SwiGLU multiplies them, and the down projection is all-reduced.

Strict equal sharding requires `V`, `H`, `I`, `Q`, and `K` to divide by TP. SmolLM2 has `Q=9` and
`K=3`, so distributed execution accepts TP=3 and rejects TP=2/4. TinyLlama has `Q=32` and `K=4`,
so TP=2 and TP=4 are valid.

Per-rank persistent planning is:

```text
local sharded weights + replicated RMSNorm weights
  + 2 × L × C × (K/TP) × D × 4 bytes
```

The aggregate materialized parameter count exceeds the architectural count by the extra RMSNorm
copies. [`TensorParallelPartition`](../crates/tensor/src/plan.rs) computes these values and applies
the exact `persistent <= cgroup limit` placement boundary before weight download.

During generation every rank receives the same token IDs, executes all layers and vocabulary
logit all-gather, then rank 0 chooses `argmax` and sends a typed token/continue/stop decision. The
synthetic test in
[`tensor_parallel.rs`](../crates/runtime/src/model/tensor_parallel.rs) compares prefill and cached
decode logits with the monolithic owned Llama at `1e-4`, for tied/untied heads and centralized/ring
all-reduce.

## Live observation

Every rank publishes sequenced layer, native-collective, control, token, barrier, and completion
events as flushed JSON Lines. The host validates each rank-local sequence and feeds the identical
records to text progress, the final `TensorParallelReport`, and the optional Ratatui reducer. The
tensor dashboard shows rank shard ranges, the active layer and collective/ring step, logical and
cgroup memory, communication bytes, TTFT, and decode progress. It is observational only: `q`/Esc
disables rendering while inference continues, and it never sends rank commands.
