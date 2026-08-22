# Project documentation

This documentation explains the implementation through `v0.5-tensor`: single-process
inference, in-memory point-to-point communication, one TCP rank process per Docker container,
correctness-first pipeline stages, and strict tensor-sharded Llama execution. It is written for
readers who know basic Rust but are new to transformer inference and distributed runtimes.

The goal is not merely to describe the modules. Each guide connects an inference concept to its
equations, tensor shapes, Rust implementation, runtime output, and correctness tests.

## Choose a reading path

### Concept-first

Use this path to understand what an inference runtime does before following individual Rust
functions:

1. [Getting started](getting-started.md)
2. [Architecture and request flow](architecture.md)
3. [Ranks and point-to-point communication](ranks-and-point-to-point.md)
4. [TCP rendezvous and barrier](tcp-rendezvous-and-barrier.md)
5. [Docker resource topologies](docker-topologies.md)
6. [Model registry, artifacts, and prompts](registry-artifacts-and-prompts.md)
7. [The owned Llama forward pass](llama-forward-pass.md)
8. [KV cache and generation](kv-cache-and-generation.md)
9. [Pipeline partitioning and execution](pipeline-parallelism.md)
10. [Distributed events and the observational TUI](events-and-tui.md)
11. [Native collectives and benchmarks](native-collectives.md)
12. [Tensor-parallel Llama and GQA](tensor-parallelism.md)
13. [Code-reading guide](code-reading-guide.md)
14. [Glossary](glossary.md)

### Code-first

Start with the [code-reading guide](code-reading-guide.md). It contains ordered paths for
`dlir generate`, `dlir p2p`, `dlir launch`, `dlir pipeline`, and `dlir tensor`, with links to the relevant
theory.

## Notation

The guides use these symbols consistently:

| Symbol | Meaning |
| --- | --- |
| `B` | Batch size; fixed at `1` through v0.5 |
| `S` | Tokens in the current forward call |
| `T` | Total cached key/value length visible to attention |
| `C` | Allocated KV-cache capacity |
| `H` | Model hidden size |
| `I` | MLP intermediate size |
| `Q` | Number of query/attention heads |
| `K` | Number of key/value heads |
| `D` | Head dimension, `H / Q` |
| `L` | Number of transformer layers |
| `V` | Vocabulary size |

Tensor shapes are written in brackets. For example, `[1, K, C, D]` is a batch-one KV-cache
tensor with `K` heads, capacity `C`, and head dimension `D`.

## Source boundary

The first-party implementation lives in [`crates/runtime`](../crates/runtime/src/lib.rs),
[`crates/collectives`](../crates/collectives/src/lib.rs),
[`crates/pipeline`](../crates/pipeline/src/lib.rs), [`crates/tui`](../crates/tui/src/lib.rs), and
[`crates/tensor`](../crates/tensor/src/lib.rs),
[`crates/cli`](../crates/cli/src/main.rs). The [`vendor`](../vendor) directory contains third-party
Candle sources used through narrow compatibility overrides and is outside this code tour.

## Scope

These guides describe behavior implemented through `v0.5-tensor`. Future parallelism
checkpoints are listed in the root [README](../README.md#release-checkpoints), but their designs
are not presented here as implemented behavior.
