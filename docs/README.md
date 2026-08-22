# v0.1-single documentation

This documentation explains the complete `v0.1-single` implementation: one model, one process,
one CPU device, batch size one, and deterministic greedy generation. It is written for readers
who know basic Rust but are new to transformer inference.

The goal is not merely to describe the modules. Each guide connects an inference concept to its
equations, tensor shapes, Rust implementation, runtime output, and correctness tests.

## Choose a reading path

### Concept-first

Use this path to understand what an inference runtime does before following individual Rust
functions:

1. [Getting started](getting-started.md)
2. [Architecture and request flow](architecture.md)
3. [Model registry, artifacts, and prompts](registry-artifacts-and-prompts.md)
4. [The owned Llama forward pass](llama-forward-pass.md)
5. [KV cache and generation](kv-cache-and-generation.md)
6. [Code-reading guide](code-reading-guide.md)
7. [Glossary](glossary.md)

### Code-first

Start with the [code-reading guide](code-reading-guide.md). It follows a `dlir generate`
invocation from CLI parsing to the final report and links to the relevant theory when a concept
first appears.

## Notation

The guides use these symbols consistently:

| Symbol | Meaning |
| --- | --- |
| `B` | Batch size; fixed at `1` in v0.1 |
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

The first-party implementation lives in [`crates/runtime`](../crates/runtime/src/lib.rs) and
[`crates/cli`](../crates/cli/src/main.rs). The [`vendor`](../vendor) directory contains third-party
Candle sources used through narrow compatibility overrides and is outside this code tour.

## Scope

These guides describe behavior that exists in `v0.1-single`. Future parallelism checkpoints are
listed in the root [README](../README.md#release-checkpoints), but their designs are intentionally
not presented here as implemented behavior.
