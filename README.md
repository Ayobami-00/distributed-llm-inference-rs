# distributed-llm-inference-rs

`dlir` is a learning-oriented distributed LLM inference runtime written in Rust. The
`v0.1-single` checkpoint establishes a deliberately small baseline: one model, one process,
one CPU device, batch size one, and deterministic greedy generation.

The workspace contains two crates:

- `dlir-runtime` owns the model registry, artifact validation, Llama forward path, KV cache,
  memory planning, generation, events, and reports.
- `dlir-cli` provides the `dlir` command and owns terminal/file output and exit behavior.

## Build

Rust 1.85 or newer is required.

```console
cargo build --release --locked
./target/release/dlir models
```

Candle is pinned to 0.11.0. The relevant Candle crates are vendored with narrow compatibility
patches for Rust 1.85: an unused AArch64 FP16 specialization is disabled, integer divisibility
checks use stable arithmetic, and Candle's ZIP dependency is pinned to its API-compatible
Rust-1.83 release. These changes do not affect the CPU/F32 execution path supported here.

## Supported models

The registry is closed by design. Model IDs, repositories, revisions, configurations, tensor
layouts, checkpoint dtypes, chat templates, and parameter counts are compiled into the binary.
There is no arbitrary Hub ID, local checkpoint path, revision override, or custom template.

| `dlir` model ID | Pinned Hugging Face repository | Parameters | Max context | CPU/F32 | CUDA |
| --- | --- | ---: | ---: | --- | --- |
| `smollm2-135m-instruct` | `HuggingFaceTB/SmolLM2-135M-Instruct` at `12fd25f77366fa6b3b4b768ec3050bf629380bac` | 134,515,008 | 8,192 | validated | planned |
| `tinyllama-1.1b-chat` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` at `5243d158d6f4b356f1142ea8fd6a99cb5ac2c0e1` | 1,100,048,384 | 2,048 | validated | planned |

List the registry without network access:

```console
dlir models
dlir models --format json
```

## Inspect and plan memory

`inspect` uses only embedded registry metadata and never downloads model artifacts.

```console
dlir inspect \
  --model smollm2-135m-instruct \
  --dtype f32 \
  --context-length 512 \
  --device-memory-budget 500MiB
```

The plan contains the architectural parameter breakdown, logical weight bytes, KV-cache
capacity, persistent minimum, and a rank-0 placement verdict. KV capacity is calculated as:

```text
2 × layers × context × KV heads × head dimension × dtype bytes
```

The budget accepts raw bytes or the unambiguous IEC suffixes `KiB`, `MiB`, and `GiB`.
Suffixes such as `M`, `MB`, decimal values, and negative values are rejected. In v0.1 the
budget is a host-domain, user-declared, per-rank planning input. It is not detected from the
machine and is not enforced through the OS, Docker, cgroups, RSS limits, or an accelerator.

A failed placement remains a successful inspection and exits zero. Select JSON and/or write
the result to a file with `--format json --output inspection.json`.

The persistent estimate is intentionally not peak process memory. It excludes activations,
operator workspaces, allocator fragmentation, and runtime overhead.

## Generate

```console
dlir generate \
  --model smollm2-135m-instruct \
  --device cpu \
  --dtype f32 \
  --prompt "Explain tensor parallelism." \
  --max-new-tokens 32 \
  --report run.json
```

On the first run, `dlir` resolves the pinned configuration, tokenizer, and tokenizer
configuration; validates them; applies the model's fixed one-user-message chat template; and
tokenizes the prompt. It then performs the exact memory preflight before resolving the weight
file. A failed placement exits non-zero before checkpoint download. Hugging Face's local cache
is reused on later runs.

After download, `dlir` verifies every safetensor name, shape, dtype, and the total parameter
count before loading. Checkpoint access is memory-mapped through a narrowly isolated unsafe
boundary. Runtime weights are materialized as F32.

Only the assistant completion is written to stdout. Artifact status, model-load status, and a
human-readable metrics summary go to stderr, so completion text can be piped safely. The model
uses prefill, a preallocated per-layer KV cache, one-token cached decode, and argmax sampling.
EOS is not printed.

CPU execution rejects F16 and BF16. Those dtypes are accepted by `inspect` only for hypothetical
logical memory plans.

### JSON run report

`--report PATH` writes a schema-versioned JSON document. Schema version 1 includes:

- the model ID, pinned repository and revision, device, dtype, and `world_size=1` topology;
- completion text, prompt/generated token counts and IDs, and stop reason;
- rank-0 memory plan, planned cache capacity, and final populated cache bytes;
- integer nanosecond timings for artifact resolution, model loading, tokenization, prefill,
  TTFT, decode, generation, and cold start, plus throughput values;
- ordered rank-0 events for artifacts, load, prefill, decode steps, emitted tokens, and
  completion.

Generation time excludes artifact/model loading but includes tokenization. Cold-start time
includes all work. Candle's device is synchronized around measured model operations.

## Test

The normal suite is fully offline and uses tiny deterministic synthetic Llama fixtures:

```console
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --no-deps -- -D warnings
cargo test --workspace --all-targets
```

The synthetic model is checked against Candle's reference Llama implementation at F32 and
also verifies full recomputation against prefill plus cached decode. Real-checkpoint tests are
ignored by default:

```console
cargo test -p dlir-runtime --test e2e -- --ignored --nocapture
```

That command can download approximately 269 MB for SmolLM2 and 2.2 GB for TinyLlama and may
require several gigabytes of host memory because runtime weights are F32.

## Release checkpoints

Each tag is intended to leave the project in a runnable, measurable state that supports one
checkpoint in the accompanying article. Only tags marked **implemented** currently exist;
the remaining names describe the planned progression and may be refined as the work develops.

| Tag | Status | What it contains |
| --- | --- | --- |
| [`v0.1-single`](https://github.com/Ayobami-00/distributed-llm-inference-rs/tree/v0.1-single) | Released | A single CPU process and device; the closed model registry; model inspection and memory planning; an owned Llama forward path; prefill and cached decode; greedy generation; structured events, metrics, and JSON reports. |
| `v0.2-collectives` | Planned | In-memory ranks, point-to-point send/receive, and correctness-first native collective algorithms tested against single-process tensor results. |
| `v0.3-tcp` | Planned | One rank per process, TCP transport, rendezvous, process startup, and reproducible multi-container CPU topologies. |
| `v0.4-pipeline` | Planned | Pipeline-stage model partitioning, rank-local weight loading, activation transfer between stages, and autoregressive token feedback. |
| `v0.5-tensor` | Planned | Column- and row-parallel linear layers, sharded attention and MLP execution, tensor-parallel collectives, and distributed generation. |
| `v0.6-hybrid` | Planned | Process groups and combined tensor and pipeline parallelism, including topology-aware memory and communication measurements. |
| `v0.7-cuda-nccl` | Planned | CUDA execution and an NCCL communicator implementing the same distributed semantics used by the CPU backends. |
| `v0.8-expert` | Planned | A small mixture-of-experts model, expert placement, top-k token routing, and all-to-all expert-parallel execution. |
| `v1.0-lab` | Planned | PP-versus-TP experiments, per-rank observability, communication and idle-time metrics, and a terminal UI for comparing distributed inference trade-offs. |
