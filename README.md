# distributed-llm-inference-rs

`dlir` is a learning-oriented distributed LLM inference runtime written in Rust. The
`v0.1-single` checkpoint establishes a deliberately small inference baseline. The
`v0.2-collectives` checkpoint adds logical ranks and copied point-to-point tensor communication
without changing the single-rank model execution path. `v0.3-tcp` moves those ranks into separate
Docker containers, connects them through TCP, and verifies enforced per-rank CPU and memory limits.
`v0.4-pipeline` partitions the owned Llama model into contiguous CPU stages, transfers residual
activations over TCP, feeds selected tokens back to rank 0, and exposes the run through a
read-only terminal dashboard.

The workspace contains five crates:

- `dlir-runtime` owns the model registry, artifact validation, Llama forward path, KV cache,
  memory planning, generation, events, and reports.
- `dlir-collectives` owns rank identity, in-memory and TCP transports, rendezvous, barrier,
  tensor/control packets, and send/receive.
- `dlir-pipeline` owns deterministic stage assignment, stage memory plans, the rank-local
  pipeline state machine, typed control messages, events, and distributed reports.
- `dlir-tui` reduces the event stream into an observational Ratatui dashboard. It owns no
  cluster or process lifecycle operations.
- `dlir-cli` provides `dlir`, launches Docker rank processes, verifies cgroup limits, and owns
  terminal/file output and exit behavior.

## Documentation

The [documentation hub](docs/README.md) connects inference and communication theory to tensor
shapes, runtime behavior, Rust modules, and correctness tests. Two reading paths are available:

- Start with [architecture and request flow](docs/architecture.md) for a concept-first tour.
- Start with the [code-reading guide](docs/code-reading-guide.md) to follow one generation request
  from CLI parsing to its final report.

The guides cover the [model registry and artifact boundary](docs/registry-artifacts-and-prompts.md),
the [owned Llama forward pass](docs/llama-forward-pass.md),
[KV-cached generation](docs/kv-cache-and-generation.md),
[ranks and point-to-point communication](docs/ranks-and-point-to-point.md),
[TCP rendezvous and barrier](docs/tcp-rendezvous-and-barrier.md), and
[Docker resource topologies](docs/docker-topologies.md),
[pipeline partitioning and execution](docs/pipeline-parallelism.md), and the
[event/TUI architecture](docs/events-and-tui.md).

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

## Point-to-point tensor exchange

The first distributed foundation uses one worker thread and one logical CPU device per rank:

```text
one rank = one worker thread = one logical CPU device
```

Run a deterministic two-rank exchange:

```console
dlir p2p --world-size 2
```

Rank 0 sends `[1, 2, 3, 4]` to rank 1 while rank 1 sends `[5, 6, 7, 8]` back to rank 0. For larger
world sizes every rank sends to its next neighbor and receives from its previous neighbor:

```console
dlir p2p --world-size 4 --format json
```

The backend transfers owned CPU/F32 tensor packets through in-memory FIFO channels. It copies
shape and values rather than sharing Candle tensor handles across rank boundaries. The JSON result
uses schema version 1 and contains deterministic, rank-ordered inputs, outputs, and correctness
verdicts. No model artifacts or network access are involved.

## TCP rank containers

Start one rank process per Docker container and split explicit resource totals equally:

```console
dlir launch \
  --nproc 4 \
  --total-cpus 2 \
  --total-memory 1GiB
```

The launcher checks Docker Engine capacity, assigns each rank `0.5` CPU and `256 MiB`, and reads
the resulting cgroup limits back inside every container. Rank 0 coordinates rendezvous, all rank
pairs establish direct persistent TCP connections, and two barriers surround the same
deterministic ring used by `dlir p2p`. The image `dlir:v0.3-tcp` is built from the repository
Dockerfile when missing and reused afterward.

Docker status goes to stderr. The final rank-ordered report goes to stdout; add `--format json`
for schema-v1 output. The topology uses one trusted Docker Engine and publishes no host ports.
It does not load or partition a model.

## Pipeline-parallel generation

Split one deterministic generation request across two CPU stage containers:

```console
dlir pipeline \
  --model smollm2-135m-instruct \
  --device cpu \
  --dtype f32 \
  --prompt "Explain pipeline parallelism." \
  --max-new-tokens 8 \
  --nproc 2 \
  --total-cpus 2 \
  --total-memory 2GiB \
  --report pipeline-run.json
```

The host applies the fixed chat template and tokenizes once, constructs a balanced contiguous
layer partition, and checks every stage against its enforced container memory limit before it
downloads the checkpoint. It then bind-mounts the validated checkpoint and request manifest
read-only into every rank container. Rank 0 owns embeddings, the final rank owns final RMSNorm
and the LM head, and each rank materializes only its local transformer layers and local KV cache.

Prefill sends `[1, prompt_tokens, hidden_size]` residual activations down the stage chain. Decode
sends `[1, 1, hidden_size]`. The final stage keeps logits local, selects `argmax`, and returns a
typed token control message directly to rank 0. Rank 0 emits the token and broadcasts a typed
continue/stop decision. This correctness-first schedule is sequential: it has no microbatches or
stage overlap and does not claim a throughput speedup.

Add `--tui` to display the read-only Ratatui dashboard on stderr. `q`/Esc closes only the
visualization and generation continues; Ctrl-C requests launcher cleanup. The assistant
completion remains on stdout, progress and metrics use stderr, and `--report` writes the complete
schema-v1 distributed result. See [pipeline partitioning](docs/pipeline-parallelism.md) and
[events and TUI](docs/events-and-tui.md).

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

The normal suite is fully offline except for loopback TCP sockets. It uses tiny deterministic
synthetic Llama fixtures, threaded transports, and child-process communication tests:

```console
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --no-deps -- -D warnings
cargo test --workspace --all-targets --locked
cargo test --workspace --doc --locked
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items --locked
```

The synthetic model is checked against Candle's reference Llama implementation at F32 and
also verifies full recomputation against prefill plus cached decode. Real-checkpoint tests are
ignored by default:

```console
cargo test -p dlir-runtime --test e2e -- --ignored --nocapture
cargo test -p dlir-cli --test pipeline_docker -- --ignored --nocapture
cargo test -p dlir-cli --test pipeline_process -- --ignored --nocapture
```

Those commands can download approximately 269 MB for SmolLM2 and 2.2 GB for TinyLlama and may
require several gigabytes of host memory because runtime weights are F32.

## Release checkpoints

Each tag is intended to leave the project in a runnable, measurable state that supports one
checkpoint in the accompanying article. Only tags marked **Released** currently exist;
the remaining names describe the planned progression and may be refined as the work develops.

| Tag | Status | What it contains |
| --- | --- | --- |
| [`v0.1-single`](https://github.com/Ayobami-00/distributed-llm-inference-rs/tree/v0.1-single) | Released | A single CPU process and device; the closed model registry; model inspection and memory planning; an owned Llama forward path; prefill and cached decode; greedy generation; structured events, metrics, and JSON reports. |
| [`v0.2-collectives`](https://github.com/Ayobami-00/distributed-llm-inference-rs/tree/v0.2-collectives) | Released | Thread-hosted logical ranks; an in-memory transport; copied CPU/F32 tensor packets; tagged point-to-point send/receive; deterministic text and JSON ring demonstrations; and offline communication correctness tests. |
| [`v0.3-tcp`](https://github.com/Ayobami-00/distributed-llm-inference-rs/tree/v0.3-tcp) | Released | One rank per process and Docker container; versioned TCP transport; rank-0 rendezvous; a full peer mesh; reusable barrier synchronization; enforced per-rank CPU/memory limits; and reproducible text/JSON topology reports. |
| [`v0.4-pipeline`](https://github.com/Ayobami-00/distributed-llm-inference-rs/tree/v0.4-pipeline) | Released | Balanced contiguous pipeline stages; rank-local weight and KV-cache materialization; TCP activation transfer; typed autoregressive token feedback; per-stage placement and distributed schema-v1 reports; and an optional read-only Ratatui event dashboard. |
| `v0.5-tensor` | Planned | Column- and row-parallel linear layers, sharded attention and MLP execution, tensor-parallel collectives, and distributed generation. |
| `v0.6-hybrid` | Planned | Process groups and combined tensor and pipeline parallelism, including topology-aware memory and communication measurements. |
| `v0.7-cuda-nccl` | Planned | CUDA execution and an NCCL communicator implementing the same distributed semantics used by the CPU backends. |
| `v0.8-expert` | Planned | A small mixture-of-experts model, expert placement, top-k token routing, and all-to-all expert-parallel execution. |
| `v1.0-lab` | Planned | PP-versus-TP experiments, per-rank observability, communication and idle-time metrics, and a terminal UI for comparing distributed inference trade-offs. |
