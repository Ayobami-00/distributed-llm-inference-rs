# Getting started

This chapter runs the three CLI surfaces and explains what crosses the boundary between the CLI
and the runtime. The next chapters unpack the implementation behind each result.

## Build

The workspace requires Rust 1.85 or newer. From the repository root:

```console
cargo build --release --locked
```

The executable is `target/release/dlir`. The two crates have separate responsibilities:

- `dlir-cli` parses commands and owns terminal and file output.
- `dlir-runtime` owns model knowledge, inference, memory planning, and reports.

## List supported models

```console
./target/release/dlir models
./target/release/dlir models --format json
```

`models` reads the compiled registry and never contacts Hugging Face. The registry is closed:
only `smollm2-135m-instruct` and `tinyllama-1.1b-chat` are accepted. See
[Model registry, artifacts, and prompts](registry-artifacts-and-prompts.md) for why this is a
correctness boundary rather than only a user-interface choice.

## Inspect a model without downloading it

```console
./target/release/dlir inspect \
  --model smollm2-135m-instruct \
  --dtype f32 \
  --context-length 512 \
  --device-memory-budget 500MiB
```

For this request, inspection derives:

```text
Logical weights:    513.1 MiB (538060032)
KV cache capacity:  22.5 MiB (23592960)
Persistent minimum: 535.6 MiB (561652992)
Device budget:      500.0 MiB (524288000)
Placement:          FAILED
```

The command still exits successfully: it answered the question, and the answer is that the
persistent estimate does not fit. `inspect` accepts F16 and BF16 for hypothetical planning even
though generation is validated only for CPU/F32. The budget is a user-declared planning value; it
does not reserve or constrain host memory.

Use `--format json` for the schema-v1 representation and `--output PATH` to write it to a file.

## Generate a completion

This run uses SmolLM2 on the CPU and writes the structured result to `run.json`:

```console
./target/release/dlir generate \
  --model smollm2-135m-instruct \
  --device cpu \
  --dtype f32 \
  --prompt "Explain tensor parallelism." \
  --max-new-tokens 32 \
  --report run.json
```

One observed run produced:

```text
model: smollm2-135m-instruct (HuggingFaceTB/SmolLM2-135M-Instruct)
revision: 12fd25f77366fa6b3b4b768ec3050bf629380bac
checkpoint: model.safetensors (known download size 256.6 MiB)
resolving configuration and tokenizer...
resolving checkpoint weights...
loading model on CPU...
prefill: 34 prompt tokens
Tensor parallelism is a mathematical concept that describes the relationship between two tensors, often denoted as A and B, where the second tensor is a linear transformation that maps the
report: run.json

RUN SUMMARY
prompt tokens:       34
generated tokens:    32
stop reason:         max_new_tokens
logical weights:     513.1 MiB
KV capacity:         2.9 MiB for 66 tokens
final KV used:       2.9 MiB
placement:           NOT EVALUATED
artifacts:           0.017 s
model load:          4.115 s
tokenization:        8.849 ms
prefill:             167.305 ms (203.22 tok/s)
TTFT:                179.393 ms
decode forwards:     31
mean decode latency: 21.047 ms
decode throughput:   47.51 tok/s
generation total:    0.834 s
cold start total:    5.032 s
```

The generated wording is deterministic for the same pinned artifacts and request. Timings depend
on the processor, operating system, current system load, and whether Hugging Face artifacts are
already cached.

### Why the terminal output looks mixed

The process uses separate streams:

| Destination | Content |
| --- | --- |
| Standard output | Assistant completion only |
| Standard error | Artifact progress, model status, report path, and human summary |
| `--report` file | Complete schema-v1 JSON report |

The terminal renders stdout and stderr together. Their separation becomes visible when piping:

```console
./target/release/dlir generate \
  --model smollm2-135m-instruct \
  --prompt "Explain tensor parallelism." \
  > completion.txt
```

Only the completion goes to `completion.txt`; diagnostics remain visible.

## What the command did

At a high level, generation performed the following operations:

1. Looked up the pinned `ModelSpec`.
2. Downloaded and validated small configuration and tokenizer artifacts.
3. Rendered the model-specific chat template and tokenized it without adding duplicate special
   tokens.
4. Planned weight and KV-cache memory and checked any explicit budget.
5. Downloaded and validated the checkpoint tensor manifest.
6. Loaded the owned Llama implementation and allocated a per-layer cache.
7. Ran prompt prefill once.
8. Repeated one-token decode and greedy `argmax` selection.
9. Streamed token text and assembled metrics, events, and the JSON report.

Follow that sequence through the system in [Architecture and request flow](architecture.md), or
start at the individual Rust symbols in the [code-reading guide](code-reading-guide.md).
