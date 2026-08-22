# Architecture and request flow

`v0.1-single` is the control experiment for later distributed checkpoints. All model weights,
compute, KV state, token selection, and reporting belong to rank 0.

```text
world_size = 1
rank       = 0
TP         = 1
PP         = 1
EP         = 1
```

## Workspace boundaries

```mermaid
flowchart LR
    User[User or shell] --> CLI[dlir-cli]
    CLI --> Runtime[dlir-runtime]
    Runtime --> Registry[Registry and planning]
    Runtime --> Artifacts[Artifact validation]
    Runtime --> Model[Owned Llama and KV cache]
    Runtime --> Reports[Events and reports]
    Artifacts --> Hub[Hugging Face cache]
    Model --> Candle[Candle tensors and neural-network primitives]
```

`dlir-cli` depends on `dlir-runtime`; the runtime never depends on the CLI. This direction lets a
future interface consume the same requests, events, and reports without duplicating inference.

## First-party module map

| Module | Responsibility | Main concepts |
| --- | --- | --- |
| [`cli/main.rs`](../crates/cli/src/main.rs) | Parse arguments; render text/JSON; stream completion | Interface boundary, stdout/stderr |
| [`registry.rs`](../crates/runtime/src/registry.rs) | Describe the only accepted models | Reproducibility, architecture contract |
| [`artifacts.rs`](../crates/runtime/src/artifacts.rs) | Resolve and validate Hub files | Staged loading, trust boundary |
| [`prompt.rs`](../crates/runtime/src/prompt.rs) | Render fixed chat templates | Prompt representation, special tokens |
| [`inspect.rs`](../crates/runtime/src/inspect.rs) | Build network-free inspection reports | Architecture metadata |
| [`memory.rs`](../crates/runtime/src/memory.rs) | Count parameters and persistent bytes | Placement planning, GQA cache size |
| [`model/llama.rs`](../crates/runtime/src/model/llama.rs) | Load and execute the supported Llama subset | Transformer forward pass |
| [`model/cache.rs`](../crates/runtime/src/model/cache.rs) | Preallocate and append layer K/V state | Autoregressive state |
| [`generation.rs`](../crates/runtime/src/generation.rs) | Orchestrate artifacts, prefill, decode, and timing | Request lifecycle |
| [`report.rs`](../crates/runtime/src/report.rs) | Define stable events and report records | Observability contract |
| [`error.rs`](../crates/runtime/src/error.rs) | Describe expected failure categories | Fail-fast validation |

## Inspection flow

Inspection deliberately stops before artifacts or tensors:

```mermaid
sequenceDiagram
    participant U as User
    participant C as dlir-cli
    participant I as inspect
    participant R as Model registry
    participant M as Memory planner

    U->>C: dlir inspect --model ...
    C->>I: InspectionRequest
    I->>R: SupportedModelId::spec()
    R-->>I: embedded ModelSpec
    I->>M: for_model(spec, dtype, context, budget)
    M-->>I: RankMemoryPlan
    I-->>C: schema-v1 InspectionReport
    C-->>U: text or JSON
```

No `ArtifactRepository` is constructed, so inspection is deterministic and network-free. A failed
placement is report data, not an inspection error.

## Generation flow

```mermaid
sequenceDiagram
    participant U as User
    participant C as dlir-cli
    participant G as generate
    participant R as Registry
    participant H as Artifact repository
    participant P as Prompt/tokenizer
    participant M as Memory planner
    participant L as Llama
    participant K as KV cache
    participant O as Observer/report

    U->>C: dlir generate ...
    C->>G: GenerationRequest + CliObserver
    G->>R: lookup and validate CPU/F32 spec
    G->>H: resolve config/tokenizer files
    H-->>G: local cache paths
    G->>H: validate model config and chat-template markers
    G->>P: render and tokenize prompt
    P-->>G: prompt token IDs
    G->>M: exact capacity and placement preflight
    alt placement fails
        G-->>C: PlacementFailed before weight download
    else placement succeeds or no budget
        G->>H: resolve checkpoint
        G->>H: validate names, shapes, dtypes, parameter count
        G->>L: mmap VarBuilder and load weights
        G->>K: allocate all layer caches
        G->>L: prefill [1, prompt length]
        L->>K: append prompt keys and values
        L-->>G: last-position logits
        G->>G: argmax first token
        G->>O: TokenGenerated
        loop until EOS, token limit, or context limit
            G->>L: decode [1, 1] at cache.len()
            L->>K: append one key/value position
            L-->>G: next-token logits
            G->>G: argmax
            G->>O: decode/token events
        end
        G-->>C: GenerationReport
        C-->>U: completion + summary + optional JSON
    end
```

## Architectural invariants

The implementation defends these properties at module boundaries:

- A model ID always maps to exactly one pinned specification.
- Generation executes only CPU/F32; inspection may model other logical dtypes.
- Batch size is exactly one.
- A forward call's `position` equals the current cache length.
- Prompt tokens are never silently truncated.
- Cache capacity never exceeds the registered model context.
- Checkpoint tensors must exactly match the expected manifest.
- Only the final sequence position is projected to next-token logits.
- All events and memory plans identify rank 0.
- JSON report shapes are versioned independently of human text output.

These invariants turn accidental mismatches into explicit errors close to their source. Unit tests
beside the owning modules exercise the corresponding success and failure paths.
