# Code-reading guide

This guide contains two executable paths. The first follows `dlir generate`; the second follows
the smaller `dlir p2p` communication path.

## 1. CLI parsing and request construction

Start at [`main`](../crates/cli/src/main.rs). `Cli` and `Command` define the closed command surface.
The `Generate` arm constructs `GenerationRequest` and passes it with a `CliObserver` to
`run_generate`.

Notice the ownership boundary: the CLI decides where output goes, while the request contains no
terminal concepts. See [Getting started](getting-started.md#why-the-terminal-output-looks-mixed).

## 2. Runtime entry and early validation

Open [`generate`](../crates/runtime/src/generation.rs). The function rejects zero new tokens and an
empty prompt, obtains `SupportedModelId::spec`, validates CPU/F32 support, and checks head
divisibility before network access.

`EventRecorder` is constructed here. All later progress events share its cold-start clock and
rank-0 identity; the same events are delivered live and retained in the final report.

## 3. Registry and small artifacts

Follow `SupportedModelId::spec` into [`registry.rs`](../crates/runtime/src/registry.rs). Compare the
selected `ModelSpec` with `ArtifactRepository::new` and `download_metadata` in
[`artifacts.rs`](../crates/runtime/src/artifacts.rs).

`validate_metadata` reconstructs a `ModelConfig` from `config.json` and requires exact equality
with the registry. This is where the supported Llama subset is enforced. See
[Model registry, artifacts, and prompts](registry-artifacts-and-prompts.md).

## 4. Prompt representation

Return to `generate` and follow `render_prompt` into
[`prompt.rs`](../crates/runtime/src/prompt.rs). The fixed template adds the assistant-generation
suffix. `Tokenizer::encode(..., false)` avoids adding another layer of special tokens.

At this point the request has concrete prompt token IDs, so cache capacity can be exact rather
than based on prompt characters.

## 5. Memory preflight

Read `generation_memory_preflight`, then `RankMemoryPlan::for_model` in
[`memory.rs`](../crates/runtime/src/memory.rs). Track:

```text
prompt tokens → remaining context → effective generation → cache capacity
              → weight bytes + cache bytes → placement verdict
```

If placement fails, `generate` returns before `download_weights`. Compare this with
[`inspect`](../crates/runtime/src/inspect.rs), which returns the same plan as successful inspection
data. The formulas and budget comparison are documented beside `RankMemoryPlan::for_model`.

## 6. Checkpoint validation and model construction

Continue through `download_weights` and `validate_checkpoint`. `expected_tensor_shapes` derives
the complete safetensor manifest from `ModelConfig`.

After validation, an mmap-backed `VarBuilder` enters `Llama::load` in
[`model/llama.rs`](../crates/runtime/src/model/llama.rs). Read its construction from the outside in:

1. token embeddings and tied/untied LM head;
2. one `Block` per registered layer;
3. final RMSNorm;
4. RoPE sine/cosine tables sized to capacity.

Then `KvCache::new` in [`model/cache.rs`](../crates/runtime/src/model/cache.rs) allocates K and V
for every layer. See [The owned Llama forward pass](llama-forward-pass.md).

## 7. Prefill

Back in `generate`, the prompt IDs become a `[1, S]` Candle tensor. The synchronized
`model.forward(input, 0, &mut cache)` call is prefill.

Inside `Llama::forward`, follow embeddings → each `Block::forward` → final norm → last-position
selection → LM head. Within attention, follow Q/K/V projection → RoPE → cache append → GQA repeat
→ scores → causal mask → softmax → value aggregation → output projection.

The returned `[1, V]` logits produce the first token through `greedy_token`; no decode forward was
needed yet. See [KV cache and generation](kv-cache-and-generation.md#prefill).

## 8. Cached decode

The loop feeds the previous generated token as `[1, 1]` at `position = cache.len()`. Inspect
`LayerKvCache::append`: `slice_set` writes one new sequence position and returns only the populated
prefix.

After the synchronized forward, `argmax` selects a token. EOS breaks before emission; otherwise
`emit_token` records the ID and text and notifies the observer. Limits are checked before another
forward. See [KV cache and generation](kv-cache-and-generation.md#decode).

## 9. Report and presentation

`build_timings` converts measured `Duration` values to integer nanoseconds and calculated rates.
`GenerationReport` combines identity, topology, memory, result, timings, and recorded events.

Return to CLI `run_generate`: `CliObserver::finish` reconciles streamed fragments with the final
completion, the optional JSON file is written, and `generation_summary` formats stderr. See
[`report.rs`](../crates/runtime/src/report.rs) for the schema-v1 event, timing, topology, and result
fields.

## 10. Read the proofs

End in the unit tests next to `model/llama.rs`. The `compare_with_oracle` test is the shortest
executable specification of the model: identical deterministic weights enter dlir and Candle,
then prefill and cached logits are compared. Follow with memory, artifact, generation, CLI, and
ignored end-to-end tests to see how each module's boundary is exercised.

## Inspect path shortcut

For `dlir inspect`, the shorter path is:

```text
CLI Command::Inspect
  → InspectionRequest
  → inspect
  → SupportedModelId::spec
  → RankMemoryPlan::for_model
  → InspectionReport
  → CLI text/JSON renderer
```

No artifact, tokenizer, Candle device, model, or cache object is created.

## Point-to-point path

For `dlir p2p --world-size 2`, follow this order:

1. `Command::P2p` in [`cli/main.rs`](../crates/cli/src/main.rs) selects text or JSON and calls
   `run_p2p_ring` with a five-second receive deadline.
2. `run_p2p_ring` in [`collectives/report.rs`](../crates/collectives/src/report.rs) defines each
   rank's deterministic tensor and its previous/next ring peers.
3. `run_in_memory` in [`collectives/runner.rs`](../crates/collectives/src/runner.rs) creates the
   world, gives one exclusive endpoint to each worker thread, joins every worker, and restores
   rank ordering.
4. `Communicator::send_tensor` converts the Candle tensor into a `TensorPacket`. Follow the copy
   through [`collectives/tensor.rs`](../crates/collectives/src/tensor.rs).
5. `InMemoryTransport::send` places the owned packet on the destination's source-specific FIFO
   channel. `recv` matches source and tag, retaining other tags in its pending queue.
6. `Communicator::recv_tensor` reconstructs a new CPU tensor. The worker compares it with the
   deterministic single-process values for the source rank.
7. The CLI renders the ordered schema-v1 `P2pReport`; a failed verification returns a nonzero exit.

No registry, artifact, tokenizer, Llama, cache, generation report, TCP socket, or process launcher
is involved. The conceptual boundary and failure cases are described in
[Ranks and point-to-point communication](ranks-and-point-to-point.md).

## TCP Docker path

For `dlir launch --nproc 2 --total-cpus 1 --total-memory 512MiB`:

1. `Command::Launch` constructs a `LaunchRequest` and enters `run_launch` in
   [`cli/launch.rs`](../crates/cli/src/launch.rs).
2. The launcher parses totals, reads `docker info`, divides resources, and validates the plan
   before creating anything.
3. Docker starts one `dlir rank` process per constrained container on a private labelled network.
4. Each rank reads its own cgroup files and rejects any CPU or memory mismatch before rendezvous.
5. `TcpTransport::connect` in [`collectives/tcp.rs`](../crates/collectives/src/tcp.rs) binds the
   peer listener, registers with rank 0, validates the peer table, and establishes the full mesh.
6. `Communicator::barrier`, `send_tensor`, and `recv_tensor` run the startup barrier, TCP ring,
   and completion barrier using versioned frames.
7. Each process writes one JSON report. The launcher collects rank logs, restores rank ordering,
   renders the final report, and removes only resources labelled with the run ID.

Continue with [TCP rendezvous and barrier](tcp-rendezvous-and-barrier.md) and
[Docker resource topologies](docker-topologies.md).
