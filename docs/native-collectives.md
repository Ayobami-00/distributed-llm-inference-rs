# Native collectives and benchmarks

## From point-to-point messages to group operations

[`NativeCollectives`](../crates/collectives/src/collective.rs) owns no sockets. It receives a
transport-backed communicator and derives group operations solely from copied CPU/F32 `send` and
`recv`. Consequently the same algorithms run over in-memory channels and full-mesh TCP.

| Operation | Correctness-first algorithm |
| --- | --- |
| broadcast | root sends directly to every peer |
| reduce | peers send to root; root sums in rank order |
| all-gather | root gathers equal axis shards, concatenates in rank order, broadcasts |
| reduce-scatter | root reduces full tensors, splits an equal axis, scatters |
| all-to-all | deterministic pairwise shard exchanges, then source-order concatenation |
| centralized all-reduce | gather/reduce on rank 0, then broadcast |
| ring all-reduce | ring reduce-scatter, then ring all-gather |

Ring and pairwise exchanges send and receive concurrently, preventing a payload larger than a TCP
socket buffer from making every rank wait in `send`. Collective tags reserve an independent
namespace and encode the rank-local sequence, operation, phase, and step. A receive still matches
an exact source and tag, so stale traffic cannot satisfy a later call.

For a ring with `P` ranks and `N` F32 elements, `N` must divide evenly by `P`. Each rank sends one
`N/P` chunk in each of `P-1` reduce-scatter steps, then repeats that traffic during all-gather.

`dlir collectives check --world-size 4` exercises all six public operations, including both
all-reduce algorithms. [`collectives.rs`](../crates/collectives/tests/collectives.rs) proves exact
rank ordering, multidimensional axes, invalid split rejection, schema round trips, and 2/3/4-rank
correctness.

The benchmark report uses the maximum rank latency for each synchronized iteration, then reports
mean, p50, p95, payload bandwidth, and logical sent bytes. Payload bandwidth is
`payload_bytes / mean_max_rank_latency`; it is not a claim about raw link bandwidth.

`dlir collectives bench` is a physical Docker/TCP workload, not the offline in-memory checker.
The host creates one constrained container per rank, each rank joins the protocol-v2 full mesh,
and every warmup and measured iteration starts behind a reusable TCP barrier. Rank-local samples
and traces are emitted as JSON, validated and aggregated in rank order, then the labelled
containers and private network are removed. The in-memory benchmark function remains available to
tests and library callers, but it is not the CLI benchmark path.
