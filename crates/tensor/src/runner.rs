//! Rank-local distributed greedy generation over a native collective communicator.

use crate::{
    NoopTensorEventSink, Result, TensorEventSink, TensorParallelError, TensorParallelManifest,
    TensorParallelRankReport, TensorParallelRankTimings,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::var_builder::ShardedSafeTensors;
use dlir_collectives::{
    BarrierTransport, CollectiveDescriptor, CollectiveObserver, CollectiveTrace, ControlTransport,
    MessageTag, NativeCollectives, PeerInfo, Transport,
};
use dlir_pipeline::{PipelineEvent, ResourceSnapshot};
use dlir_runtime::{
    CollectiveKind as RuntimeCollectiveKind, ControlPurpose, ExecutionPhase, RunEvent,
    RunEventKind, StopReason, TensorParallelKvCache, TensorParallelLlama, TensorParallelObserver,
};
use serde::{Deserialize, Serialize};
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};
use tokenizers::Tokenizer;

const DECISION_TAG_BASE: u64 = 0x3000_0000_0000_0000;

#[derive(Debug, Serialize, Deserialize)]
struct TokenDecision {
    step: usize,
    token: u32,
    emit: bool,
    continue_generation: bool,
    stop_reason: Option<StopReason>,
}

struct EventState {
    next_sequence: u64,
    events: Vec<PipelineEvent>,
}

#[derive(Clone)]
struct EventEmitter {
    started: Instant,
    rank: usize,
    run_id: String,
    request_id: String,
    sink: Arc<dyn TensorEventSink>,
    state: Arc<Mutex<EventState>>,
}

impl EventEmitter {
    fn new(rank: usize, manifest: &TensorParallelManifest, sink: Arc<dyn TensorEventSink>) -> Self {
        Self {
            started: Instant::now(),
            rank,
            run_id: manifest.run_id.clone(),
            request_id: manifest.request_id.clone(),
            sink,
            state: Arc::new(Mutex::new(EventState {
                next_sequence: 0,
                events: Vec::new(),
            })),
        }
    }

    fn emit(&self, kind: RunEventKind) {
        let mut state = self.state.lock().expect("tensor event mutex poisoned");
        let event = PipelineEvent {
            schema_version: 1,
            run_id: self.run_id.clone(),
            request_id: self.request_id.clone(),
            event: RunEvent {
                sequence: state.next_sequence,
                rank: self.rank,
                elapsed_ns: ns(self.started.elapsed()),
                event: kind,
            },
        };
        state.next_sequence += 1;
        self.sink.publish(&event);
        state.events.push(event);
    }

    fn events(&self) -> Vec<PipelineEvent> {
        self.state
            .lock()
            .expect("tensor event mutex poisoned")
            .events
            .clone()
    }
}

struct CollectiveEventObserver(EventEmitter);

impl CollectiveObserver for CollectiveEventObserver {
    fn collective_started(&mut self, descriptor: &CollectiveDescriptor) {
        self.0.emit(RunEventKind::TensorCollectiveStarted {
            collective: format!("{:?}", descriptor.kind).to_ascii_lowercase(),
            algorithm: format!("{:?}", descriptor.algorithm).to_ascii_lowercase(),
            collective_sequence: descriptor.sequence,
            shape: descriptor.input_shape.clone(),
        });
    }

    fn collective_completed(&mut self, trace: &CollectiveTrace) {
        self.0.emit(RunEventKind::TensorCollectiveCompleted {
            collective: format!("{:?}", trace.descriptor.kind).to_ascii_lowercase(),
            collective_sequence: trace.descriptor.sequence,
            sent_bytes: trace.sent_bytes,
            received_bytes: trace.received_bytes,
            duration_ns: trace.duration_ns,
        });
    }
}

struct LayerEventObserver {
    emitter: EventEmitter,
    phase: ExecutionPhase,
    step: usize,
}

impl TensorParallelObserver for LayerEventObserver {
    fn layer_started(&mut self, layer: usize) {
        self.emitter.emit(RunEventKind::LayerStarted {
            layer,
            phase: self.phase,
            step: self.step,
        });
    }

    fn layer_completed(&mut self, layer: usize, duration: std::time::Duration) {
        self.emitter.emit(RunEventKind::LayerCompleted {
            layer,
            phase: self.phase,
            step: self.step,
            duration_ns: ns(duration),
        });
    }
}

/// Executes one tensor rank after its TCP world and cgroup checks are established.
pub fn run_tensor_parallel_rank<T>(
    transport: T,
    manifest: &TensorParallelManifest,
    peers: Vec<PeerInfo>,
) -> Result<TensorParallelRankReport>
where
    T: Transport + ControlTransport + BarrierTransport,
{
    run_tensor_parallel_rank_observed(
        transport,
        manifest,
        peers,
        ResourceSnapshot::default(),
        Arc::new(NoopTensorEventSink),
    )
}

/// Executes one tensor rank while publishing the shared distributed event stream live.
pub fn run_tensor_parallel_rank_observed<T>(
    transport: T,
    manifest: &TensorParallelManifest,
    peers: Vec<PeerInfo>,
    resources: ResourceSnapshot,
    sink: Arc<dyn TensorEventSink>,
) -> Result<TensorParallelRankReport>
where
    T: Transport + ControlTransport + BarrierTransport,
{
    let total_started = Instant::now();
    if manifest.schema_version != 1 || manifest.dtype != dlir_runtime::PlanDType::F32 {
        return Err(TensorParallelError::InvalidRequest(
            "unsupported manifest schema or dtype".into(),
        ));
    }
    let rank = transport.rank().global_rank();
    if transport.rank().world_size() != manifest.tensor_parallel
        || manifest.partition.tensor_parallel != manifest.tensor_parallel
    {
        return Err(TensorParallelError::InvalidRequest(
            "manifest and transport topology disagree".into(),
        ));
    }
    let memory = manifest.partition.ranks.get(rank).cloned().ok_or_else(|| {
        TensorParallelError::InvalidRequest(format!("manifest has no rank {rank} plan"))
    })?;
    let spec = manifest.model.spec();
    let emitter = EventEmitter::new(rank, manifest, sink);
    emitter.emit(RunEventKind::MemorySample {
        current_bytes: resources.memory_current_bytes,
        limit_bytes: resources.memory_limit_bytes,
    });

    let load_started = Instant::now();
    emitter.emit(RunEventKind::ModelLoadStarted);
    // SAFETY: the launcher bind-mounts an immutable checkpoint for the complete rank lifetime.
    let vb = unsafe {
        ShardedSafeTensors::var_builder(&[&manifest.checkpoint_path], DType::F32, &Device::Cpu)
    }?;
    let parallel = dlir_runtime::ParallelContext::tensor_parallel(rank, manifest.tensor_parallel)?;
    let model = TensorParallelLlama::load_sharded(
        vb,
        &spec.config,
        parallel,
        manifest.context_capacity,
        manifest.all_reduce,
    )?;
    let mut cache = TensorParallelKvCache::new_tensor_parallel(
        &spec.config,
        manifest.tensor_parallel,
        manifest.context_capacity,
        DType::F32,
        &Device::Cpu,
    )?;
    let model_load_ns = ns(load_started.elapsed());
    emitter.emit(RunEventKind::ModelLoadFinished);
    let mut collectives = NativeCollectives::with_observer(
        dlir_collectives::Communicator::new(transport),
        CollectiveEventObserver(emitter.clone()),
    );
    emitter.emit(RunEventKind::CollectiveStarted {
        collective: RuntimeCollectiveKind::Barrier,
        generation: 0,
    });
    let barrier_started = Instant::now();
    collectives.barrier()?;
    emitter.emit(RunEventKind::CollectiveCompleted {
        collective: RuntimeCollectiveKind::Barrier,
        generation: 0,
        duration_ns: ns(barrier_started.elapsed()),
    });

    let prompt = Tensor::from_vec(
        manifest.prompt_token_ids.clone(),
        (1, manifest.prompt_token_ids.len()),
        &Device::Cpu,
    )?;
    let tokenizer = if rank == 0 {
        Some(
            Tokenizer::from_file(&manifest.tokenizer_path)
                .map_err(|error| TensorParallelError::Tokenizer(error.to_string()))?,
        )
    } else {
        None
    };
    let mut input = prompt;
    let mut position = 0usize;
    let mut generated_tokens = Vec::new();
    let mut prefill_ns = 0;
    let mut decode_total_ns = 0;
    let mut decode_forward_count = 0;
    let mut stop_reason = None;

    for step in 0..manifest.effective_max_new_tokens {
        let phase = if step == 0 {
            emitter.emit(RunEventKind::PrefillStarted {
                prompt_tokens: manifest.prompt_token_ids.len(),
            });
            ExecutionPhase::Prefill
        } else {
            emitter.emit(RunEventKind::DecodeStepStarted { step });
            ExecutionPhase::Decode
        };
        let started = Instant::now();
        let mut layer_observer = LayerEventObserver {
            emitter: emitter.clone(),
            phase,
            step,
        };
        let logits = model.forward_observed(
            &input,
            position,
            &mut cache,
            &mut collectives,
            &mut layer_observer,
        )?;
        let elapsed = ns(started.elapsed());
        if step == 0 {
            prefill_ns = elapsed;
        } else {
            decode_total_ns += elapsed;
            decode_forward_count += 1;
        }
        let decision_tag = MessageTag(DECISION_TAG_BASE + step as u64);
        let decision = if rank == 0 {
            let token = logits.argmax(candle_core::D::Minus1)?.to_vec1::<u32>()?[0];
            let emit = token != spec.config.eos_token_id;
            let at_limit = step + 1 == manifest.effective_max_new_tokens;
            let reason = if !emit {
                Some(StopReason::Eos)
            } else if at_limit
                && manifest.effective_max_new_tokens < manifest.requested_max_new_tokens
            {
                Some(StopReason::ContextLimit)
            } else if at_limit {
                Some(StopReason::MaxNewTokens)
            } else {
                None
            };
            let decision = TokenDecision {
                step,
                token,
                emit,
                continue_generation: reason.is_none(),
                stop_reason: reason,
            };
            let bytes = serde_json::to_vec(&decision)?;
            for peer in 1..manifest.tensor_parallel {
                let control_started = Instant::now();
                collectives.send_control(peer, decision_tag, bytes.clone())?;
                emitter.emit(RunEventKind::ControlSent {
                    peer,
                    purpose: ControlPurpose::TensorDecision,
                    phase,
                    step,
                    bytes: bytes.len() as u64,
                    duration_ns: ns(control_started.elapsed()),
                });
            }
            decision
        } else {
            let control_started = Instant::now();
            let bytes = collectives.recv_control(0, decision_tag)?;
            emitter.emit(RunEventKind::ControlReceived {
                peer: 0,
                purpose: ControlPurpose::TensorDecision,
                phase,
                step,
                bytes: bytes.len() as u64,
                duration_ns: ns(control_started.elapsed()),
            });
            serde_json::from_slice::<TokenDecision>(&bytes)?
        };
        if decision.step != step {
            return Err(TensorParallelError::InvalidRequest(
                "stale token decision step".into(),
            ));
        }
        if decision.emit {
            generated_tokens.push(decision.token);
            if rank == 0 {
                emitter.emit(RunEventKind::TokenGenerated {
                    token_id: decision.token,
                    text: String::new(),
                });
            }
        }
        if step == 0 {
            emitter.emit(RunEventKind::PrefillFinished);
        } else {
            emitter.emit(RunEventKind::DecodeStepFinished { step });
        }
        if !decision.continue_generation {
            stop_reason = Some(decision.stop_reason.ok_or_else(|| {
                TensorParallelError::InvalidRequest("terminal decision omitted stop reason".into())
            })?);
            break;
        }
        position += input.dim(1)?;
        input = Tensor::new(&[[decision.token]], &Device::Cpu)?;
    }
    let stop_reason = stop_reason.ok_or_else(|| {
        TensorParallelError::InvalidRequest("generation ended without a stop reason".into())
    })?;
    emitter.emit(RunEventKind::GenerationFinished { stop_reason });
    emitter.emit(RunEventKind::CollectiveStarted {
        collective: RuntimeCollectiveKind::Barrier,
        generation: 1,
    });
    let barrier_started = Instant::now();
    collectives.barrier()?;
    emitter.emit(RunEventKind::CollectiveCompleted {
        collective: RuntimeCollectiveKind::Barrier,
        generation: 1,
        duration_ns: ns(barrier_started.elapsed()),
    });
    let traces = collectives.take_traces();
    let sent_bytes = traces.iter().map(|trace| trace.sent_bytes).sum();
    let received_bytes = traces.iter().map(|trace| trace.received_bytes).sum();
    let completion = if let Some(tokenizer) = tokenizer {
        tokenizer
            .decode(&generated_tokens, true)
            .map_err(|error| TensorParallelError::Tokenizer(error.to_string()))?
    } else {
        String::new()
    };
    let final_kv_cache_bytes = cache.used_bytes(
        &{
            let mut local = spec.config;
            local.hidden_size /= manifest.tensor_parallel;
            local.num_attention_heads /= manifest.tensor_parallel;
            local.num_key_value_heads /= manifest.tensor_parallel;
            local
        },
        DType::F32,
    )?;
    Ok(TensorParallelRankReport {
        schema_version: 1,
        run_id: manifest.run_id.clone(),
        request_id: manifest.request_id.clone(),
        rank,
        peers,
        memory,
        resources,
        final_kv_cache_bytes,
        collectives: traces,
        sent_bytes,
        received_bytes,
        generated_tokens,
        completion,
        stop_reason,
        timings: TensorParallelRankTimings {
            model_load_ns,
            prefill_ns,
            decode_total_ns,
            decode_forward_count,
            total_ns: ns(total_started.elapsed()),
        },
        events: emitter.events(),
        barriers_passed: true,
        success: true,
    })
}

fn ns(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}
