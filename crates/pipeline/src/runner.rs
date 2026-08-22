//! Transport-generic correctness-first pipeline rank state machine.

use crate::{
    CommunicationReport, EventRecorder, PipelineControl, PipelineDecision, PipelineError,
    PipelineEventSink, PipelineManifest, PipelineRankReport, PipelineRankTimings, ResourceSnapshot,
    Result, activation_tag, decision_tag, token_tag,
};
use candle_core::{D, Device, Tensor};
use candle_nn::VarBuilder;
use dlir_collectives::{BarrierTransport, Communicator, ControlTransport, PeerInfo, Transport};
use dlir_runtime::{
    CollectiveKind, ControlPurpose, ExecutionPhase, LayerObserver, LlamaStage, PlacementVerdict,
    RunEventKind, StageKvCache, StopReason, TensorPurpose,
};
use std::{path::Path, time::Instant};
use tokenizers::Tokenizer;

/// Executes one assigned rank using an established point-to-point world.
pub fn run_pipeline_rank<T>(
    transport: T,
    manifest: &PipelineManifest,
    resources: ResourceSnapshot,
    peers: Vec<PeerInfo>,
    sink: &mut dyn PipelineEventSink,
) -> Result<PipelineRankReport>
where
    T: Transport + ControlTransport + BarrierTransport,
{
    let total_started = Instant::now();
    let communicator = Communicator::new(transport);
    let rank = communicator.rank();
    if rank.world_size() != manifest.partition.stages.len() {
        return Err(PipelineError::InvalidTopology(format!(
            "rank world size {} differs from manifest stages {}",
            rank.world_size(),
            manifest.partition.stages.len()
        )));
    }
    let assignment = manifest.partition.stage(rank.global_rank())?.clone();
    let memory = manifest
        .memory_plans
        .get(rank.global_rank())
        .ok_or_else(|| PipelineError::InvalidTopology("missing rank memory plan".into()))?
        .clone();
    if memory.placement == PlacementVerdict::DoesNotFit {
        return Err(PipelineError::PlacementFailed {
            rank: rank.global_rank(),
            required_bytes: memory.persistent_bytes,
            budget_bytes: memory.budget_bytes,
        });
    }
    let cpu_mismatch = manifest.expected_cpu_millis > 0
        && resources.cpu_millis != Some(manifest.expected_cpu_millis);
    let memory_mismatch = manifest.expected_memory_bytes > 0
        && resources.memory_limit_bytes != Some(manifest.expected_memory_bytes);
    if cpu_mismatch || memory_mismatch {
        return Err(PipelineError::InvalidTopology(format!(
            "rank {} observed CPU/memory {:?}/{:?}, expected {}/{}",
            rank.global_rank(),
            resources.cpu_millis,
            resources.memory_limit_bytes,
            manifest.expected_cpu_millis,
            manifest.expected_memory_bytes
        )));
    }
    let spec = manifest.model.spec();
    spec.validate_cpu_dtype(manifest.dtype)?;
    let mut recorder = EventRecorder::new(
        rank.global_rank(),
        &manifest.run_id,
        &manifest.request_id,
        sink,
    );
    recorder.emit(RunEventKind::MemorySample {
        current_bytes: resources.memory_current_bytes,
        limit_bytes: resources.memory_limit_bytes,
    });
    recorder.emit(RunEventKind::ModelLoadStarted);
    let model_load_started = Instant::now();
    let device = Device::Cpu;
    // SAFETY: the immutable bind-mounted checkpoint remains available for the container lifetime;
    // the stage owns every tensor materialized through the builder.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[Path::new(&manifest.checkpoint_path)],
            manifest.dtype.candle(),
            &device,
        )
    }?;
    let stage = LlamaStage::load(
        vb,
        &spec.config,
        assignment.layer_start,
        assignment.layer_end,
        assignment.owns_embeddings,
        assignment.owns_lm_head,
        manifest.context_capacity,
    )?;
    let mut cache = StageKvCache::new_for_layers(
        &spec.config,
        assignment.layer_count(),
        manifest.context_capacity,
        manifest.dtype.candle(),
        &device,
    )?;
    device.synchronize()?;
    let model_load_ns = ns(model_load_started.elapsed());
    recorder.emit(RunEventKind::ModelLoadFinished);
    recorder.emit(RunEventKind::MemorySample {
        current_bytes: resources.memory_current_bytes,
        limit_bytes: resources.memory_limit_bytes,
    });

    let mut communication = CommunicationReport::default();
    observed_barrier(&communicator, &mut recorder, 0)?;

    let mut generated_tokens = Vec::new();
    let mut tokenizer = if assignment.rank == 0 {
        Some(
            Tokenizer::from_file(&manifest.tokenizer_path)
                .map_err(|error| PipelineError::Tokenizer(error.to_string()))?,
        )
    } else {
        None
    };
    let mut decode_stream = tokenizer
        .as_ref()
        .map(|tokenizer| tokenizer.decode_stream(true));
    let mut completion = String::new();
    let mut prefill_ns = 0;
    let mut decode_total_ns = 0;
    let mut decode_forward_count = 0;
    let mut time_to_first_token_ns = None;
    let mut step = 0usize;
    let mut previous_token = None;
    let stop_reason;

    loop {
        let phase = if step == 0 {
            ExecutionPhase::Prefill
        } else {
            ExecutionPhase::Decode
        };
        let phase_started = Instant::now();
        match phase {
            ExecutionPhase::Prefill => recorder.emit(RunEventKind::PrefillStarted {
                prompt_tokens: manifest.prompt_token_ids.len(),
            }),
            ExecutionPhase::Decode => {
                recorder.emit(RunEventKind::DecodeStepStarted { step });
                decode_forward_count += 1;
            }
        }

        let position = cache.len();
        let hidden = if assignment.rank == 0 {
            let input = if step == 0 {
                Tensor::new(manifest.prompt_token_ids.as_slice(), &device)?.unsqueeze(0)?
            } else {
                Tensor::new(
                    &[previous_token.ok_or_else(|| {
                        PipelineError::Protocol("decode has no previous token".into())
                    })?],
                    &device,
                )?
                .unsqueeze(0)?
            };
            device.synchronize()?;
            let mut layer_observer = RankLayerObserver {
                recorder: &mut recorder,
                phase,
                step,
            };
            stage.forward_tokens(&input, position, &mut cache, &mut layer_observer)?
        } else {
            let hidden = recv_activation(
                &communicator,
                assignment.rank - 1,
                phase,
                step,
                &mut communication,
                &mut recorder,
            )?;
            validate_activation(&hidden, spec.config.hidden_size, phase)?;
            device.synchronize()?;
            let mut layer_observer = RankLayerObserver {
                recorder: &mut recorder,
                phase,
                step,
            };
            stage.forward_hidden(&hidden, position, &mut cache, &mut layer_observer)?
        };
        device.synchronize()?;

        if assignment.rank + 1 < assignment.world_size {
            send_activation(
                &communicator,
                assignment.rank + 1,
                &hidden,
                phase,
                step,
                &mut communication,
                &mut recorder,
            )?;
        } else {
            let logits = stage.finish(&hidden)?;
            let token_id = logits.squeeze(0)?.argmax(D::Minus1)?.to_scalar::<u32>()?;
            send_control(
                &communicator,
                0,
                token_tag(step)?,
                &PipelineControl::Token { step, token_id },
                ControlPurpose::TokenFeedback,
                phase,
                step,
                &mut communication,
                &mut recorder,
            )?;
        }

        let decision = if assignment.rank == 0 {
            let control = recv_control(
                &communicator,
                assignment.world_size - 1,
                token_tag(step)?,
                ControlPurpose::TokenFeedback,
                phase,
                step,
                &mut communication,
                &mut recorder,
            )?;
            let PipelineControl::Token {
                step: actual_step,
                token_id,
            } = control
            else {
                return Err(PipelineError::Protocol(
                    "rank 0 expected final-stage token feedback".into(),
                ));
            };
            if actual_step != step {
                return Err(PipelineError::Protocol(format!(
                    "received token for step {actual_step}, expected {step}"
                )));
            }
            let decision = if token_id == spec.config.eos_token_id {
                PipelineDecision::Stop {
                    reason: StopReason::Eos,
                }
            } else {
                generated_tokens.push(token_id);
                let text = decode_stream
                    .as_mut()
                    .expect("rank 0 has decoder")
                    .step(token_id)
                    .map_err(|error| PipelineError::Tokenizer(error.to_string()))?
                    .unwrap_or_default();
                recorder.emit(RunEventKind::TokenGenerated { token_id, text });
                previous_token = Some(token_id);
                if generated_tokens.len() >= manifest.effective_max_new_tokens {
                    PipelineDecision::Stop {
                        reason: if manifest.requested_max_new_tokens
                            > manifest.effective_max_new_tokens
                        {
                            StopReason::ContextLimit
                        } else {
                            StopReason::MaxNewTokens
                        },
                    }
                } else {
                    PipelineDecision::Continue
                }
            };
            let control = PipelineControl::Decision { step, decision };
            for destination in 1..assignment.world_size {
                send_control(
                    &communicator,
                    destination,
                    decision_tag(step)?,
                    &control,
                    ControlPurpose::Decision,
                    phase,
                    step,
                    &mut communication,
                    &mut recorder,
                )?;
            }
            decision
        } else {
            let control = recv_control(
                &communicator,
                0,
                decision_tag(step)?,
                ControlPurpose::Decision,
                phase,
                step,
                &mut communication,
                &mut recorder,
            )?;
            let PipelineControl::Decision {
                step: actual_step,
                decision,
            } = control
            else {
                return Err(PipelineError::Protocol(
                    "stage expected rank-0 decision".into(),
                ));
            };
            if actual_step != step {
                return Err(PipelineError::Protocol(format!(
                    "received decision for step {actual_step}, expected {step}"
                )));
            }
            decision
        };

        let phase_ns = ns(phase_started.elapsed());
        match phase {
            ExecutionPhase::Prefill => {
                prefill_ns = phase_ns;
                if assignment.rank == 0 {
                    time_to_first_token_ns = Some(phase_ns);
                }
                recorder.emit(RunEventKind::PrefillFinished);
            }
            ExecutionPhase::Decode => {
                decode_total_ns += phase_ns;
                recorder.emit(RunEventKind::DecodeStepFinished { step });
            }
        }
        recorder.emit(RunEventKind::MemorySample {
            current_bytes: resources.memory_current_bytes,
            limit_bytes: resources.memory_limit_bytes,
        });
        match decision {
            PipelineDecision::Continue => step += 1,
            PipelineDecision::Stop { reason } => {
                stop_reason = reason;
                break;
            }
        }
    }

    if assignment.rank == 0 {
        completion = tokenizer
            .take()
            .expect("rank 0 has tokenizer")
            .decode(&generated_tokens, true)
            .map_err(|error| PipelineError::Tokenizer(error.to_string()))?;
    }
    recorder.emit(RunEventKind::GenerationFinished { stop_reason });
    observed_barrier(&communicator, &mut recorder, 1)?;
    let final_kv_cache_bytes = cache.used_bytes(&spec.config, manifest.dtype.candle())?;
    let layer_compute_ns = recorder
        .events
        .iter()
        .filter_map(|event| match event.event.event {
            RunEventKind::LayerCompleted { duration_ns, .. } => Some(duration_ns),
            _ => None,
        })
        .fold(0u64, u64::saturating_add);
    let total_ns = ns(total_started.elapsed());
    Ok(PipelineRankReport {
        schema_version: 1,
        run_id: manifest.run_id.clone(),
        request_id: manifest.request_id.clone(),
        rank: assignment.rank,
        peers,
        assignment,
        memory,
        resources,
        final_kv_cache_bytes,
        communication,
        timings: PipelineRankTimings {
            model_load_ns,
            prefill_ns,
            decode_total_ns,
            layer_compute_ns,
            decode_forward_count,
            time_to_first_token_ns,
            total_ns,
        },
        generated_tokens,
        completion,
        stop_reason,
        barriers_passed: true,
        events: recorder.events,
        success: true,
    })
}

struct RankLayerObserver<'a, 'b> {
    recorder: &'a mut EventRecorder<'b>,
    phase: ExecutionPhase,
    step: usize,
}

impl LayerObserver for RankLayerObserver<'_, '_> {
    fn layer_started(&mut self, layer: usize) {
        self.recorder.emit(RunEventKind::LayerStarted {
            layer,
            phase: self.phase,
            step: self.step,
        });
    }

    fn layer_completed(&mut self, layer: usize, duration: std::time::Duration) {
        self.recorder.emit(RunEventKind::LayerCompleted {
            layer,
            phase: self.phase,
            step: self.step,
            duration_ns: ns(duration),
        });
    }
}

fn observed_barrier<T: BarrierTransport>(
    communicator: &Communicator<T>,
    recorder: &mut EventRecorder<'_>,
    generation: u64,
) -> Result<()> {
    recorder.emit(RunEventKind::CollectiveStarted {
        collective: CollectiveKind::Barrier,
        generation,
    });
    let started = Instant::now();
    communicator.barrier()?;
    recorder.emit(RunEventKind::CollectiveCompleted {
        collective: CollectiveKind::Barrier,
        generation,
        duration_ns: ns(started.elapsed()),
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn send_activation<T: Transport>(
    communicator: &Communicator<T>,
    destination: usize,
    tensor: &Tensor,
    phase: ExecutionPhase,
    step: usize,
    communication: &mut CommunicationReport,
    recorder: &mut EventRecorder<'_>,
) -> Result<()> {
    let shape = tensor.dims().to_vec();
    let bytes = tensor_bytes(tensor)?;
    let started = Instant::now();
    communicator.send_tensor(destination, activation_tag(step)?, tensor)?;
    let duration_ns = ns(started.elapsed());
    communication.tensor_messages_sent += 1;
    communication.tensor_bytes_sent += bytes;
    communication.communication_ns += duration_ns;
    recorder.emit(RunEventKind::TensorSent {
        peer: destination,
        purpose: TensorPurpose::Activation,
        phase,
        step,
        shape,
        bytes,
        duration_ns,
    });
    Ok(())
}

fn recv_activation<T: Transport>(
    communicator: &Communicator<T>,
    source: usize,
    phase: ExecutionPhase,
    step: usize,
    communication: &mut CommunicationReport,
    recorder: &mut EventRecorder<'_>,
) -> Result<Tensor> {
    let started = Instant::now();
    let tensor = communicator.recv_tensor(source, activation_tag(step)?)?;
    let duration_ns = ns(started.elapsed());
    let shape = tensor.dims().to_vec();
    let bytes = tensor_bytes(&tensor)?;
    communication.tensor_messages_received += 1;
    communication.tensor_bytes_received += bytes;
    communication.communication_ns += duration_ns;
    recorder.emit(RunEventKind::TensorReceived {
        peer: source,
        purpose: TensorPurpose::Activation,
        phase,
        step,
        shape,
        bytes,
        duration_ns,
    });
    Ok(tensor)
}

#[allow(clippy::too_many_arguments)]
fn send_control<T: ControlTransport>(
    communicator: &Communicator<T>,
    destination: usize,
    tag: dlir_collectives::MessageTag,
    control: &PipelineControl,
    purpose: ControlPurpose,
    phase: ExecutionPhase,
    step: usize,
    communication: &mut CommunicationReport,
    recorder: &mut EventRecorder<'_>,
) -> Result<()> {
    let bytes = control.encode()?;
    let length = bytes.len() as u64;
    let started = Instant::now();
    communicator.send_control(destination, tag, bytes)?;
    communication.control_messages_sent += 1;
    communication.control_bytes_sent += length;
    let duration_ns = ns(started.elapsed());
    communication.communication_ns += duration_ns;
    recorder.emit(RunEventKind::ControlSent {
        peer: destination,
        purpose,
        phase,
        step,
        bytes: length,
        duration_ns,
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn recv_control<T: ControlTransport>(
    communicator: &Communicator<T>,
    source: usize,
    tag: dlir_collectives::MessageTag,
    purpose: ControlPurpose,
    phase: ExecutionPhase,
    step: usize,
    communication: &mut CommunicationReport,
    recorder: &mut EventRecorder<'_>,
) -> Result<PipelineControl> {
    let started = Instant::now();
    let bytes = communicator.recv_control(source, tag)?;
    let length = bytes.len() as u64;
    communication.control_messages_received += 1;
    communication.control_bytes_received += length;
    let control = PipelineControl::decode(&bytes)?;
    let duration_ns = ns(started.elapsed());
    communication.communication_ns += duration_ns;
    recorder.emit(RunEventKind::ControlReceived {
        peer: source,
        purpose,
        phase,
        step,
        bytes: length,
        duration_ns,
    });
    Ok(control)
}

fn validate_activation(tensor: &Tensor, hidden_size: usize, phase: ExecutionPhase) -> Result<()> {
    let (batch, sequence, hidden) = tensor.dims3()?;
    let expected_sequence = match phase {
        ExecutionPhase::Prefill => None,
        ExecutionPhase::Decode => Some(1),
    };
    if batch != 1 || hidden != hidden_size || expected_sequence.is_some_and(|s| sequence != s) {
        return Err(PipelineError::Protocol(format!(
            "invalid {phase:?} activation shape {:?}; expected [1,S,{hidden_size}]",
            tensor.dims()
        )));
    }
    Ok(())
}

fn tensor_bytes(tensor: &Tensor) -> Result<u64> {
    let elements = tensor
        .dims()
        .iter()
        .try_fold(1u64, |count, dimension| {
            count.checked_mul(*dimension as u64)
        })
        .ok_or_else(|| PipelineError::Protocol("tensor byte count overflow".into()))?;
    elements
        .checked_mul(tensor.dtype().size_in_bytes() as u64)
        .ok_or_else(|| PipelineError::Protocol("tensor byte count overflow".into()))
}

fn ns(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}
