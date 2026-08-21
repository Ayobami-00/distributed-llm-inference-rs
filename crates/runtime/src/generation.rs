use crate::{
    DlirError, MemoryBudget, PlanDType, RankMemoryPlan, Result, RunEvent, RunEventKind, StopReason,
    SupportedModelId, TimingReport, TopologyReport,
    artifacts::{ArtifactRepository, validate_checkpoint, validate_metadata},
    model::{KvCache, Llama},
    prompt::render_prompt,
    report::GenerationReport,
};
use candle_core::{D, Device, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub model: SupportedModelId,
    pub dtype: PlanDType,
    pub prompt: String,
    pub max_new_tokens: usize,
    pub device_memory_budget: Option<MemoryBudget>,
}

pub trait EventObserver {
    fn on_event(&mut self, event: &RunEvent);
}

#[derive(Default)]
pub struct NoopObserver;

impl EventObserver for NoopObserver {
    fn on_event(&mut self, _event: &RunEvent) {}
}

struct EventRecorder<'a> {
    start: Instant,
    sequence: u64,
    events: Vec<RunEvent>,
    observer: &'a mut dyn EventObserver,
}

impl<'a> EventRecorder<'a> {
    fn new(start: Instant, observer: &'a mut dyn EventObserver) -> Self {
        Self {
            start,
            sequence: 0,
            events: Vec::new(),
            observer,
        }
    }

    fn emit(&mut self, kind: RunEventKind) {
        let event = RunEvent {
            sequence: self.sequence,
            rank: 0,
            elapsed_ns: duration_ns(self.start.elapsed()),
            event: kind,
        };
        self.sequence += 1;
        self.observer.on_event(&event);
        self.events.push(event);
    }
}

pub fn generate(
    request: &GenerationRequest,
    observer: &mut dyn EventObserver,
) -> Result<GenerationReport> {
    let cold_start = Instant::now();
    if request.max_new_tokens == 0 {
        return Err(DlirError::InvalidConfig(
            "max_new_tokens must be at least one".into(),
        ));
    }
    if request.prompt.trim().is_empty() {
        return Err(DlirError::EmptyPrompt);
    }
    let spec = request.model.spec();
    spec.validate_cpu_dtype(request.dtype)?;
    spec.config.head_dim()?;
    let mut recorder = EventRecorder::new(cold_start, observer);

    recorder.emit(RunEventKind::ArtifactResolutionStarted);
    let artifact_start = Instant::now();
    let repository = ArtifactRepository::new(spec)?;
    let metadata = repository.download_metadata()?;
    validate_metadata(spec, &metadata)?;
    let artifact_metadata_time = artifact_start.elapsed();
    recorder.emit(RunEventKind::ArtifactResolutionFinished);

    let tokenizer = Tokenizer::from_file(&metadata.tokenizer)
        .map_err(|err| DlirError::Tokenizer(err.to_string()))?;
    let tokenization_start = Instant::now();
    let rendered_prompt = render_prompt(spec.prompt_template, &request.prompt)?;
    let prompt_encoding = tokenizer
        .encode(rendered_prompt, false)
        .map_err(|err| DlirError::Tokenizer(err.to_string()))?;
    let prompt_tokens = prompt_encoding.get_ids().to_vec();
    let tokenization_time = tokenization_start.elapsed();
    if prompt_tokens.is_empty() {
        return Err(DlirError::EmptyPrompt);
    }
    let (effective_max_new_tokens, memory) = generation_memory_preflight(
        spec,
        request.dtype,
        prompt_tokens.len(),
        request.max_new_tokens,
        request.device_memory_budget,
    )?;
    let capacity = memory.context_length;
    if memory.placement == crate::PlacementVerdict::DoesNotFit {
        let budget = memory.budget.expect("failed placement always has a budget");
        return Err(DlirError::PlacementFailed {
            required_bytes: memory.persistent_bytes,
            budget_bytes: budget.bytes,
        });
    }

    recorder.emit(RunEventKind::ArtifactResolutionStarted);
    let weight_download_start = Instant::now();
    let weights = repository.download_weights(spec)?;
    validate_checkpoint(spec, &weights)?;
    let weight_download_time = weight_download_start.elapsed();
    recorder.emit(RunEventKind::ArtifactResolutionFinished);

    let device = Device::Cpu;
    recorder.emit(RunEventKind::ModelLoadStarted);
    let model_load_start = Instant::now();
    // SAFETY: the immutable Hub cache file remains at `weights` while the returned model owns
    // all tensors materialized through this VarBuilder.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[&weights], request.dtype.candle(), &device)
    }?;
    let model = Llama::load(vb, &spec.config, capacity)?;
    let mut cache = KvCache::new(&spec.config, capacity, request.dtype.candle(), &device)?;
    device.synchronize()?;
    let model_load_time = model_load_start.elapsed();
    recorder.emit(RunEventKind::ModelLoadFinished);

    let generation_start = Instant::now();
    recorder.emit(RunEventKind::PrefillStarted {
        prompt_tokens: prompt_tokens.len(),
    });
    device.synchronize()?;
    let prefill_start = Instant::now();
    let input = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
    let mut logits = model.forward(&input, 0, &mut cache)?;
    device.synchronize()?;
    let prefill_time = prefill_start.elapsed();
    recorder.emit(RunEventKind::PrefillFinished);

    let mut generated_tokens = Vec::new();
    let mut decode_total = Duration::ZERO;
    let mut decode_forward_count = 0usize;
    let first_token_start = generation_start;
    let mut next_token = greedy_token(&logits)?;
    let mut decode_stream = tokenizer.decode_stream(true);

    if next_token != spec.config.eos_token_id {
        let text = decode_stream
            .step(next_token)
            .map_err(|err| DlirError::Tokenizer(err.to_string()))?
            .unwrap_or_default();
        emit_token(next_token, text, &mut generated_tokens, &mut recorder);
        let time_to_first_token = tokenization_time + first_token_start.elapsed();

        let stop_reason = loop {
            if generated_tokens.len() >= effective_max_new_tokens {
                break token_limit_stop_reason(request.max_new_tokens, effective_max_new_tokens);
            }

            let step = decode_forward_count + 1;
            recorder.emit(RunEventKind::DecodeStepStarted { step });
            device.synchronize()?;
            let decode_start = Instant::now();
            let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            let position = cache.len();
            logits = model.forward(&input, position, &mut cache)?;
            device.synchronize()?;
            decode_total += decode_start.elapsed();
            decode_forward_count += 1;
            recorder.emit(RunEventKind::DecodeStepFinished { step });

            next_token = greedy_token(&logits)?;
            if next_token == spec.config.eos_token_id {
                break StopReason::Eos;
            }
            let text = decode_stream
                .step(next_token)
                .map_err(|err| DlirError::Tokenizer(err.to_string()))?
                .unwrap_or_default();
            emit_token(next_token, text, &mut generated_tokens, &mut recorder);
        };

        recorder.emit(RunEventKind::GenerationFinished { stop_reason });
        let model_generation_time = generation_start.elapsed();
        let generation_time = tokenization_time + model_generation_time;
        let completion = tokenizer
            .decode(&generated_tokens, true)
            .map_err(|err| DlirError::Tokenizer(err.to_string()))?;
        let final_kv_cache_bytes = cache.used_bytes(&spec.config, request.dtype.candle())?;
        let artifact_time = artifact_metadata_time + weight_download_time;
        let timings = build_timings(
            artifact_time,
            model_load_time,
            tokenization_time,
            prefill_time,
            time_to_first_token,
            decode_total,
            decode_forward_count,
            prompt_tokens.len(),
            generation_time,
            cold_start.elapsed(),
        );
        return Ok(GenerationReport {
            schema_version: 1,
            model: request.model,
            repository: spec.repository.into(),
            revision: spec.revision.into(),
            device: "cpu".into(),
            dtype: request.dtype,
            topology: TopologyReport::default(),
            memory,
            final_kv_cache_bytes,
            prompt_characters: request.prompt.chars().count(),
            prompt_tokens: prompt_tokens.len(),
            requested_max_new_tokens: request.max_new_tokens,
            generated_tokens,
            completion,
            stop_reason,
            timings,
            events: recorder.events,
        });
    }

    let stop_reason = StopReason::Eos;
    recorder.emit(RunEventKind::GenerationFinished { stop_reason });
    let completion = String::new();
    let model_generation_time = generation_start.elapsed();
    let generation_time = tokenization_time + model_generation_time;
    let artifact_time = artifact_metadata_time + weight_download_time;
    Ok(GenerationReport {
        schema_version: 1,
        model: request.model,
        repository: spec.repository.into(),
        revision: spec.revision.into(),
        device: "cpu".into(),
        dtype: request.dtype,
        topology: TopologyReport::default(),
        memory,
        final_kv_cache_bytes: cache.used_bytes(&spec.config, request.dtype.candle())?,
        prompt_characters: request.prompt.chars().count(),
        prompt_tokens: prompt_tokens.len(),
        requested_max_new_tokens: request.max_new_tokens,
        generated_tokens,
        completion,
        stop_reason,
        timings: build_timings(
            artifact_time,
            model_load_time,
            tokenization_time,
            prefill_time,
            tokenization_time + model_generation_time,
            decode_total,
            decode_forward_count,
            prompt_tokens.len(),
            generation_time,
            cold_start.elapsed(),
        ),
        events: recorder.events,
    })
}

fn emit_token(
    token: u32,
    text: String,
    generated_tokens: &mut Vec<u32>,
    recorder: &mut EventRecorder<'_>,
) {
    generated_tokens.push(token);
    recorder.emit(RunEventKind::TokenGenerated {
        token_id: token,
        text,
    });
}

fn greedy_token(logits: &Tensor) -> Result<u32> {
    Ok(logits.squeeze(0)?.argmax(D::Minus1)?.to_scalar::<u32>()?)
}

fn generation_memory_preflight(
    spec: &crate::ModelSpec,
    dtype: PlanDType,
    prompt_tokens: usize,
    requested_new_tokens: usize,
    budget: Option<MemoryBudget>,
) -> Result<(usize, RankMemoryPlan)> {
    if prompt_tokens >= spec.config.max_position_embeddings {
        return Err(DlirError::PromptTooLong {
            prompt_tokens,
            max_context: spec.config.max_position_embeddings,
        });
    }
    let available_generation = spec.config.max_position_embeddings - prompt_tokens;
    let effective_new_tokens = requested_new_tokens.min(available_generation);
    let capacity = prompt_tokens + effective_new_tokens;
    let memory = RankMemoryPlan::for_model(spec, dtype, capacity, budget)?;
    Ok((effective_new_tokens, memory))
}

fn token_limit_stop_reason(requested: usize, effective: usize) -> StopReason {
    if requested > effective {
        StopReason::ContextLimit
    } else {
        StopReason::MaxNewTokens
    }
}

#[allow(clippy::too_many_arguments)]
fn build_timings(
    artifact: Duration,
    model_load: Duration,
    tokenization: Duration,
    prefill: Duration,
    ttft: Duration,
    decode: Duration,
    decode_forwards: usize,
    prompt_tokens: usize,
    generation: Duration,
    cold_start: Duration,
) -> TimingReport {
    let decode_seconds = decode.as_secs_f64();
    TimingReport {
        artifact_resolution_ns: duration_ns(artifact),
        model_load_ns: duration_ns(model_load),
        tokenization_ns: duration_ns(tokenization),
        prefill_ns: duration_ns(prefill),
        time_to_first_token_ns: duration_ns(ttft),
        decode_total_ns: duration_ns(decode),
        decode_forward_count: decode_forwards,
        mean_decode_ns: (decode_forwards > 0).then(|| duration_ns(decode) / decode_forwards as u64),
        prefill_tokens_per_second: prompt_tokens as f64 / prefill.as_secs_f64(),
        decode_tokens_per_second: (decode_forwards > 0 && decode_seconds > 0.0)
            .then(|| decode_forwards as f64 / decode_seconds),
        generation_total_ns: duration_ns(generation),
        cold_start_total_ns: duration_ns(cold_start),
    }
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preflight_rejects_context_exhaustion_without_truncation() {
        let spec = SupportedModelId::SmolLm2_135MInstruct.spec();
        assert!(matches!(
            generation_memory_preflight(
                spec,
                PlanDType::F32,
                spec.config.max_position_embeddings,
                1,
                None,
            ),
            Err(DlirError::PromptTooLong { .. })
        ));
    }

    #[test]
    fn preflight_caps_cache_at_context_and_reports_context_limit() {
        let spec = SupportedModelId::SmolLm2_135MInstruct.spec();
        let prompt_tokens = spec.config.max_position_embeddings - 2;
        let (effective, plan) =
            generation_memory_preflight(spec, PlanDType::F32, prompt_tokens, 5, None).unwrap();
        assert_eq!(effective, 2);
        assert_eq!(plan.context_length, spec.config.max_position_embeddings);
        assert_eq!(
            token_limit_stop_reason(5, effective),
            StopReason::ContextLimit
        );
    }

    #[test]
    fn preflight_uses_the_same_exact_placement_boundary_as_inspect() {
        let spec = SupportedModelId::SmolLm2_135MInstruct.spec();
        let (_, without_budget) =
            generation_memory_preflight(spec, PlanDType::F32, 10, 2, None).unwrap();
        let (_, exact) = generation_memory_preflight(
            spec,
            PlanDType::F32,
            10,
            2,
            Some(MemoryBudget::user_declared(without_budget.persistent_bytes)),
        )
        .unwrap();
        assert_eq!(
            exact.placement,
            crate::PlacementVerdict::FitsPersistentEstimate
        );
        let (_, failed) = generation_memory_preflight(
            spec,
            PlanDType::F32,
            10,
            2,
            Some(MemoryBudget::user_declared(
                without_budget.persistent_bytes - 1,
            )),
        )
        .unwrap();
        assert_eq!(failed.placement, crate::PlacementVerdict::DoesNotFit);
    }
}
