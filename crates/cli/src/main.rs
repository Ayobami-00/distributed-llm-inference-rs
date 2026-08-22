//! Command-line presentation layer for the single-process dlir runtime.
//!
//! The binary converts CLI arguments into runtime request types, renders model and inspection
//! results, streams assistant text through an event observer, writes optional JSON reports, and
//! owns exit behavior. Inference and planning remain in `dlir-runtime`.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use dlir_runtime::{
    EventObserver, GenerationRequest, InspectionReport, InspectionRequest, MemoryBudget, ModelSpec,
    PlanDType, RankMemoryPlan, RunEvent, RunEventKind, SupportedModelId, format_bytes, generate,
    inspect, supported_models,
};
use serde_json::json;
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

#[derive(Debug, Parser)]
#[command(
    name = "dlir",
    version,
    about = "A single-process Llama inference baseline"
)]
struct Cli {
    /// Operation to perform.
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// List the closed set of model checkpoints supported by this release.
    Models {
        /// Render human-readable text or schema-versioned JSON.
        #[arg(long, value_enum, default_value_t)]
        format: OutputFormat,
    },
    /// Inspect architecture and logical memory without downloading artifacts.
    Inspect {
        /// Exact identifier from `dlir models`; arbitrary Hub IDs are rejected.
        #[arg(long)]
        model: SupportedModelId,
        /// Logical planning dtype: f16, bf16, or f32.
        #[arg(long, default_value = "f32")]
        dtype: PlanDType,
        /// KV-cache capacity to model in token positions.
        #[arg(long, default_value_t = 512)]
        context_length: usize,
        /// Optional advisory per-rank host budget using bytes, KiB, MiB, or GiB.
        #[arg(long)]
        device_memory_budget: Option<MemoryBudget>,
        /// Render human-readable text or schema-versioned JSON.
        #[arg(long, value_enum, default_value_t)]
        format: OutputFormat,
        /// Write the selected representation to a file instead of stdout.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Generate one deterministic assistant completion on CPU.
    Generate {
        /// Exact identifier from `dlir models`; arbitrary Hub IDs are rejected.
        #[arg(long)]
        model: SupportedModelId,
        /// Execution device; v0.1 accepts only cpu.
        #[arg(long, value_enum, default_value_t)]
        device: DeviceArg,
        /// Runtime dtype; v0.1 CPU generation accepts only f32.
        #[arg(long, default_value = "f32")]
        dtype: PlanDType,
        /// Non-empty user message to wrap in the model's registered chat template.
        #[arg(long)]
        prompt: String,
        /// Maximum number of non-EOS tokens to emit; must be at least one.
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,
        /// Advisory per-rank host budget in bytes/KiB/MiB/GiB, checked before weight download.
        #[arg(long)]
        device_memory_budget: Option<MemoryBudget>,
        /// Write the complete schema-v1 generation report as JSON.
        #[arg(long)]
        report: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum OutputFormat {
    #[default]
    Text,
    Json,
}

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum DeviceArg {
    #[default]
    Cpu,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Models { format } => print_models(format),
        Command::Inspect {
            model,
            dtype,
            context_length,
            device_memory_budget,
            format,
            output,
        } => run_inspect(
            InspectionRequest {
                model,
                dtype,
                context_length,
                device_memory_budget,
            },
            format,
            output.as_deref(),
        ),
        Command::Generate {
            model,
            device: DeviceArg::Cpu,
            dtype,
            prompt,
            max_new_tokens,
            device_memory_budget,
            report,
        } => run_generate(
            GenerationRequest {
                model,
                dtype,
                prompt,
                max_new_tokens,
                device_memory_budget,
            },
            report.as_deref(),
        ),
    }
}

fn print_models(format: OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Text => {
            println!("SUPPORTED MODELS");
            for spec in supported_models() {
                println!("\n{}", spec.id);
                println!("  repository:       {}", spec.repository);
                println!("  revision:         {}", spec.revision);
                println!("  parameters:       {}", grouped(spec.expected_parameters));
                println!(
                    "  max context:      {}",
                    spec.config.max_position_embeddings
                );
                println!("  CPU/F32:          validated");
                println!("  CUDA:             planned");
                println!("  chat template:    {}", spec.prompt_template.name());
            }
        }
        OutputFormat::Json => {
            let models = supported_models()
                .iter()
                .map(model_json)
                .collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "schema_version": 1,
                    "models": models,
                }))?
            );
        }
    }
    Ok(())
}

fn model_json(spec: &ModelSpec) -> serde_json::Value {
    json!({
        "id": spec.id,
        "repository": spec.repository,
        "revision": spec.revision,
        "weight_file": spec.weight_file,
        "expected_parameters": spec.expected_parameters,
        "expected_checkpoint_bytes": spec.expected_checkpoint_bytes,
        "checkpoint_dtype": spec.checkpoint_dtype,
        "tensor_layout": spec.tensor_layout,
        "configuration": spec.config,
        "chat_template": spec.prompt_template.name(),
        "execution": {
            "cpu_f32": "validated",
            "cuda": "planned",
        },
    })
}

fn run_inspect(
    request: InspectionRequest,
    format: OutputFormat,
    output: Option<&Path>,
) -> Result<()> {
    let report = inspect(&request)?;
    let rendered = match format {
        OutputFormat::Text => inspection_text(&report),
        OutputFormat::Json => format!("{}\n", serde_json::to_string_pretty(&report)?),
    };
    if let Some(path) = output {
        fs::write(path, rendered)
            .with_context(|| format!("failed to write inspection to {}", path.display()))?;
    } else {
        print!("{rendered}");
    }
    Ok(())
}

fn inspection_text(report: &InspectionReport) -> String {
    let cfg = &report.config;
    let memory = &report.memory;
    let mut text = format!(
        "MODEL INSPECTION\n\
         Model:              {}\n\
         Repository:         {}\n\
         Revision:           {}\n\
         Parameters:         {}\n\
         Layers:             {}\n\
         Attention heads:    {}\n\
         KV heads:           {}\n\
         Hidden dimension:   {}\n\
         Head dimension:     {}\n\
         Maximum context:    {}\n\
         Planned context:    {}\n\
         Dtype:              {}\n\
         Logical weights:    {} ({})\n\
         KV cache capacity:  {} ({})\n\
         Persistent minimum: {} ({})\n",
        report.model,
        report.repository,
        report.revision,
        grouped(memory.parameter_count),
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.hidden_size,
        cfg.hidden_size / cfg.num_attention_heads,
        cfg.max_position_embeddings,
        memory.context_length,
        memory.dtype,
        format_bytes(memory.weight_bytes),
        memory.weight_bytes,
        format_bytes(memory.kv_cache_capacity_bytes),
        memory.kv_cache_capacity_bytes,
        format_bytes(memory.persistent_bytes),
        memory.persistent_bytes,
    );
    if let Some(budget) = memory.budget {
        text.push_str(&format!(
            "Device budget:      {} ({})\n\
             Budget semantics:   host, per-rank, user-declared, not OS-enforced\n",
            format_bytes(budget.bytes),
            budget.bytes,
        ));
    }
    text.push_str(&format!("Placement:          {}\n", memory.placement));
    text.push_str(&format!("Note: {0}\n", report.caveat));
    text
}

fn run_generate(request: GenerationRequest, report_path: Option<&Path>) -> Result<()> {
    let spec = request.model.spec();
    eprintln!(
        "model: {} ({})\nrevision: {}\ncheckpoint: {} (known download size {})",
        request.model,
        spec.repository,
        spec.revision,
        spec.weight_file,
        format_bytes(spec.expected_checkpoint_bytes),
    );
    let mut observer = CliObserver::default();
    let report = generate(&request, &mut observer)?;
    observer.finish(&report.completion)?;

    if let Some(path) = report_path {
        let json = serde_json::to_vec_pretty(&report)?;
        fs::write(path, json)
            .with_context(|| format!("failed to write run report to {}", path.display()))?;
        eprintln!("report: {}", path.display());
    }
    eprint!("{}", generation_summary(&report.memory, &report));
    Ok(())
}

#[derive(Default)]
struct CliObserver {
    artifact_phase: usize,
    streamed: String,
}

impl CliObserver {
    fn finish(&mut self, completion: &str) -> Result<()> {
        if let Some(suffix) = completion.strip_prefix(&self.streamed) {
            print!("{suffix}");
        }
        println!();
        io::stdout().flush()?;
        Ok(())
    }
}

impl EventObserver for CliObserver {
    fn on_event(&mut self, event: &RunEvent) {
        match &event.event {
            RunEventKind::ArtifactResolutionStarted => {
                self.artifact_phase += 1;
                let phase = if self.artifact_phase == 1 {
                    "resolving configuration and tokenizer"
                } else {
                    "resolving checkpoint weights"
                };
                eprintln!("{phase}...");
            }
            RunEventKind::ModelLoadStarted => eprintln!("loading model on CPU..."),
            RunEventKind::PrefillStarted { prompt_tokens } => {
                eprintln!("prefill: {prompt_tokens} prompt tokens");
            }
            RunEventKind::TokenGenerated { text, .. } => {
                print!("{text}");
                let _ = io::stdout().flush();
                self.streamed.push_str(text);
            }
            _ => {}
        }
    }
}

fn generation_summary(memory: &RankMemoryPlan, report: &dlir_runtime::GenerationReport) -> String {
    let timings = &report.timings;
    format!(
        "\nRUN SUMMARY\n\
         prompt tokens:       {}\n\
         generated tokens:    {}\n\
         stop reason:         {}\n\
         logical weights:     {}\n\
         KV capacity:         {} for {} tokens\n\
         final KV used:       {}\n\
         placement:           {}\n\
         artifacts:           {:.3} s\n\
         model load:          {:.3} s\n\
         tokenization:        {:.3} ms\n\
         prefill:             {:.3} ms ({:.2} tok/s)\n\
         TTFT:                {:.3} ms\n\
         decode forwards:     {}\n\
         mean decode latency: {}\n\
         decode throughput:   {}\n\
         generation total:    {:.3} s\n\
         cold start total:    {:.3} s\n",
        report.prompt_tokens,
        report.generated_tokens.len(),
        report.stop_reason,
        format_bytes(memory.weight_bytes),
        format_bytes(memory.kv_cache_capacity_bytes),
        memory.context_length,
        format_bytes(report.final_kv_cache_bytes),
        memory.placement,
        ns_seconds(timings.artifact_resolution_ns),
        ns_seconds(timings.model_load_ns),
        ns_millis(timings.tokenization_ns),
        ns_millis(timings.prefill_ns),
        timings.prefill_tokens_per_second,
        ns_millis(timings.time_to_first_token_ns),
        timings.decode_forward_count,
        timings
            .mean_decode_ns
            .map(|ns| format!("{:.3} ms", ns_millis(ns)))
            .unwrap_or_else(|| "n/a".into()),
        timings
            .decode_tokens_per_second
            .map(|rate| format!("{rate:.2} tok/s"))
            .unwrap_or_else(|| "n/a".into()),
        ns_seconds(timings.generation_total_ns),
        ns_seconds(timings.cold_start_total_ns),
    )
}

fn ns_millis(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}

fn ns_seconds(ns: u64) -> f64 {
    ns as f64 / 1_000_000_000.0
}

fn grouped(value: u64) -> String {
    let digits = value.to_string();
    let mut result = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, byte) in digits.bytes().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            result.push(',');
        }
        result.push(char::from(byte));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parser_rejects_unknown_models_and_ambiguous_budgets() {
        assert!(Cli::try_parse_from(["dlir", "inspect", "--model", "someone/model"]).is_err());
        assert!(
            Cli::try_parse_from([
                "dlir",
                "inspect",
                "--model",
                "smollm2-135m-instruct",
                "--device-memory-budget",
                "500M",
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "dlir",
                "generate",
                "--model",
                "smollm2-135m-instruct",
                "--device",
                "cuda",
                "--prompt",
                "hello",
            ])
            .is_err()
        );
    }

    #[test]
    fn inspect_text_matches_the_acceptance_baseline() {
        let report = inspect(&InspectionRequest {
            model: SupportedModelId::SmolLm2_135MInstruct,
            dtype: PlanDType::F32,
            context_length: 512,
            device_memory_budget: Some(MemoryBudget::user_declared(500 << 20)),
        })
        .unwrap();
        let text = inspection_text(&report);
        assert!(text.contains("Parameters:         134,515,008"));
        assert!(text.contains("Logical weights:    513.1 MiB (538060032)"));
        assert!(text.contains("KV cache capacity:  22.5 MiB (23592960)"));
        assert!(text.contains("Persistent minimum: 535.6 MiB (561652992)"));
        assert!(text.contains("Placement:          FAILED"));
    }

    #[test]
    fn model_json_is_versionable_and_includes_pin_and_support() {
        let value = model_json(SupportedModelId::TinyLlama1_1BChat.spec());
        assert_eq!(value["id"], "tinyllama-1.1b-chat");
        assert_eq!(
            value["revision"],
            "5243d158d6f4b356f1142ea8fd6a99cb5ac2c0e1"
        );
        assert_eq!(value["execution"]["cpu_f32"], "validated");
        assert_eq!(value["execution"]["cuda"], "planned");
    }

    #[test]
    fn generation_rejects_non_f32_before_resolving_artifacts() {
        let error = generate(
            &GenerationRequest {
                model: SupportedModelId::SmolLm2_135MInstruct,
                dtype: PlanDType::Bf16,
                prompt: "hello".into(),
                max_new_tokens: 1,
                device_memory_budget: None,
            },
            &mut dlir_runtime::NoopObserver,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            dlir_runtime::DlirError::UnsupportedExecution { .. }
        ));
    }

    #[test]
    fn inspect_json_can_be_written_when_placement_fails() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("inspection.json");
        run_inspect(
            InspectionRequest {
                model: SupportedModelId::SmolLm2_135MInstruct,
                dtype: PlanDType::F32,
                context_length: 512,
                device_memory_budget: Some(MemoryBudget::user_declared(500 << 20)),
            },
            OutputFormat::Json,
            Some(&path),
        )
        .unwrap();
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["memory"]["placement"], "does_not_fit");
    }
}
