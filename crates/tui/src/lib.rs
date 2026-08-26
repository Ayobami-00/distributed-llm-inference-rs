//! Read-only Ratatui projection over distributed pipeline and tensor events.
//!
//! This crate cannot launch, stop, retry, or reconfigure ranks. It reduces event records into a
//! dashboard state and renders that state until the producer reports completion or the user
//! disables visualization with `q`/Escape.

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use dlir_pipeline::{PipelineEvent, StageAssignment, StageMemoryPlan};
use dlir_runtime::{ExecutionPhase, RunEventKind};
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::Line,
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, Wrap},
};
use std::{
    io::{self, Stderr},
    sync::mpsc::{Receiver, TryRecvError},
    time::Duration,
};

/// Message accepted by the observational dashboard loop.
#[derive(Debug)]
pub enum DashboardMessage {
    /// One live rank event.
    Event(PipelineEvent),
    /// All rank streams completed or failed.
    Finished,
}

/// Reason the dashboard loop returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DashboardExit {
    /// The producer completed and no more events remain.
    Finished,
    /// The user disabled visualization while rank execution continues.
    Disabled,
    /// The user requested launcher-level cancellation with Ctrl-C.
    Interrupted,
}

/// Current read-only presentation state for one rank.
#[derive(Debug, Clone)]
pub struct RankView {
    /// Global rank.
    pub rank: usize,
    /// Human-readable assigned layer range.
    pub layers: String,
    /// Current execution phase.
    pub phase: String,
    /// Current or most recently completed global layer.
    pub layer: Option<usize>,
    /// Current rank state.
    pub state: String,
    /// Last completed layer compute duration.
    pub compute_ns: u64,
    /// Last tensor communication duration.
    pub communication_ns: u64,
    /// Logical persistent-stage bytes.
    pub logical_memory_bytes: u64,
    /// Latest cgroup memory usage.
    pub memory_current_bytes: Option<u64>,
    /// Enforced cgroup memory maximum.
    pub memory_limit_bytes: Option<u64>,
    phase_started_ns: Option<u64>,
}

/// Reducer state rendered by the terminal frontend.
#[derive(Debug, Clone)]
pub struct DashboardState {
    /// Display model ID.
    pub model: String,
    /// Transport backend.
    pub backend: String,
    /// Physical world size.
    pub world_size: usize,
    /// Tensor-parallel group size.
    pub tensor_parallel: usize,
    /// Pipeline-parallel group size.
    pub pipeline_parallel: usize,
    /// Expert dimension size; always one in supported execution.
    pub expert_parallel: usize,
    /// Selected native all-reduce, when TP is active.
    pub all_reduce: Option<String>,
    /// Rank assignment column heading.
    pub assignment_label: String,
    /// Rank-ordered rows.
    pub ranks: Vec<RankView>,
    /// Rank-0 prefill duration when complete.
    pub prefill_ns: Option<u64>,
    /// Mean completed rank-0 decode duration.
    pub decode_ns: Option<u64>,
    /// Rank-0 time from prefill start through the first generated token.
    pub time_to_first_token_ns: Option<u64>,
    /// Total logical tensor payload bytes observed from send events.
    pub communication_bytes: u64,
    /// Count of generated non-EOS tokens.
    pub generated_tokens: usize,
    /// Recent human-readable activity lines.
    pub recent: Vec<String>,
    decode_total_ns: u64,
    decode_count: u64,
    prefill_started_ns: Option<u64>,
}

impl DashboardState {
    /// Creates a dashboard from rank-ordered assignments and memory plans.
    pub fn new(
        model: impl Into<String>,
        assignments: &[StageAssignment],
        memory: &[StageMemoryPlan],
    ) -> Self {
        let ranks = assignments
            .iter()
            .zip(memory)
            .map(|(assignment, memory)| RankView {
                rank: assignment.rank,
                layers: format!("{}..{}", assignment.layer_start, assignment.layer_end),
                phase: "startup".to_owned(),
                layer: None,
                state: "WAITING".to_owned(),
                compute_ns: 0,
                communication_ns: 0,
                logical_memory_bytes: memory.persistent_bytes,
                memory_current_bytes: None,
                memory_limit_bytes: Some(memory.budget_bytes),
                phase_started_ns: None,
            })
            .collect();
        Self {
            model: model.into(),
            backend: "tcp".to_owned(),
            world_size: assignments.len(),
            tensor_parallel: 1,
            pipeline_parallel: assignments.len(),
            expert_parallel: 1,
            all_reduce: None,
            assignment_label: "Layers".to_owned(),
            ranks,
            prefill_ns: None,
            decode_ns: None,
            time_to_first_token_ns: None,
            communication_bytes: 0,
            generated_tokens: 0,
            recent: Vec::new(),
            decode_total_ns: 0,
            decode_count: 0,
            prefill_started_ns: None,
        }
    }

    /// Creates a read-only tensor-parallel dashboard from exact rank shard plans.
    pub fn new_tensor(
        model: impl Into<String>,
        memory: &[dlir_tensor::TensorParallelMemoryPlan],
        all_reduce: impl Into<String>,
    ) -> Self {
        let ranks = memory
            .iter()
            .map(|memory| RankView {
                rank: memory.rank,
                layers: format!(
                    "V{}..{} Q{}..{} K{}..{} I{}..{}",
                    memory.shard.vocabulary.start,
                    memory.shard.vocabulary.end,
                    memory.shard.query_heads.start,
                    memory.shard.query_heads.end,
                    memory.shard.kv_heads.start,
                    memory.shard.kv_heads.end,
                    memory.shard.intermediate.start,
                    memory.shard.intermediate.end,
                ),
                phase: "startup".to_owned(),
                layer: None,
                state: "WAITING".to_owned(),
                compute_ns: 0,
                communication_ns: 0,
                logical_memory_bytes: memory.persistent_bytes,
                memory_current_bytes: None,
                memory_limit_bytes: memory.budget_bytes,
                phase_started_ns: None,
            })
            .collect::<Vec<_>>();
        Self {
            model: model.into(),
            backend: "native/tcp".to_owned(),
            world_size: memory.len(),
            tensor_parallel: memory.len(),
            pipeline_parallel: 1,
            expert_parallel: 1,
            all_reduce: Some(all_reduce.into()),
            assignment_label: "Tensor shards".to_owned(),
            ranks,
            prefill_ns: None,
            decode_ns: None,
            time_to_first_token_ns: None,
            communication_bytes: 0,
            generated_tokens: 0,
            recent: Vec::new(),
            decode_total_ns: 0,
            decode_count: 0,
            prefill_started_ns: None,
        }
    }

    /// Applies one published rank event without changing cluster execution.
    pub fn apply(&mut self, published: &PipelineEvent) {
        let rank = published.event.rank;
        let Some(view) = self.ranks.get_mut(rank) else {
            self.push_recent(format!("ignored event from unknown rank {rank}"));
            return;
        };
        match &published.event.event {
            RunEventKind::ModelLoadStarted => view.state = "LOADING".to_owned(),
            RunEventKind::ModelLoadFinished => view.state = "READY".to_owned(),
            RunEventKind::LayerStarted { layer, phase, .. } => {
                view.layer = Some(*layer);
                view.phase = phase_name(*phase).to_owned();
                view.state = "COMPUTE".to_owned();
            }
            RunEventKind::LayerCompleted {
                layer, duration_ns, ..
            } => {
                view.layer = Some(*layer);
                view.compute_ns = *duration_ns;
                view.state = "READY".to_owned();
            }
            RunEventKind::TensorSent {
                bytes, duration_ns, ..
            } => {
                view.communication_ns = *duration_ns;
                view.state = "SEND".to_owned();
                self.communication_bytes = self.communication_bytes.saturating_add(*bytes);
            }
            RunEventKind::TensorReceived { duration_ns, .. } => {
                view.communication_ns = *duration_ns;
                view.state = "RECV".to_owned();
            }
            RunEventKind::ControlSent {
                bytes, duration_ns, ..
            } => {
                view.communication_ns = *duration_ns;
                view.state = "SEND CTRL".to_owned();
                self.communication_bytes = self.communication_bytes.saturating_add(*bytes);
            }
            RunEventKind::ControlReceived { duration_ns, .. } => {
                view.communication_ns = *duration_ns;
                view.state = "RECV CTRL".to_owned();
            }
            RunEventKind::CollectiveStarted { .. } => view.state = "BARRIER".to_owned(),
            RunEventKind::CollectiveCompleted { .. } => view.state = "READY".to_owned(),
            RunEventKind::TensorCollectiveStarted {
                collective,
                algorithm,
                collective_sequence,
                ..
            } => {
                view.state = format!(
                    "{} #{}",
                    collective.to_ascii_uppercase(),
                    collective_sequence
                );
                self.push_recent(format!(
                    "rank {rank}: {algorithm} {collective} #{collective_sequence}"
                ));
            }
            RunEventKind::TensorCollectiveCompleted {
                collective,
                sent_bytes,
                duration_ns,
                ..
            } => {
                view.state = "READY".to_owned();
                view.communication_ns = *duration_ns;
                self.communication_bytes = self.communication_bytes.saturating_add(*sent_bytes);
                self.push_recent(format!("rank {rank}: {collective} complete"));
            }
            RunEventKind::MemorySample {
                current_bytes,
                limit_bytes,
            } => {
                view.memory_current_bytes = *current_bytes;
                view.memory_limit_bytes = *limit_bytes;
            }
            RunEventKind::PrefillStarted { .. } => {
                view.phase = "prefill".to_owned();
                view.phase_started_ns = Some(published.event.elapsed_ns);
                if rank == 0 {
                    self.prefill_started_ns = Some(published.event.elapsed_ns);
                }
            }
            RunEventKind::PrefillFinished => {
                view.state = "READY".to_owned();
                if rank == 0 {
                    self.prefill_ns = view
                        .phase_started_ns
                        .map(|started| published.event.elapsed_ns.saturating_sub(started));
                }
            }
            RunEventKind::DecodeStepStarted { .. } => {
                view.phase = "decode".to_owned();
                view.phase_started_ns = Some(published.event.elapsed_ns);
            }
            RunEventKind::DecodeStepFinished { .. } => {
                view.state = "READY".to_owned();
                if rank == 0 {
                    let duration = view
                        .phase_started_ns
                        .map(|started| published.event.elapsed_ns.saturating_sub(started))
                        .unwrap_or(0);
                    self.decode_total_ns = self.decode_total_ns.saturating_add(duration);
                    self.decode_count = self.decode_count.saturating_add(1);
                    self.decode_ns = Some(self.decode_total_ns / self.decode_count);
                }
            }
            RunEventKind::TokenGenerated { token_id, .. } => {
                self.generated_tokens += 1;
                if rank == 0 && self.time_to_first_token_ns.is_none() {
                    self.time_to_first_token_ns = self
                        .prefill_started_ns
                        .map(|started| published.event.elapsed_ns.saturating_sub(started));
                }
                self.push_recent(format!("rank {rank} generated token {token_id}"));
            }
            RunEventKind::GenerationFinished { stop_reason } => {
                view.state = "DONE".to_owned();
                self.push_recent(format!("rank {rank} finished: {stop_reason}"));
            }
            _ => {}
        }
    }

    fn push_recent(&mut self, line: String) {
        self.recent.push(line);
        if self.recent.len() > 5 {
            self.recent.remove(0);
        }
    }
}

/// Runs the alternate-screen dashboard until completion or user disablement.
pub fn run_dashboard(
    state: &mut DashboardState,
    receiver: &Receiver<DashboardMessage>,
) -> io::Result<DashboardExit> {
    let mut terminal = TerminalGuard::enter()?;
    loop {
        loop {
            match receiver.try_recv() {
                Ok(DashboardMessage::Event(event)) => state.apply(&event),
                Ok(DashboardMessage::Finished) => {
                    terminal.draw(|frame| render(frame, state))?;
                    return Ok(DashboardExit::Finished);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return Ok(DashboardExit::Finished),
            }
        }
        terminal.draw(|frame| render(frame, state))?;
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press
                    && matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)
                {
                    return Ok(DashboardExit::Disabled);
                }
                if key.kind == KeyEventKind::Press
                    && key.code == KeyCode::Char('c')
                    && key.modifiers.contains(KeyModifiers::CONTROL)
                {
                    return Ok(DashboardExit::Interrupted);
                }
            }
        }
    }
}

/// Renders one dashboard frame; exposed for deterministic `TestBackend` snapshots.
pub fn render(frame: &mut Frame<'_>, state: &DashboardState) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(7),
            Constraint::Length(6),
        ])
        .split(area);
    let header = Paragraph::new(format!(
        "Model: {}   TP={} PP={} EP={}   Backend={}{}",
        state.model,
        state.tensor_parallel,
        state.pipeline_parallel,
        state.expert_parallel,
        state.backend,
        state
            .all_reduce
            .as_ref()
            .map(|value| format!("   AllReduce={value}"))
            .unwrap_or_default(),
    ))
    .block(
        Block::default()
            .title(" Distributed Inference ")
            .borders(Borders::ALL),
    );
    frame.render_widget(header, chunks[0]);
    render_ranks(frame, chunks[1], state);
    let footer = Paragraph::new(vec![
        Line::from(format!(
            "Prefill: {}   TTFT: {}",
            format_duration(state.prefill_ns),
            format_duration(state.time_to_first_token_ns),
        )),
        Line::from(format!(
            "Mean decode: {}   Tokens: {}",
            format_duration(state.decode_ns),
            state.generated_tokens
        )),
        Line::from(format!(
            "Communication: {}   q/Esc: disable visualization",
            format_bytes(state.communication_bytes)
        )),
        Line::from(state.recent.last().cloned().unwrap_or_default()),
    ])
    .wrap(Wrap { trim: true })
    .block(Block::default().title(" Summary ").borders(Borders::ALL));
    frame.render_widget(footer, chunks[2]);
}

fn render_ranks(frame: &mut Frame<'_>, area: Rect, state: &DashboardState) {
    let header = Row::new(vec![
        "Rank".to_owned(),
        state.assignment_label.clone(),
        "Phase".to_owned(),
        "Layer".to_owned(),
        "State".to_owned(),
        "Compute".to_owned(),
        "Comm".to_owned(),
        "Memory".to_owned(),
    ])
    .style(
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );
    let rows = state.ranks.iter().map(|rank| {
        Row::new(vec![
            Cell::from(rank.rank.to_string()),
            Cell::from(rank.layers.clone()),
            Cell::from(rank.phase.clone()),
            Cell::from(rank.layer.map_or_else(|| "-".to_owned(), |v| v.to_string())),
            Cell::from(rank.state.clone()),
            Cell::from(format_duration(Some(rank.compute_ns))),
            Cell::from(format_duration(Some(rank.communication_ns))),
            Cell::from(format!(
                "{}/{}",
                format_bytes(
                    rank.memory_current_bytes
                        .unwrap_or(rank.logical_memory_bytes)
                ),
                format_bytes(rank.memory_limit_bytes.unwrap_or(0))
            )),
        ])
    });
    let widths = [
        Constraint::Length(5),
        Constraint::Length(if state.tensor_parallel > 1 { 32 } else { 9 }),
        Constraint::Length(8),
        Constraint::Length(6),
        Constraint::Length(9),
        Constraint::Length(9),
        Constraint::Length(9),
        Constraint::Min(14),
    ];
    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().title(" Ranks ").borders(Borders::ALL));
    frame.render_widget(table, area);
}

fn phase_name(phase: ExecutionPhase) -> &'static str {
    match phase {
        ExecutionPhase::Prefill => "prefill",
        ExecutionPhase::Decode => "decode",
    }
}

fn format_duration(duration: Option<u64>) -> String {
    match duration {
        None => "-".to_owned(),
        Some(ns) if ns >= 1_000_000 => format!("{:.2}ms", ns as f64 / 1_000_000.0),
        Some(ns) if ns >= 1_000 => format!("{:.2}us", ns as f64 / 1_000.0),
        Some(ns) => format!("{ns}ns"),
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.2}GiB", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.1}MiB", bytes as f64 / (1u64 << 20) as f64)
    } else if bytes >= 1 << 10 {
        format!("{:.1}KiB", bytes as f64 / (1u64 << 10) as f64)
    } else {
        format!("{bytes}B")
    }
}

struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<Stderr>>,
}

impl TerminalGuard {
    fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stderr = io::stderr();
        if let Err(error) = execute!(stderr, EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(error);
        }
        let terminal = Terminal::new(CrosstermBackend::new(stderr))?;
        Ok(Self { terminal })
    }
}

impl std::ops::Deref for TerminalGuard {
    type Target = Terminal<CrosstermBackend<Stderr>>;

    fn deref(&self) -> &Self::Target {
        &self.terminal
    }
}

impl std::ops::DerefMut for TerminalGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.terminal
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dlir_pipeline::{PipelinePartition, StageMemoryPlan};
    use dlir_runtime::{PlanDType, RunEvent, SupportedModelId};
    use ratatui::{Terminal, backend::TestBackend};

    fn state() -> DashboardState {
        let spec = SupportedModelId::SmolLm2_135MInstruct.spec();
        let partition = PipelinePartition::balanced(spec, 2).unwrap();
        let memory = partition
            .stages
            .iter()
            .map(|stage| {
                StageMemoryPlan::for_stage(spec, stage, PlanDType::F32, 8, 1 << 30).unwrap()
            })
            .collect::<Vec<_>>();
        DashboardState::new(spec.id.as_str(), &partition.stages, &memory)
    }

    fn event(sequence: u64, elapsed_ns: u64, kind: RunEventKind) -> PipelineEvent {
        PipelineEvent {
            schema_version: 1,
            run_id: "run".to_owned(),
            request_id: "request".to_owned(),
            event: RunEvent {
                sequence,
                rank: 0,
                elapsed_ns,
                event: kind,
            },
        }
    }

    #[test]
    fn renders_normal_and_narrow_terminals() {
        for (width, height) in [(100, 24), (70, 16)] {
            let backend = TestBackend::new(width, height);
            let mut terminal = Terminal::new(backend).unwrap();
            let state = state();
            terminal.draw(|frame| render(frame, &state)).unwrap();
            let rendered = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|cell| cell.symbol())
                .collect::<String>();
            assert!(rendered.contains("Model"));
            assert!(rendered.contains("Rank"));
        }
    }

    #[test]
    fn reducer_calculates_phase_durations_and_sent_bytes() {
        let mut state = state();
        state.apply(&event(
            0,
            100,
            RunEventKind::PrefillStarted { prompt_tokens: 4 },
        ));
        state.apply(&event(
            1,
            150,
            RunEventKind::TensorSent {
                peer: 1,
                purpose: dlir_runtime::TensorPurpose::Activation,
                phase: ExecutionPhase::Prefill,
                step: 0,
                shape: vec![1, 4, 8],
                bytes: 128,
                duration_ns: 10,
            },
        ));
        state.apply(&event(
            2,
            180,
            RunEventKind::TokenGenerated {
                token_id: 3,
                text: "x".into(),
            },
        ));
        state.apply(&event(3, 200, RunEventKind::PrefillFinished));
        state.apply(&event(4, 300, RunEventKind::DecodeStepStarted { step: 1 }));
        state.apply(&event(5, 360, RunEventKind::DecodeStepFinished { step: 1 }));
        assert_eq!(state.prefill_ns, Some(100));
        assert_eq!(state.time_to_first_token_ns, Some(80));
        assert_eq!(state.decode_ns, Some(60));
        assert_eq!(state.communication_bytes, 128);
        assert_eq!(state.generated_tokens, 1);
    }

    #[test]
    fn tensor_dashboard_renders_shards_and_reduces_collective_events() {
        let partition = dlir_tensor::TensorParallelPartition::plan(
            SupportedModelId::SmolLm2_135MInstruct,
            3,
            8,
            Some(512 << 20),
        )
        .unwrap();
        let mut state =
            DashboardState::new_tensor("smollm2-135m-instruct", &partition.ranks, "ring");
        state.apply(&event(
            0,
            10,
            RunEventKind::TensorCollectiveStarted {
                collective: "all_reduce".into(),
                algorithm: "ring".into(),
                collective_sequence: 4,
                shape: vec![1, 4, 576],
            },
        ));
        state.apply(&event(
            1,
            20,
            RunEventKind::TensorCollectiveCompleted {
                collective: "all_reduce".into(),
                collective_sequence: 4,
                sent_bytes: 9_216,
                received_bytes: 9_216,
                duration_ns: 10,
            },
        ));
        assert_eq!(state.tensor_parallel, 3);
        assert_eq!(state.pipeline_parallel, 1);
        assert_eq!(state.communication_bytes, 9_216);
        assert!(state.ranks[0].layers.contains("V0..16384"));
        let backend = TestBackend::new(120, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, &state)).unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(rendered.contains("TP=3 PP=1 EP=1"));
        assert!(rendered.contains("AllReduce=ring"));
    }
}
