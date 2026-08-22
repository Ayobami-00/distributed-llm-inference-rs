//! Native synchronous collectives derived exclusively from point-to-point tensor operations.

use crate::{
    BarrierTransport, CollectivesError, Communicator, ControlTransport, MessageTag, Rank, Result,
    Transport,
};
use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use std::time::Instant;

const COLLECTIVE_TAG_BASE: u64 = 0x4000_0000_0000_0000;
const COLLECTIVE_TAG_LIMIT: u64 = 0x5000_0000_0000_0000;
const TAG_SEQUENCE_STRIDE: u64 = 1 << 16;

/// Reduction operation supported by the correctness-first native backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReduceOp {
    /// Elementwise floating-point addition.
    Sum,
}

/// Native algorithm selected for all-reduce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AllReduceAlgorithm {
    /// Gather and sum on rank 0, followed by a rank-0 broadcast.
    Centralized,
    /// Ring reduce-scatter followed by ring all-gather.
    Ring,
}

/// Collective operation identity used by traces and reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollectiveKind {
    /// One root distributes a tensor to every rank.
    Broadcast,
    /// Rank-local shards are concatenated and returned to every rank.
    AllGather,
    /// Tensors are reduced onto one root.
    Reduce,
    /// Tensors are reduced and returned to every rank.
    AllReduce,
    /// Tensors are reduced and one result shard is returned to each rank.
    ReduceScatter,
    /// Every rank sends a distinct shard to every destination rank.
    AllToAll,
}

impl CollectiveKind {
    const fn tag_id(self) -> u64 {
        match self {
            Self::Broadcast => 1,
            Self::AllGather => 2,
            Self::Reduce => 3,
            Self::AllReduce => 4,
            Self::ReduceScatter => 5,
            Self::AllToAll => 6,
        }
    }
}

/// Concrete algorithm recorded for one collective call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollectiveAlgorithm {
    /// Root-to-peer direct sends.
    Direct,
    /// Rank-0 gather/reduce/scatter or broadcast composition.
    Centralized,
    /// Deterministic peer-exchange rounds.
    Pairwise,
    /// Ring reduce-scatter plus all-gather.
    Ring,
}

/// Metadata emitted immediately before a collective starts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectiveDescriptor {
    /// Rank-local monotonically increasing collective sequence.
    pub sequence: u64,
    /// Collective operation.
    pub kind: CollectiveKind,
    /// Native implementation used for the operation.
    pub algorithm: CollectiveAlgorithm,
    /// Root rank for rooted operations.
    pub root: Option<usize>,
    /// Tensor axis used for shard concatenation or splitting.
    pub axis: Option<usize>,
    /// Input tensor dimensions.
    pub input_shape: Vec<usize>,
    /// Logical F32 bytes in the local input.
    pub input_bytes: u64,
}

/// Completed collective trace with rank-local communication accounting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectiveTrace {
    /// Operation metadata.
    #[serde(flatten)]
    pub descriptor: CollectiveDescriptor,
    /// Output tensor dimensions on this rank, when the operation returns one.
    pub output_shape: Option<Vec<usize>>,
    /// Logical F32 payload bytes sent by this rank.
    pub sent_bytes: u64,
    /// Logical F32 payload bytes received by this rank.
    pub received_bytes: u64,
    /// Rank-local wall time spent in the complete collective.
    pub duration_ns: u64,
}

/// Live observational hook for native collective boundaries.
pub trait CollectiveObserver: Send {
    /// Called before any point-to-point work for one collective.
    fn collective_started(&mut self, descriptor: &CollectiveDescriptor);

    /// Called after a collective returns successfully.
    fn collective_completed(&mut self, trace: &CollectiveTrace);
}

/// Observer that discards collective notifications.
#[derive(Default)]
pub struct NoopCollectiveObserver;

impl CollectiveObserver for NoopCollectiveObserver {
    fn collective_started(&mut self, _descriptor: &CollectiveDescriptor) {}
    fn collective_completed(&mut self, _trace: &CollectiveTrace) {}
}

/// Transport-independent collective contract used by tensor-parallel model execution.
pub trait CollectiveCommunicator {
    /// Returns this participant's rank identity.
    fn rank(&self) -> Rank;

    /// Broadcasts the root tensor to every rank.
    fn broadcast(&mut self, tensor: &Tensor, root: usize) -> Result<Tensor>;

    /// Reduces tensors onto `root`; non-root ranks return `None`.
    fn reduce(&mut self, tensor: &Tensor, root: usize, op: ReduceOp) -> Result<Option<Tensor>>;

    /// Concatenates equal local shards along `axis` and returns the result to every rank.
    fn all_gather(&mut self, tensor: &Tensor, axis: usize) -> Result<Tensor>;

    /// Reduces tensors and returns the complete result to every rank.
    fn all_reduce(
        &mut self,
        tensor: &Tensor,
        op: ReduceOp,
        algorithm: AllReduceAlgorithm,
    ) -> Result<Tensor>;

    /// Reduces complete tensors and returns one equal shard along `axis` to each rank.
    fn reduce_scatter(&mut self, tensor: &Tensor, axis: usize, op: ReduceOp) -> Result<Tensor>;

    /// Exchanges equal input shards and concatenates received shards in source-rank order.
    fn all_to_all(&mut self, tensor: &Tensor, axis: usize) -> Result<Tensor>;
}

/// Correctness-first native collective implementation over a point-to-point communicator.
pub struct NativeCollectives<T: Transport> {
    communicator: Communicator<T>,
    next_sequence: u64,
    observer: Box<dyn CollectiveObserver>,
    traces: Vec<CollectiveTrace>,
}

impl<T: Transport> NativeCollectives<T> {
    /// Creates a native backend with a no-op observer.
    pub fn new(communicator: Communicator<T>) -> Self {
        Self::with_observer(communicator, NoopCollectiveObserver)
    }

    /// Creates a native backend that publishes live collective boundaries.
    pub fn with_observer<O>(communicator: Communicator<T>, observer: O) -> Self
    where
        O: CollectiveObserver + 'static,
    {
        Self {
            communicator,
            next_sequence: 0,
            observer: Box::new(observer),
            traces: Vec::new(),
        }
    }

    /// Returns completed rank-local traces in execution order.
    pub fn traces(&self) -> &[CollectiveTrace] {
        &self.traces
    }

    /// Removes and returns all completed traces.
    pub fn take_traces(&mut self) -> Vec<CollectiveTrace> {
        std::mem::take(&mut self.traces)
    }

    /// Returns the underlying point-to-point communicator.
    pub fn into_communicator(self) -> Communicator<T> {
        self.communicator
    }

    fn begin(
        &mut self,
        kind: CollectiveKind,
        algorithm: CollectiveAlgorithm,
        root: Option<usize>,
        axis: Option<usize>,
        tensor: &Tensor,
    ) -> Result<CollectiveDescriptor> {
        ensure_f32_cpu(tensor)?;
        let sequence = self.next_sequence;
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| CollectivesError::Collective("collective sequence overflow".into()))?;
        // Validate that at least the first tag for this sequence remains in the reserved range.
        collective_tag(sequence, kind, 0, 0)?;
        if let Some(root) = root {
            validate_root(self.rank(), root)?;
        }
        if let Some(axis) = axis {
            validate_axis(tensor, axis)?;
        }
        let descriptor = CollectiveDescriptor {
            sequence,
            kind,
            algorithm,
            root,
            axis,
            input_shape: tensor.dims().to_vec(),
            input_bytes: tensor_bytes(tensor)?,
        };
        self.observer.collective_started(&descriptor);
        Ok(descriptor)
    }

    fn finish(
        &mut self,
        descriptor: CollectiveDescriptor,
        output: Option<&Tensor>,
        sent_bytes: u64,
        received_bytes: u64,
        started: Instant,
    ) -> Result<()> {
        let trace = CollectiveTrace {
            descriptor,
            output_shape: output.map(|tensor| tensor.dims().to_vec()),
            sent_bytes,
            received_bytes,
            duration_ns: ns(started.elapsed()),
        };
        self.observer.collective_completed(&trace);
        self.traces.push(trace);
        Ok(())
    }

    fn send(&self, destination: usize, tag: MessageTag, tensor: &Tensor) -> Result<u64> {
        self.communicator.send_tensor(destination, tag, tensor)?;
        tensor_bytes(tensor)
    }

    fn recv(&self, source: usize, tag: MessageTag) -> Result<(Tensor, u64)> {
        let tensor = self.communicator.recv_tensor(source, tag)?;
        let bytes = tensor_bytes(&tensor)?;
        Ok((tensor, bytes))
    }

    fn exchange(
        &self,
        destination: usize,
        source: usize,
        send_tag: MessageTag,
        recv_tag: MessageTag,
        tensor: &Tensor,
    ) -> Result<(Tensor, u64, u64)> {
        let send_bytes = tensor_bytes(tensor)?;
        let communicator = &self.communicator;
        let result = std::thread::scope(|scope| {
            let send = scope.spawn(move || communicator.send_tensor(destination, send_tag, tensor));
            let received = communicator.recv_tensor(source, recv_tag);
            let sent = send.join().map_err(|_| {
                CollectivesError::Collective("collective send worker panicked".into())
            })?;
            sent?;
            received
        })?;
        let received_bytes = tensor_bytes(&result)?;
        Ok((result, send_bytes, received_bytes))
    }

    fn centralized_all_reduce(&self, tensor: &Tensor, sequence: u64) -> Result<(Tensor, u64, u64)> {
        let rank = self.rank();
        let root = 0;
        let gather_tag = collective_tag(sequence, CollectiveKind::AllReduce, 1, 0)?;
        let broadcast_tag = collective_tag(sequence, CollectiveKind::AllReduce, 2, 0)?;
        let mut sent = 0;
        let mut received = 0;
        if rank.global_rank() == root {
            let mut reduced = tensor.clone();
            for source in 1..rank.world_size() {
                let (incoming, bytes) = self.recv(source, gather_tag)?;
                require_same_shape(tensor, &incoming, CollectiveKind::AllReduce)?;
                reduced = (&reduced + &incoming)?;
                received += bytes;
            }
            for destination in 1..rank.world_size() {
                sent += self.send(destination, broadcast_tag, &reduced)?;
            }
            Ok((reduced, sent, received))
        } else {
            sent += self.send(root, gather_tag, tensor)?;
            let (result, bytes) = self.recv(root, broadcast_tag)?;
            require_same_shape(tensor, &result, CollectiveKind::AllReduce)?;
            received += bytes;
            Ok((result, sent, received))
        }
    }

    fn ring_all_reduce(&self, tensor: &Tensor, sequence: u64) -> Result<(Tensor, u64, u64)> {
        let rank = self.rank();
        let world = rank.world_size();
        if world == 1 {
            return Ok((tensor.clone(), 0, 0));
        }
        let flattened = tensor.flatten_all()?;
        let elements = flattened.elem_count();
        if elements % world != 0 {
            return Err(CollectivesError::Collective(format!(
                "ring all-reduce requires {elements} elements to divide evenly across {world} ranks"
            )));
        }
        let chunk_elements = elements / world;
        let mut chunks = (0..world)
            .map(|index| flattened.narrow(0, index * chunk_elements, chunk_elements))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let next = (rank.global_rank() + 1) % world;
        let previous = (rank.global_rank() + world - 1) % world;
        let mut sent = 0;
        let mut received = 0;

        for step in 0..world - 1 {
            let send_index = (rank.global_rank() + world - step) % world;
            let receive_index = (rank.global_rank() + world - step - 1) % world;
            let tag = collective_tag(sequence, CollectiveKind::AllReduce, 1, step)?;
            let (incoming, sent_bytes, received_bytes) =
                self.exchange(next, previous, tag, tag, &chunks[send_index])?;
            require_same_shape(&chunks[receive_index], &incoming, CollectiveKind::AllReduce)?;
            chunks[receive_index] = (&chunks[receive_index] + &incoming)?;
            sent += sent_bytes;
            received += received_bytes;
        }

        for step in 0..world - 1 {
            let send_index = (rank.global_rank() + 1 + world - step) % world;
            let receive_index = (rank.global_rank() + world - step) % world;
            let tag = collective_tag(sequence, CollectiveKind::AllReduce, 2, step)?;
            let (incoming, sent_bytes, received_bytes) =
                self.exchange(next, previous, tag, tag, &chunks[send_index])?;
            require_same_shape(&chunks[receive_index], &incoming, CollectiveKind::AllReduce)?;
            chunks[receive_index] = incoming;
            sent += sent_bytes;
            received += received_bytes;
        }
        let refs = chunks.iter().collect::<Vec<_>>();
        let result = Tensor::cat(&refs, 0)?.reshape(tensor.shape())?;
        Ok((result, sent, received))
    }
}

impl<T: ControlTransport> NativeCollectives<T> {
    /// Sends one typed-application control payload outside the collective tag namespace.
    pub fn send_control(&self, destination: usize, tag: MessageTag, bytes: Vec<u8>) -> Result<()> {
        self.communicator.send_control(destination, tag, bytes)
    }

    /// Receives one source/tag-matched typed-application control payload.
    pub fn recv_control(&self, source: usize, tag: MessageTag) -> Result<Vec<u8>> {
        self.communicator.recv_control(source, tag)
    }
}

impl<T: BarrierTransport> NativeCollectives<T> {
    /// Enters the transport's next reusable barrier generation.
    pub fn barrier(&self) -> Result<()> {
        self.communicator.barrier()
    }
}

impl<T: Transport> CollectiveCommunicator for NativeCollectives<T> {
    fn rank(&self) -> Rank {
        self.communicator.rank()
    }

    fn broadcast(&mut self, tensor: &Tensor, root: usize) -> Result<Tensor> {
        let descriptor = self.begin(
            CollectiveKind::Broadcast,
            CollectiveAlgorithm::Direct,
            Some(root),
            None,
            tensor,
        )?;
        let started = Instant::now();
        let tag = collective_tag(descriptor.sequence, descriptor.kind, 0, 0)?;
        let rank = self.rank();
        let mut sent = 0;
        let mut received = 0;
        let result = if rank.global_rank() == root {
            for destination in 0..rank.world_size() {
                if destination != root {
                    sent += self.send(destination, tag, tensor)?;
                }
            }
            tensor.clone()
        } else {
            let (result, bytes) = self.recv(root, tag)?;
            received += bytes;
            result
        };
        self.finish(descriptor, Some(&result), sent, received, started)?;
        Ok(result)
    }

    fn reduce(&mut self, tensor: &Tensor, root: usize, _op: ReduceOp) -> Result<Option<Tensor>> {
        let descriptor = self.begin(
            CollectiveKind::Reduce,
            CollectiveAlgorithm::Centralized,
            Some(root),
            None,
            tensor,
        )?;
        let started = Instant::now();
        let tag = collective_tag(descriptor.sequence, descriptor.kind, 0, 0)?;
        let rank = self.rank();
        let mut sent = 0;
        let mut received = 0;
        let result = if rank.global_rank() == root {
            let mut reduced = tensor.clone();
            for source in 0..rank.world_size() {
                if source != root {
                    let (incoming, bytes) = self.recv(source, tag)?;
                    require_same_shape(tensor, &incoming, descriptor.kind)?;
                    reduced = (&reduced + &incoming)?;
                    received += bytes;
                }
            }
            Some(reduced)
        } else {
            sent += self.send(root, tag, tensor)?;
            None
        };
        self.finish(descriptor, result.as_ref(), sent, received, started)?;
        Ok(result)
    }

    fn all_gather(&mut self, tensor: &Tensor, axis: usize) -> Result<Tensor> {
        let descriptor = self.begin(
            CollectiveKind::AllGather,
            CollectiveAlgorithm::Centralized,
            Some(0),
            Some(axis),
            tensor,
        )?;
        let started = Instant::now();
        let rank = self.rank();
        let gather_tag = collective_tag(descriptor.sequence, descriptor.kind, 1, 0)?;
        let broadcast_tag = collective_tag(descriptor.sequence, descriptor.kind, 2, 0)?;
        let mut sent = 0;
        let mut received = 0;
        let result = if rank.global_rank() == 0 {
            let mut shards = vec![tensor.clone()];
            for source in 1..rank.world_size() {
                let (incoming, bytes) = self.recv(source, gather_tag)?;
                require_gather_shape(tensor, &incoming, axis)?;
                shards.push(incoming);
                received += bytes;
            }
            let refs = shards.iter().collect::<Vec<_>>();
            let gathered = Tensor::cat(&refs, axis)?;
            for destination in 1..rank.world_size() {
                sent += self.send(destination, broadcast_tag, &gathered)?;
            }
            gathered
        } else {
            sent += self.send(0, gather_tag, tensor)?;
            let (gathered, bytes) = self.recv(0, broadcast_tag)?;
            received += bytes;
            gathered
        };
        self.finish(descriptor, Some(&result), sent, received, started)?;
        Ok(result)
    }

    fn all_reduce(
        &mut self,
        tensor: &Tensor,
        _op: ReduceOp,
        algorithm: AllReduceAlgorithm,
    ) -> Result<Tensor> {
        let trace_algorithm = match algorithm {
            AllReduceAlgorithm::Centralized => CollectiveAlgorithm::Centralized,
            AllReduceAlgorithm::Ring => CollectiveAlgorithm::Ring,
        };
        let descriptor = self.begin(
            CollectiveKind::AllReduce,
            trace_algorithm,
            (algorithm == AllReduceAlgorithm::Centralized).then_some(0),
            None,
            tensor,
        )?;
        let started = Instant::now();
        let (result, sent, received) = match algorithm {
            AllReduceAlgorithm::Centralized => {
                self.centralized_all_reduce(tensor, descriptor.sequence)?
            }
            AllReduceAlgorithm::Ring => self.ring_all_reduce(tensor, descriptor.sequence)?,
        };
        self.finish(descriptor, Some(&result), sent, received, started)?;
        Ok(result)
    }

    fn reduce_scatter(&mut self, tensor: &Tensor, axis: usize, _op: ReduceOp) -> Result<Tensor> {
        let descriptor = self.begin(
            CollectiveKind::ReduceScatter,
            CollectiveAlgorithm::Centralized,
            Some(0),
            Some(axis),
            tensor,
        )?;
        let world = self.rank().world_size();
        if tensor.dim(axis)? % world != 0 {
            return Err(CollectivesError::Collective(format!(
                "reduce-scatter axis {axis} length {} is not divisible by {world}",
                tensor.dim(axis)?
            )));
        }
        let started = Instant::now();
        let gather_tag = collective_tag(descriptor.sequence, descriptor.kind, 1, 0)?;
        let scatter_tag = collective_tag(descriptor.sequence, descriptor.kind, 2, 0)?;
        let rank = self.rank();
        let mut sent = 0;
        let mut received = 0;
        let result = if rank.global_rank() == 0 {
            let mut reduced = tensor.clone();
            for source in 1..world {
                let (incoming, bytes) = self.recv(source, gather_tag)?;
                require_same_shape(tensor, &incoming, descriptor.kind)?;
                reduced = (&reduced + &incoming)?;
                received += bytes;
            }
            let chunk = reduced.dim(axis)? / world;
            let local = reduced.narrow(axis, 0, chunk)?;
            for destination in 1..world {
                let shard = reduced.narrow(axis, destination * chunk, chunk)?;
                sent += self.send(destination, scatter_tag, &shard)?;
            }
            local
        } else {
            sent += self.send(0, gather_tag, tensor)?;
            let (shard, bytes) = self.recv(0, scatter_tag)?;
            received += bytes;
            shard
        };
        self.finish(descriptor, Some(&result), sent, received, started)?;
        Ok(result)
    }

    fn all_to_all(&mut self, tensor: &Tensor, axis: usize) -> Result<Tensor> {
        let descriptor = self.begin(
            CollectiveKind::AllToAll,
            CollectiveAlgorithm::Pairwise,
            None,
            Some(axis),
            tensor,
        )?;
        let rank = self.rank();
        let world = rank.world_size();
        if tensor.dim(axis)? % world != 0 {
            return Err(CollectivesError::Collective(format!(
                "all-to-all axis {axis} length {} is not divisible by {world}",
                tensor.dim(axis)?
            )));
        }
        let started = Instant::now();
        let chunk = tensor.dim(axis)? / world;
        let shards = (0..world)
            .map(|destination| tensor.narrow(axis, destination * chunk, chunk))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let mut received_shards = (0..world).map(|_| None).collect::<Vec<Option<Tensor>>>();
        received_shards[rank.global_rank()] = Some(shards[rank.global_rank()].clone());
        let mut sent = 0;
        let mut received = 0;
        for step in 1..world {
            let destination = (rank.global_rank() + step) % world;
            let source = (rank.global_rank() + world - step) % world;
            let tag = collective_tag(descriptor.sequence, descriptor.kind, 0, step - 1)?;
            let (incoming, sent_bytes, received_bytes) =
                self.exchange(destination, source, tag, tag, &shards[destination])?;
            require_gather_shape(&shards[destination], &incoming, axis)?;
            received_shards[source] = Some(incoming);
            sent += sent_bytes;
            received += received_bytes;
        }
        let received_shards = received_shards
            .into_iter()
            .map(|shard| shard.expect("every source contributes one all-to-all shard"))
            .collect::<Vec<_>>();
        let refs = received_shards.iter().collect::<Vec<_>>();
        let result = Tensor::cat(&refs, axis)?;
        self.finish(descriptor, Some(&result), sent, received, started)?;
        Ok(result)
    }
}

fn validate_root(rank: Rank, root: usize) -> Result<()> {
    if root >= rank.world_size() {
        return Err(CollectivesError::Collective(format!(
            "root rank {root} is outside world size {}",
            rank.world_size()
        )));
    }
    Ok(())
}

fn validate_axis(tensor: &Tensor, axis: usize) -> Result<()> {
    if axis >= tensor.rank() {
        return Err(CollectivesError::Collective(format!(
            "axis {axis} is outside tensor rank {}",
            tensor.rank()
        )));
    }
    Ok(())
}

fn ensure_f32_cpu(tensor: &Tensor) -> Result<()> {
    crate::TensorPacket::from_tensor(tensor).map(|_| ())
}

fn require_same_shape(expected: &Tensor, actual: &Tensor, kind: CollectiveKind) -> Result<()> {
    if expected.dims() != actual.dims() {
        return Err(CollectivesError::Collective(format!(
            "{kind:?} expected shape {:?}, received {:?}",
            expected.dims(),
            actual.dims()
        )));
    }
    Ok(())
}

fn require_gather_shape(expected: &Tensor, actual: &Tensor, axis: usize) -> Result<()> {
    if expected.rank() != actual.rank()
        || expected
            .dims()
            .iter()
            .zip(actual.dims())
            .enumerate()
            .any(|(index, (left, right))| index != axis && left != right)
    {
        return Err(CollectivesError::Collective(format!(
            "gather axis {axis} cannot combine shapes {:?} and {:?}",
            expected.dims(),
            actual.dims()
        )));
    }
    Ok(())
}

fn tensor_bytes(tensor: &Tensor) -> Result<u64> {
    let elements = u64::try_from(tensor.elem_count()).map_err(|_| {
        CollectivesError::Collective("tensor element count does not fit u64".into())
    })?;
    elements
        .checked_mul(4)
        .ok_or_else(|| CollectivesError::Collective("tensor byte count overflow".into()))
}

fn collective_tag(
    sequence: u64,
    kind: CollectiveKind,
    phase: usize,
    step: usize,
) -> Result<MessageTag> {
    if phase > 0x0f || step > u8::MAX as usize {
        return Err(CollectivesError::Collective(
            "collective phase or step exceeds tag capacity".into(),
        ));
    }
    let tag = sequence
        .checked_mul(TAG_SEQUENCE_STRIDE)
        .and_then(|value| value.checked_add(kind.tag_id() << 12))
        .and_then(|value| value.checked_add((phase as u64) << 8))
        .and_then(|value| value.checked_add(step as u64))
        .and_then(|value| value.checked_add(COLLECTIVE_TAG_BASE))
        .filter(|value| *value < COLLECTIVE_TAG_LIMIT)
        .ok_or_else(|| CollectivesError::Collective("collective message-tag overflow".into()))?;
    Ok(MessageTag(tag))
}

fn ns(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}
