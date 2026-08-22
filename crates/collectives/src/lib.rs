//! Point-to-point and barrier communication for the distributed inference laboratory.
//!
//! `v0.2-collectives` introduces logical ranks hosted by worker threads, an in-memory transport,
//! and copied CPU/F32 tensor send and receive operations. `v0.3-tcp` adds one rank per process,
//! rank-0 rendezvous, full-mesh TCP streams, and reusable barrier synchronization. `v0.4` adds
//! protocol-v2 bounded control packets so pipeline ranks can exchange token IDs and decisions
//! independently of copied activation tensors.
//!
//! # Two-rank tensor exchange
//!
//! ```
//! use candle_core::{Device, Tensor};
//! use dlir_collectives::{MessageTag, run_in_memory};
//! use std::time::Duration;
//!
//! let results = run_in_memory(2, Duration::from_secs(1), |communicator| {
//!     let rank = communicator.rank().global_rank();
//!     if rank == 0 {
//!         let tensor = Tensor::new(&[1f32, 2., 3., 4.], &Device::Cpu)?;
//!         communicator.send_tensor(1, MessageTag(7), &tensor)?;
//!         Ok(Vec::new())
//!     } else {
//!         let tensor = communicator.recv_tensor(0, MessageTag(7))?;
//!         Ok(tensor.to_vec1::<f32>()?)
//!     }
//! })?;
//! assert_eq!(results[1], vec![1., 2., 3., 4.]);
//! # Ok::<(), dlir_collectives::CollectivesError>(())
//! ```

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

mod barrier;
mod benchmark;
mod check;
mod collective;
mod communicator;
mod control;
mod error;
mod in_memory;
mod rank;
mod report;
mod runner;
mod tcp;
mod tensor;
mod transport;

pub use barrier::BarrierTransport;
pub use benchmark::{
    CollectiveBenchmarkCase, CollectiveBenchmarkManifest, CollectiveBenchmarkRankCase,
    CollectiveBenchmarkRankReport, CollectiveBenchmarkReport, CollectiveBenchmarkResources,
    aggregate_all_reduce_benchmark, run_all_reduce_benchmark_rank,
    run_in_memory_all_reduce_benchmark,
};
pub use check::{
    CollectiveCheckOperation, CollectiveCheckRankReport, CollectiveCheckReport,
    run_collective_check,
};
pub use collective::{
    AllReduceAlgorithm, CollectiveAlgorithm, CollectiveCommunicator, CollectiveDescriptor,
    CollectiveKind, CollectiveObserver, CollectiveTrace, NativeCollectives, NoopCollectiveObserver,
    ReduceOp,
};
pub use communicator::Communicator;
pub use control::{ControlPacket, MAX_CONTROL_BYTES};
pub use error::{CollectivesError, Result};
pub use in_memory::{InMemoryTransport, create_in_memory_world};
pub use rank::{MessageTag, Rank};
pub use report::{P2pReport, RankExchangeReport, TensorSummary, run_p2p_ring};
pub use runner::run_in_memory;
pub use tcp::{
    DEFAULT_MAX_TENSOR_BYTES, PROTOCOL_VERSION, PeerInfo, TcpTransport, TcpTransportConfig,
};
pub use tensor::TensorPacket;
pub use transport::{ControlTransport, Transport};
