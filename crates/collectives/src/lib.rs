//! Point-to-point communication foundation for the distributed inference laboratory.
//!
//! `v0.2-collectives` introduces logical ranks hosted by worker threads, an in-memory transport,
//! and copied CPU/F32 tensor send and receive operations. The transport boundary contains no
//! collective algorithm or model inference behavior.
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

mod communicator;
mod error;
mod in_memory;
mod rank;
mod report;
mod runner;
mod tensor;
mod transport;

pub use communicator::Communicator;
pub use error::{CollectivesError, Result};
pub use in_memory::{InMemoryTransport, create_in_memory_world};
pub use rank::{MessageTag, Rank};
pub use report::{P2pReport, RankExchangeReport, TensorSummary, run_p2p_ring};
pub use runner::run_in_memory;
pub use tensor::TensorPacket;
pub use transport::Transport;
