//! In-memory transport whose ranks exchange owned messages through FIFO channels.

use crate::transport::MessageFrame;
use crate::{CollectivesError, MessageTag, Rank, Result, TensorPacket, Transport};
use std::{
    collections::VecDeque,
    sync::{Mutex, mpsc},
    time::{Duration, Instant},
};

/// One rank endpoint in an in-memory communication world.
///
/// Endpoints are created together with [`create_in_memory_world`]. Each directional rank pair has
/// its own unbounded FIFO channel. Pending messages preserve packets whose tag is not the one a
/// caller is currently receiving.
pub struct InMemoryTransport {
    rank: Rank,
    senders: Vec<Option<mpsc::Sender<MessageFrame>>>,
    receivers: Vec<Option<Mutex<mpsc::Receiver<MessageFrame>>>>,
    pending: Mutex<VecDeque<MessageFrame>>,
    receive_timeout: Duration,
}

/// Creates exactly one in-memory transport endpoint for every rank in `world_size`.
///
/// `receive_timeout` is a total deadline for a call to [`Transport::recv`], including time spent
/// receiving and retaining messages with other tags.
pub fn create_in_memory_world(
    world_size: usize,
    receive_timeout: Duration,
) -> Result<Vec<InMemoryTransport>> {
    if world_size == 0 {
        return Err(CollectivesError::InvalidWorldSize);
    }

    let mut sender_rows: Vec<Vec<Option<mpsc::Sender<MessageFrame>>>> = (0..world_size)
        .map(|_| (0..world_size).map(|_| None).collect())
        .collect();
    let mut receiver_rows: Vec<Vec<Option<mpsc::Receiver<MessageFrame>>>> = (0..world_size)
        .map(|_| (0..world_size).map(|_| None).collect())
        .collect();

    for source in 0..world_size {
        for destination in 0..world_size {
            if source == destination {
                continue;
            }
            let (sender, receiver) = mpsc::channel();
            sender_rows[source][destination] = Some(sender);
            receiver_rows[destination][source] = Some(receiver);
        }
    }

    sender_rows
        .into_iter()
        .zip(receiver_rows)
        .enumerate()
        .map(|(global_rank, (senders, receivers))| {
            Ok(InMemoryTransport {
                rank: Rank::new(global_rank, world_size)?,
                senders,
                receivers: receivers
                    .into_iter()
                    .map(|receiver| receiver.map(Mutex::new))
                    .collect(),
                pending: Mutex::new(VecDeque::new()),
                receive_timeout,
            })
        })
        .collect()
}

impl Transport for InMemoryTransport {
    fn rank(&self) -> Rank {
        self.rank
    }

    fn send(&self, destination: usize, tag: MessageTag, packet: TensorPacket) -> Result<()> {
        self.rank.validate_peer(destination)?;
        let sender = self.senders[destination]
            .as_ref()
            .expect("validated distinct peers always have a sender");
        sender
            .send(MessageFrame {
                source: self.rank.global_rank(),
                destination,
                tag,
                packet,
            })
            .map_err(|_| CollectivesError::SendDisconnected {
                rank: self.rank.global_rank(),
                destination,
                tag,
            })
    }

    fn recv(&self, source: usize, tag: MessageTag) -> Result<TensorPacket> {
        self.rank.validate_peer(source)?;
        if let Some(packet) = self.take_pending(source, tag)? {
            return Ok(packet);
        }

        let started = Instant::now();
        let receiver = self.receivers[source]
            .as_ref()
            .expect("validated distinct peers always have a receiver")
            .lock()
            .map_err(|_| CollectivesError::Synchronization {
                rank: self.rank.global_rank(),
            })?;

        loop {
            let remaining = self.receive_timeout.checked_sub(started.elapsed()).ok_or(
                CollectivesError::ReceiveTimeout {
                    rank: self.rank.global_rank(),
                    source_rank: source,
                    tag,
                    timeout: self.receive_timeout,
                },
            )?;
            let frame = match receiver.recv_timeout(remaining) {
                Ok(frame) => frame,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    return Err(CollectivesError::ReceiveTimeout {
                        rank: self.rank.global_rank(),
                        source_rank: source,
                        tag,
                        timeout: self.receive_timeout,
                    });
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(CollectivesError::ReceiveDisconnected {
                        rank: self.rank.global_rank(),
                        source_rank: source,
                        tag,
                    });
                }
            };

            debug_assert_eq!(frame.source, source);
            debug_assert_eq!(frame.destination, self.rank.global_rank());
            if frame.tag == tag {
                return Ok(frame.packet);
            }
            self.pending
                .lock()
                .map_err(|_| CollectivesError::Synchronization {
                    rank: self.rank.global_rank(),
                })?
                .push_back(frame);
        }
    }
}

impl InMemoryTransport {
    fn take_pending(&self, source: usize, tag: MessageTag) -> Result<Option<TensorPacket>> {
        let mut pending = self
            .pending
            .lock()
            .map_err(|_| CollectivesError::Synchronization {
                rank: self.rank.global_rank(),
            })?;
        let Some(index) = pending
            .iter()
            .position(|frame| frame.source == source && frame.tag == tag)
        else {
            return Ok(None);
        };
        Ok(pending.remove(index).map(|frame| frame.packet))
    }
}
