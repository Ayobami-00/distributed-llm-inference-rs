//! Versioned TCP rendezvous, full-mesh connection establishment, tensor/control transport, and
//! reusable centralized barriers.

use crate::{
    BarrierTransport, CollectivesError, ControlPacket, ControlTransport, MessageTag, Rank, Result,
    TensorPacket, Transport,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::VecDeque,
    io::{BufReader, BufWriter, Cursor, Read, Write},
    net::{TcpListener, TcpStream, ToSocketAddrs},
    sync::Mutex,
    thread,
    time::{Duration, Instant},
};

/// Wire-protocol version understood by this release.
pub const PROTOCOL_VERSION: u16 = 2;
/// Default upper bound for one encoded tensor frame.
pub const DEFAULT_MAX_TENSOR_BYTES: usize = 256 * 1024 * 1024;

const CONTROL_FRAME_LIMIT: usize = 64 * 1024;
const WIRE_MAGIC: [u8; 4] = *b"DLIR";
const TENSOR_KIND: u8 = 1;
const BARRIER_ARRIVE_KIND: u8 = 2;
const BARRIER_RELEASE_KIND: u8 = 3;
const APPLICATION_CONTROL_KIND: u8 = 4;

/// One rank and its network-visible data address.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerInfo {
    /// Global rank in the rendezvous world.
    pub rank: usize,
    /// TCP address advertised to the other ranks.
    pub address: String,
}

/// Configuration required to join one TCP rank to a rendezvous world.
#[derive(Debug, Clone)]
pub struct TcpTransportConfig {
    /// Validated rank identity and world size.
    pub rank: Rank,
    /// Identifier separating concurrent or stale rendezvous worlds.
    pub run_id: String,
    /// Address every rank uses to contact the rank-0 rendezvous server.
    pub rendezvous_addr: String,
    /// Rank-0-only bind address for the rendezvous listener.
    pub rendezvous_bind_addr: Option<String>,
    /// Local bind address for this rank's peer listener.
    pub listen_addr: String,
    /// Peer-listener address reachable from the other ranks.
    pub advertise_addr: String,
    /// Total deadline for rendezvous and full-mesh establishment.
    pub startup_timeout: Duration,
    /// Total deadline for one receive or barrier operation.
    pub operation_timeout: Duration,
    /// Maximum encoded tensor frame accepted or emitted.
    pub max_tensor_bytes: usize,
}

impl TcpTransportConfig {
    fn validate(&self) -> Result<()> {
        if self.run_id.is_empty()
            || self.run_id.len() > 64
            || !self
                .run_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(CollectivesError::Rendezvous(
                "run ID must contain 1-64 ASCII letters, digits, '-' or '_'".to_owned(),
            ));
        }
        for (name, value) in [
            ("rendezvous", self.rendezvous_addr.as_str()),
            ("listen", self.listen_addr.as_str()),
            ("advertise", self.advertise_addr.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(CollectivesError::Rendezvous(format!(
                    "{name} address cannot be empty"
                )));
            }
        }
        if self.rank.global_rank() == 0 && self.rendezvous_bind_addr.is_none() {
            return Err(CollectivesError::Rendezvous(
                "rank 0 requires a rendezvous bind address".to_owned(),
            ));
        }
        if self.startup_timeout.is_zero() || self.operation_timeout.is_zero() {
            return Err(CollectivesError::Rendezvous(
                "startup and operation timeouts must be positive".to_owned(),
            ));
        }
        if self.max_tensor_bytes < 64 {
            return Err(CollectivesError::Protocol(
                "maximum tensor frame must be at least 64 bytes".to_owned(),
            ));
        }
        Ok(())
    }
}

/// One rank endpoint backed by persistent peer-to-peer TCP streams.
pub struct TcpTransport {
    rank: Rank,
    peers: Vec<PeerInfo>,
    readers: Vec<Option<Mutex<BufReader<TcpStream>>>>,
    writers: Vec<Option<Mutex<BufWriter<TcpStream>>>>,
    pending: Vec<Mutex<VecDeque<WireFrame>>>,
    operation_timeout: Duration,
    max_tensor_bytes: usize,
    barrier_generation: Mutex<u64>,
}

impl TcpTransport {
    /// Joins rendezvous, establishes the complete peer mesh, and returns a ready endpoint.
    pub fn connect(config: TcpTransportConfig) -> Result<Self> {
        config.validate()?;
        let deadline = Instant::now() + config.startup_timeout;
        let data_listener = bind_listener(&config.listen_addr, "bind peer listener")?;
        data_listener
            .set_nonblocking(true)
            .map_err(|source| io_error("configure peer listener", source))?;

        let rendezvous_handle = if config.rank.global_rank() == 0 {
            let bind = config
                .rendezvous_bind_addr
                .as_deref()
                .expect("validated rank 0 bind address");
            let listener = bind_listener(bind, "bind rendezvous listener")?;
            listener
                .set_nonblocking(true)
                .map_err(|source| io_error("configure rendezvous listener", source))?;
            let run_id = config.run_id.clone();
            let world_size = config.rank.world_size();
            Some(thread::spawn(move || {
                run_rendezvous_server(listener, &run_id, world_size, deadline)
            }))
        } else {
            None
        };

        let peers = register_with_rendezvous(&config, deadline)?;
        if let Some(handle) = rendezvous_handle {
            match handle.join() {
                Ok(result) => result?,
                Err(_) => {
                    return Err(CollectivesError::Rendezvous(
                        "rank-0 rendezvous thread panicked".to_owned(),
                    ));
                }
            }
        }
        let streams = establish_full_mesh(&config, &peers, data_listener, deadline)?;
        let world_size = config.rank.world_size();
        let mut readers = empty_options(world_size);
        let mut writers = empty_options(world_size);
        for (peer, stream) in streams {
            stream
                .set_nodelay(true)
                .map_err(|source| io_error("enable TCP_NODELAY", source))?;
            let reader = stream
                .try_clone()
                .map_err(|source| io_error("clone peer stream", source))?;
            readers[peer] = Some(Mutex::new(BufReader::new(reader)));
            writers[peer] = Some(Mutex::new(BufWriter::new(stream)));
        }
        Ok(Self {
            rank: config.rank,
            peers,
            readers,
            writers,
            pending: (0..world_size)
                .map(|_| Mutex::new(VecDeque::new()))
                .collect(),
            operation_timeout: config.operation_timeout,
            max_tensor_bytes: config.max_tensor_bytes,
            barrier_generation: Mutex::new(0),
        })
    }

    /// Returns the ordered rendezvous peer table.
    pub fn peers(&self) -> &[PeerInfo] {
        &self.peers
    }

    fn send_frame(&self, destination: usize, frame: &WireFrame, deadline: Instant) -> Result<()> {
        self.rank.validate_peer(destination)?;
        let writer = self.writers[destination]
            .as_ref()
            .expect("connected distinct peer has writer")
            .lock()
            .map_err(|_| CollectivesError::Synchronization {
                rank: self.rank.global_rank(),
            })?;
        write_wire(writer, frame, deadline, self.max_tensor_bytes)
    }

    fn read_from(&self, source: usize, deadline: Instant) -> Result<WireFrame> {
        self.rank.validate_peer(source)?;
        let reader = self.readers[source]
            .as_ref()
            .expect("connected distinct peer has reader")
            .lock()
            .map_err(|_| CollectivesError::Synchronization {
                rank: self.rank.global_rank(),
            })?;
        let frame = read_wire(reader, deadline, self.max_tensor_bytes)?;
        if frame.source() != source || frame.destination() != self.rank.global_rank() {
            return Err(CollectivesError::Protocol(format!(
                "rank {} received frame claiming {} -> {} on peer {}",
                self.rank.global_rank(),
                frame.source(),
                frame.destination(),
                source
            )));
        }
        Ok(frame)
    }

    fn take_pending_tensor(&self, source: usize, tag: MessageTag) -> Result<Option<TensorPacket>> {
        let mut pending =
            self.pending[source]
                .lock()
                .map_err(|_| CollectivesError::Synchronization {
                    rank: self.rank.global_rank(),
                })?;
        let Some(index) = pending.iter().position(
            |frame| matches!(frame, WireFrame::Tensor { tag: candidate, .. } if *candidate == tag),
        ) else {
            return Ok(None);
        };
        match pending.remove(index) {
            Some(WireFrame::Tensor { packet, .. }) => Ok(Some(packet)),
            _ => unreachable!("matched tensor frame"),
        }
    }

    fn take_pending_control(
        &self,
        source: usize,
        tag: MessageTag,
    ) -> Result<Option<ControlPacket>> {
        let mut pending =
            self.pending[source]
                .lock()
                .map_err(|_| CollectivesError::Synchronization {
                    rank: self.rank.global_rank(),
                })?;
        let Some(index) = pending.iter().position(
            |frame| matches!(frame, WireFrame::Control { tag: candidate, .. } if *candidate == tag),
        ) else {
            return Ok(None);
        };
        match pending.remove(index) {
            Some(WireFrame::Control { packet, .. }) => Ok(Some(packet)),
            _ => unreachable!("matched control frame"),
        }
    }

    fn recv_barrier(
        &self,
        source: usize,
        expected_arrive: bool,
        generation: u64,
        deadline: Instant,
    ) -> Result<()> {
        loop {
            let pending_frame = {
                let mut pending =
                    self.pending[source]
                        .lock()
                        .map_err(|_| CollectivesError::Synchronization {
                            rank: self.rank.global_rank(),
                        })?;
                pending
                    .iter()
                    .position(|frame| {
                        matches!(
                            frame,
                            WireFrame::BarrierArrive { .. } | WireFrame::BarrierRelease { .. }
                        )
                    })
                    .and_then(|index| pending.remove(index))
            };
            let frame = match pending_frame {
                Some(frame) => frame,
                None => self.read_from(source, deadline).map_err(|error| {
                    if is_timeout_error(&error) {
                        CollectivesError::BarrierTimeout {
                            rank: self.rank.global_rank(),
                            generation,
                            timeout: self.operation_timeout,
                        }
                    } else {
                        error
                    }
                })?,
            };
            match frame {
                WireFrame::Tensor { .. } => {
                    self.pending[source]
                        .lock()
                        .map_err(|_| CollectivesError::Synchronization {
                            rank: self.rank.global_rank(),
                        })?
                        .push_back(frame);
                }
                WireFrame::BarrierArrive {
                    generation: actual, ..
                } if expected_arrive && actual == generation => return Ok(()),
                WireFrame::BarrierRelease {
                    generation: actual, ..
                } if !expected_arrive && actual == generation => return Ok(()),
                other => {
                    return Err(CollectivesError::Protocol(format!(
                        "unexpected barrier frame {other:?}, expected generation {generation}"
                    )));
                }
            }
        }
    }
}

impl Transport for TcpTransport {
    fn rank(&self) -> Rank {
        self.rank
    }

    fn send(&self, destination: usize, tag: MessageTag, packet: TensorPacket) -> Result<()> {
        let frame = WireFrame::Tensor {
            source: self.rank.global_rank(),
            destination,
            tag,
            packet,
        };
        self.send_frame(destination, &frame, Instant::now() + self.operation_timeout)
    }

    fn recv(&self, source: usize, tag: MessageTag) -> Result<TensorPacket> {
        self.rank.validate_peer(source)?;
        if let Some(packet) = self.take_pending_tensor(source, tag)? {
            return Ok(packet);
        }
        let deadline = Instant::now() + self.operation_timeout;
        loop {
            let frame = self.read_from(source, deadline).map_err(|error| {
                if is_timeout_error(&error) {
                    CollectivesError::ReceiveTimeout {
                        rank: self.rank.global_rank(),
                        source_rank: source,
                        tag,
                        timeout: self.operation_timeout,
                    }
                } else if is_disconnect_error(&error) {
                    CollectivesError::ReceiveDisconnected {
                        rank: self.rank.global_rank(),
                        source_rank: source,
                        tag,
                    }
                } else {
                    error
                }
            })?;
            match frame {
                WireFrame::Tensor {
                    tag: actual,
                    packet,
                    ..
                } if actual == tag => return Ok(packet),
                other => self.pending[source]
                    .lock()
                    .map_err(|_| CollectivesError::Synchronization {
                        rank: self.rank.global_rank(),
                    })?
                    .push_back(other),
            }
        }
    }
}

impl ControlTransport for TcpTransport {
    fn send_control(
        &self,
        destination: usize,
        tag: MessageTag,
        packet: ControlPacket,
    ) -> Result<()> {
        self.send_frame(
            destination,
            &WireFrame::Control {
                source: self.rank.global_rank(),
                destination,
                tag,
                packet,
            },
            Instant::now() + self.operation_timeout,
        )
    }

    fn recv_control(&self, source: usize, tag: MessageTag) -> Result<ControlPacket> {
        self.rank.validate_peer(source)?;
        if let Some(packet) = self.take_pending_control(source, tag)? {
            return Ok(packet);
        }
        let deadline = Instant::now() + self.operation_timeout;
        loop {
            let frame = self.read_from(source, deadline).map_err(|error| {
                if is_timeout_error(&error) {
                    CollectivesError::ReceiveTimeout {
                        rank: self.rank.global_rank(),
                        source_rank: source,
                        tag,
                        timeout: self.operation_timeout,
                    }
                } else if is_disconnect_error(&error) {
                    CollectivesError::ReceiveDisconnected {
                        rank: self.rank.global_rank(),
                        source_rank: source,
                        tag,
                    }
                } else {
                    error
                }
            })?;
            match frame {
                WireFrame::Control {
                    tag: actual,
                    packet,
                    ..
                } if actual == tag => return Ok(packet),
                other => self.pending[source]
                    .lock()
                    .map_err(|_| CollectivesError::Synchronization {
                        rank: self.rank.global_rank(),
                    })?
                    .push_back(other),
            }
        }
    }
}

impl BarrierTransport for TcpTransport {
    fn barrier(&self) -> Result<()> {
        let mut generation =
            self.barrier_generation
                .lock()
                .map_err(|_| CollectivesError::Synchronization {
                    rank: self.rank.global_rank(),
                })?;
        let current = *generation;
        let deadline = Instant::now() + self.operation_timeout;
        let rank = self.rank.global_rank();
        if rank == 0 {
            for source in 1..self.rank.world_size() {
                self.recv_barrier(source, true, current, deadline)?;
            }
            for destination in 1..self.rank.world_size() {
                self.send_frame(
                    destination,
                    &WireFrame::BarrierRelease {
                        source: 0,
                        destination,
                        generation: current,
                    },
                    deadline,
                )?;
            }
        } else {
            self.send_frame(
                0,
                &WireFrame::BarrierArrive {
                    source: rank,
                    destination: 0,
                    generation: current,
                },
                deadline,
            )?;
            self.recv_barrier(0, false, current, deadline)?;
        }
        *generation = generation
            .checked_add(1)
            .ok_or_else(|| CollectivesError::Protocol("barrier generation overflow".to_owned()))?;
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Registration {
    protocol_version: u16,
    run_id: String,
    rank: usize,
    world_size: usize,
    advertise_addr: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum RendezvousResponse {
    Ready { peers: Vec<PeerInfo> },
    Error { message: String },
}

#[derive(Debug, Serialize, Deserialize)]
struct PeerHandshake {
    protocol_version: u16,
    run_id: String,
    world_size: usize,
    source: usize,
    destination: usize,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum HandshakeResponse {
    Ready,
    Error { message: String },
}

fn run_rendezvous_server(
    listener: TcpListener,
    run_id: &str,
    world_size: usize,
    deadline: Instant,
) -> Result<()> {
    let mut registrations: Vec<Option<Registration>> = empty_options(world_size);
    let mut clients: Vec<Option<TcpStream>> = empty_options(world_size);
    while registrations.iter().any(Option::is_none) {
        let (mut stream, _) = accept_until(&listener, deadline, "accept rendezvous registration")?;
        let registration: Registration = read_json(&mut stream, deadline)?;
        let error = if registration.protocol_version != PROTOCOL_VERSION {
            Some(format!(
                "rank {} uses protocol {}, expected {}",
                registration.rank, registration.protocol_version, PROTOCOL_VERSION
            ))
        } else if registration.run_id != run_id {
            Some(format!(
                "rank {} supplied a different run ID",
                registration.rank
            ))
        } else if registration.world_size != world_size {
            Some(format!(
                "rank {} supplied world size {}, expected {}",
                registration.rank, registration.world_size, world_size
            ))
        } else if registration.rank >= world_size {
            Some(format!(
                "rank {} is outside world size {world_size}",
                registration.rank
            ))
        } else if registrations[registration.rank].is_some() {
            Some(format!(
                "rank {} registered more than once",
                registration.rank
            ))
        } else if registration.advertise_addr.trim().is_empty() {
            Some(format!(
                "rank {} advertised an empty address",
                registration.rank
            ))
        } else if registrations
            .iter()
            .flatten()
            .any(|existing| existing.advertise_addr == registration.advertise_addr)
        {
            Some(format!(
                "rank {} advertised duplicate address {}",
                registration.rank, registration.advertise_addr
            ))
        } else {
            None
        };
        if let Some(message) = error {
            let _ = write_json(
                &mut stream,
                &RendezvousResponse::Error {
                    message: message.clone(),
                },
                deadline,
            );
            for client in clients.iter_mut().flatten() {
                let _ = write_json(
                    client,
                    &RendezvousResponse::Error {
                        message: message.clone(),
                    },
                    deadline,
                );
            }
            return Err(CollectivesError::Rendezvous(message));
        }
        let rank = registration.rank;
        registrations[rank] = Some(registration);
        clients[rank] = Some(stream);
    }

    let peers = registrations
        .into_iter()
        .enumerate()
        .map(|(rank, registration)| PeerInfo {
            rank,
            address: registration.expect("all ranks registered").advertise_addr,
        })
        .collect::<Vec<_>>();
    for client in clients.iter_mut().flatten() {
        write_json(
            client,
            &RendezvousResponse::Ready {
                peers: peers.clone(),
            },
            deadline,
        )?;
    }
    Ok(())
}

fn register_with_rendezvous(
    config: &TcpTransportConfig,
    deadline: Instant,
) -> Result<Vec<PeerInfo>> {
    let mut stream = connect_until(&config.rendezvous_addr, deadline, "connect rendezvous")?;
    write_json(
        &mut stream,
        &Registration {
            protocol_version: PROTOCOL_VERSION,
            run_id: config.run_id.clone(),
            rank: config.rank.global_rank(),
            world_size: config.rank.world_size(),
            advertise_addr: config.advertise_addr.clone(),
        },
        deadline,
    )?;
    match read_json(&mut stream, deadline)? {
        RendezvousResponse::Ready { peers } => {
            if peers.len() != config.rank.world_size()
                || peers
                    .iter()
                    .enumerate()
                    .any(|(rank, peer)| peer.rank != rank)
            {
                return Err(CollectivesError::Rendezvous(
                    "coordinator returned a malformed peer table".to_owned(),
                ));
            }
            Ok(peers)
        }
        RendezvousResponse::Error { message } => Err(CollectivesError::Rendezvous(message)),
    }
}

fn establish_full_mesh(
    config: &TcpTransportConfig,
    peers: &[PeerInfo],
    listener: TcpListener,
    deadline: Instant,
) -> Result<Vec<(usize, TcpStream)>> {
    let rank = config.rank.global_rank();
    let run_id = config.run_id.clone();
    let world_size = config.rank.world_size();
    let accept_handle = thread::spawn(move || {
        let mut incoming = Vec::with_capacity(rank);
        while incoming.len() < rank {
            let (mut stream, _) = accept_until(&listener, deadline, "accept peer handshake")?;
            let handshake: PeerHandshake = read_json(&mut stream, deadline)?;
            let error = validate_handshake(&handshake, &run_id, world_size, rank)
                .or_else(|| {
                    (handshake.source >= rank)
                        .then(|| format!("rank {rank} accepts only lower-ranked dialers"))
                })
                .or_else(|| {
                    incoming
                        .iter()
                        .any(|(source, _)| *source == handshake.source)
                        .then(|| format!("rank {} connected more than once", handshake.source))
                });
            if let Some(message) = error {
                let _ = write_json(
                    &mut stream,
                    &HandshakeResponse::Error {
                        message: message.clone(),
                    },
                    deadline,
                );
                return Err(CollectivesError::Protocol(message));
            }
            write_json(&mut stream, &HandshakeResponse::Ready, deadline)?;
            incoming.push((handshake.source, stream));
        }
        Ok::<_, CollectivesError>(incoming)
    });

    let mut streams = Vec::with_capacity(world_size.saturating_sub(1));
    for peer in peers.iter().skip(rank + 1) {
        let mut stream = connect_until(&peer.address, deadline, "connect peer")?;
        write_json(
            &mut stream,
            &PeerHandshake {
                protocol_version: PROTOCOL_VERSION,
                run_id: config.run_id.clone(),
                world_size,
                source: rank,
                destination: peer.rank,
            },
            deadline,
        )?;
        match read_json(&mut stream, deadline)? {
            HandshakeResponse::Ready => streams.push((peer.rank, stream)),
            HandshakeResponse::Error { message } => {
                return Err(CollectivesError::Protocol(message));
            }
        }
    }
    match accept_handle.join() {
        Ok(result) => streams.extend(result?),
        Err(_) => {
            return Err(CollectivesError::Protocol(
                "peer accept thread panicked".to_owned(),
            ));
        }
    }
    streams.sort_by_key(|(peer, _)| *peer);
    if streams.len() != world_size.saturating_sub(1) {
        return Err(CollectivesError::Protocol(format!(
            "rank {rank} established {} of {} peer connections",
            streams.len(),
            world_size.saturating_sub(1)
        )));
    }
    Ok(streams)
}

fn validate_handshake(
    handshake: &PeerHandshake,
    run_id: &str,
    world_size: usize,
    destination: usize,
) -> Option<String> {
    if handshake.protocol_version != PROTOCOL_VERSION {
        Some(format!(
            "peer uses protocol {}, expected {}",
            handshake.protocol_version, PROTOCOL_VERSION
        ))
    } else if handshake.run_id != run_id {
        Some("peer supplied a different run ID".to_owned())
    } else if handshake.world_size != world_size {
        Some("peer supplied a different world size".to_owned())
    } else if handshake.destination != destination {
        Some(format!(
            "peer targeted rank {}, accepted by rank {destination}",
            handshake.destination
        ))
    } else if handshake.source >= world_size || handshake.source == destination {
        Some(format!("invalid peer source rank {}", handshake.source))
    } else {
        None
    }
}

#[derive(Debug)]
enum WireFrame {
    Tensor {
        source: usize,
        destination: usize,
        tag: MessageTag,
        packet: TensorPacket,
    },
    Control {
        source: usize,
        destination: usize,
        tag: MessageTag,
        packet: ControlPacket,
    },
    BarrierArrive {
        source: usize,
        destination: usize,
        generation: u64,
    },
    BarrierRelease {
        source: usize,
        destination: usize,
        generation: u64,
    },
}

impl WireFrame {
    fn source(&self) -> usize {
        match self {
            Self::Tensor { source, .. }
            | Self::Control { source, .. }
            | Self::BarrierArrive { source, .. }
            | Self::BarrierRelease { source, .. } => *source,
        }
    }

    fn destination(&self) -> usize {
        match self {
            Self::Tensor { destination, .. }
            | Self::Control { destination, .. }
            | Self::BarrierArrive { destination, .. }
            | Self::BarrierRelease { destination, .. } => *destination,
        }
    }
}

fn encode_wire(frame: &WireFrame, maximum: usize) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    body.extend_from_slice(&WIRE_MAGIC);
    body.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
    let kind = match frame {
        WireFrame::Tensor { .. } => TENSOR_KIND,
        WireFrame::Control { .. } => APPLICATION_CONTROL_KIND,
        WireFrame::BarrierArrive { .. } => BARRIER_ARRIVE_KIND,
        WireFrame::BarrierRelease { .. } => BARRIER_RELEASE_KIND,
    };
    body.push(kind);
    body.push(0);
    body.extend_from_slice(&to_u32(frame.source(), "source rank")?.to_le_bytes());
    body.extend_from_slice(&to_u32(frame.destination(), "destination rank")?.to_le_bytes());
    match frame {
        WireFrame::Tensor { tag, packet, .. } => {
            body.extend_from_slice(&tag.0.to_le_bytes());
            body.extend_from_slice(&to_u32(packet.shape().len(), "dimension count")?.to_le_bytes());
            body.extend_from_slice(
                &u64::try_from(packet.values().len())
                    .map_err(|_| CollectivesError::Protocol("element count overflow".to_owned()))?
                    .to_le_bytes(),
            );
            for dimension in packet.shape() {
                body.extend_from_slice(
                    &u64::try_from(*dimension)
                        .map_err(|_| {
                            CollectivesError::Protocol("tensor dimension overflow".to_owned())
                        })?
                        .to_le_bytes(),
                );
            }
            for value in packet.values() {
                body.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        WireFrame::Control { tag, packet, .. } => {
            body.extend_from_slice(&tag.0.to_le_bytes());
            body.extend_from_slice(
                &u32::try_from(packet.bytes().len())
                    .map_err(|_| {
                        CollectivesError::Protocol("control payload length overflow".to_owned())
                    })?
                    .to_le_bytes(),
            );
            body.extend_from_slice(packet.bytes());
        }
        WireFrame::BarrierArrive { generation, .. }
        | WireFrame::BarrierRelease { generation, .. } => {
            body.extend_from_slice(&generation.to_le_bytes());
        }
    }
    if body.len() > maximum {
        return Err(CollectivesError::Protocol(format!(
            "encoded frame is {} bytes, maximum is {maximum}",
            body.len()
        )));
    }
    Ok(body)
}

fn decode_wire(body: Vec<u8>) -> Result<WireFrame> {
    let mut cursor = Cursor::new(body.as_slice());
    let mut magic = [0u8; 4];
    cursor
        .read_exact(&mut magic)
        .map_err(|source| io_error("read frame magic", source))?;
    if magic != WIRE_MAGIC {
        return Err(CollectivesError::Protocol("invalid frame magic".to_owned()));
    }
    let version = read_u16(&mut cursor)?;
    if version != PROTOCOL_VERSION {
        return Err(CollectivesError::Protocol(format!(
            "frame protocol {version}, expected {PROTOCOL_VERSION}"
        )));
    }
    let kind = read_u8(&mut cursor)?;
    let _reserved = read_u8(&mut cursor)?;
    let source = read_u32(&mut cursor)? as usize;
    let destination = read_u32(&mut cursor)? as usize;
    let frame = match kind {
        TENSOR_KIND => {
            let tag = MessageTag(read_u64(&mut cursor)?);
            let dimensions = read_u32(&mut cursor)? as usize;
            if dimensions > 64 {
                return Err(CollectivesError::Protocol(format!(
                    "tensor has {dimensions} dimensions; maximum is 64"
                )));
            }
            let elements = usize::try_from(read_u64(&mut cursor)?)
                .map_err(|_| CollectivesError::Protocol("element count overflow".to_owned()))?;
            let mut shape = Vec::with_capacity(dimensions);
            for _ in 0..dimensions {
                shape.push(usize::try_from(read_u64(&mut cursor)?).map_err(|_| {
                    CollectivesError::Protocol("tensor dimension overflow".to_owned())
                })?);
            }
            let remaining = body.len().saturating_sub(cursor.position() as usize);
            let expected_bytes = elements.checked_mul(4).ok_or_else(|| {
                CollectivesError::Protocol("tensor payload byte count overflow".to_owned())
            })?;
            if remaining != expected_bytes {
                return Err(CollectivesError::Protocol(format!(
                    "tensor declares {elements} values but carries {remaining} payload bytes"
                )));
            }
            let mut values = Vec::with_capacity(elements);
            for _ in 0..elements {
                values.push(f32::from_bits(read_u32(&mut cursor)?));
            }
            WireFrame::Tensor {
                source,
                destination,
                tag,
                packet: TensorPacket::new(shape, values)?,
            }
        }
        APPLICATION_CONTROL_KIND => {
            let tag = MessageTag(read_u64(&mut cursor)?);
            let length = read_u32(&mut cursor)? as usize;
            let remaining = body.len().saturating_sub(cursor.position() as usize);
            if remaining != length {
                return Err(CollectivesError::Protocol(format!(
                    "control frame declares {length} bytes but carries {remaining}"
                )));
            }
            let mut bytes = vec![0u8; length];
            cursor
                .read_exact(&mut bytes)
                .map_err(|source| io_error("decode control payload", source))?;
            WireFrame::Control {
                source,
                destination,
                tag,
                packet: ControlPacket::new(bytes)?,
            }
        }
        BARRIER_ARRIVE_KIND => WireFrame::BarrierArrive {
            source,
            destination,
            generation: read_u64(&mut cursor)?,
        },
        BARRIER_RELEASE_KIND => WireFrame::BarrierRelease {
            source,
            destination,
            generation: read_u64(&mut cursor)?,
        },
        other => {
            return Err(CollectivesError::Protocol(format!(
                "unknown frame kind {other}"
            )));
        }
    };
    if cursor.position() as usize != body.len() {
        return Err(CollectivesError::Protocol(
            "frame contains trailing bytes".to_owned(),
        ));
    }
    Ok(frame)
}

fn write_wire(
    mut writer: std::sync::MutexGuard<'_, BufWriter<TcpStream>>,
    frame: &WireFrame,
    deadline: Instant,
    maximum: usize,
) -> Result<()> {
    let body = encode_wire(frame, maximum)?;
    set_write_deadline(writer.get_ref(), deadline)?;
    writer
        .write_all(&(body.len() as u64).to_le_bytes())
        .and_then(|_| writer.write_all(&body))
        .and_then(|_| writer.flush())
        .map_err(|source| io_error("write TCP frame", source))
}

fn read_wire(
    mut reader: std::sync::MutexGuard<'_, BufReader<TcpStream>>,
    deadline: Instant,
    maximum: usize,
) -> Result<WireFrame> {
    set_read_deadline(reader.get_ref(), deadline)?;
    let mut length = [0u8; 8];
    reader
        .read_exact(&mut length)
        .map_err(|source| io_error("read TCP frame length", source))?;
    let length = usize::try_from(u64::from_le_bytes(length))
        .map_err(|_| CollectivesError::Protocol("frame length overflow".to_owned()))?;
    if length > maximum {
        return Err(CollectivesError::Protocol(format!(
            "incoming frame is {length} bytes, maximum is {maximum}"
        )));
    }
    let mut body = vec![0u8; length];
    set_read_deadline(reader.get_ref(), deadline)?;
    reader
        .read_exact(&mut body)
        .map_err(|source| io_error("read TCP frame body", source))?;
    decode_wire(body)
}

fn write_json<T: Serialize>(stream: &mut TcpStream, value: &T, deadline: Instant) -> Result<()> {
    let body = serde_json::to_vec(value)?;
    if body.len() > CONTROL_FRAME_LIMIT {
        return Err(CollectivesError::Protocol(format!(
            "control frame is {} bytes, maximum is {CONTROL_FRAME_LIMIT}",
            body.len()
        )));
    }
    set_write_deadline(stream, deadline)?;
    stream
        .write_all(&(body.len() as u32).to_be_bytes())
        .and_then(|_| stream.write_all(&body))
        .map_err(|source| io_error("write control frame", source))
}

fn read_json<T: DeserializeOwned>(stream: &mut TcpStream, deadline: Instant) -> Result<T> {
    set_read_deadline(stream, deadline)?;
    let mut length = [0u8; 4];
    stream
        .read_exact(&mut length)
        .map_err(|source| io_error("read control frame length", source))?;
    let length = u32::from_be_bytes(length) as usize;
    if length > CONTROL_FRAME_LIMIT {
        return Err(CollectivesError::Protocol(format!(
            "control frame is {length} bytes, maximum is {CONTROL_FRAME_LIMIT}"
        )));
    }
    let mut body = vec![0u8; length];
    set_read_deadline(stream, deadline)?;
    stream
        .read_exact(&mut body)
        .map_err(|source| io_error("read control frame body", source))?;
    Ok(serde_json::from_slice(&body)?)
}

fn bind_listener<A: ToSocketAddrs>(address: A, context: &str) -> Result<TcpListener> {
    TcpListener::bind(address).map_err(|source| io_error(context, source))
}

fn accept_until(
    listener: &TcpListener,
    deadline: Instant,
    context: &str,
) -> Result<(TcpStream, std::net::SocketAddr)> {
    loop {
        match listener.accept() {
            Ok((stream, address)) => {
                stream
                    .set_nonblocking(false)
                    .map_err(|source| io_error("configure accepted TCP stream", source))?;
                return Ok((stream, address));
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(CollectivesError::Rendezvous(format!("{context} timed out")));
                }
                thread::sleep(Duration::from_millis(10));
            }
            Err(source) => return Err(io_error(context, source)),
        }
    }
}

fn connect_until(address: &str, deadline: Instant, context: &str) -> Result<TcpStream> {
    loop {
        match TcpStream::connect(address) {
            Ok(stream) => return Ok(stream),
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::ConnectionRefused
                        | std::io::ErrorKind::TimedOut
                        | std::io::ErrorKind::NotFound
                        | std::io::ErrorKind::AddrNotAvailable
                ) =>
            {
                if Instant::now() >= deadline {
                    return Err(CollectivesError::Rendezvous(format!(
                        "{context} to {address} timed out: {error}"
                    )));
                }
                thread::sleep(Duration::from_millis(25));
            }
            Err(source) => return Err(io_error(format!("{context} to {address}"), source)),
        }
    }
}

fn set_read_deadline(stream: &TcpStream, deadline: Instant) -> Result<()> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Err(CollectivesError::Protocol(
            "TCP read deadline expired".to_owned(),
        ));
    }
    stream
        .set_read_timeout(Some(remaining))
        .map_err(|source| io_error("set TCP read timeout", source))
}

fn set_write_deadline(stream: &TcpStream, deadline: Instant) -> Result<()> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Err(CollectivesError::Protocol(
            "TCP write deadline expired".to_owned(),
        ));
    }
    stream
        .set_write_timeout(Some(remaining))
        .map_err(|source| io_error("set TCP write timeout", source))
}

fn read_u8(cursor: &mut Cursor<&[u8]>) -> Result<u8> {
    let mut bytes = [0u8; 1];
    cursor
        .read_exact(&mut bytes)
        .map_err(|source| io_error("decode u8", source))?;
    Ok(bytes[0])
}

fn read_u16(cursor: &mut Cursor<&[u8]>) -> Result<u16> {
    let mut bytes = [0u8; 2];
    cursor
        .read_exact(&mut bytes)
        .map_err(|source| io_error("decode u16", source))?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32(cursor: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut bytes = [0u8; 4];
    cursor
        .read_exact(&mut bytes)
        .map_err(|source| io_error("decode u32", source))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(cursor: &mut Cursor<&[u8]>) -> Result<u64> {
    let mut bytes = [0u8; 8];
    cursor
        .read_exact(&mut bytes)
        .map_err(|source| io_error("decode u64", source))?;
    Ok(u64::from_le_bytes(bytes))
}

fn to_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| CollectivesError::Protocol(format!("{name} does not fit u32")))
}

fn empty_options<T>(length: usize) -> Vec<Option<T>> {
    (0..length).map(|_| None).collect()
}

fn io_error(context: impl Into<String>, source: std::io::Error) -> CollectivesError {
    CollectivesError::Io {
        context: context.into(),
        source,
    }
}

fn is_timeout_error(error: &CollectivesError) -> bool {
    matches!(
        error,
        CollectivesError::Io { source, .. }
            if matches!(
                source.kind(),
                std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
            )
    ) || matches!(error, CollectivesError::Protocol(message) if message.contains("deadline expired"))
}

fn is_disconnect_error(error: &CollectivesError) -> bool {
    matches!(
        error,
        CollectivesError::Io { source, .. }
            if matches!(
                source.kind(),
                std::io::ErrorKind::UnexpectedEof
                    | std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::BrokenPipe
                    | std::io::ErrorKind::NotConnected
            )
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn wire_tensor_round_trip_preserves_shape_tag_and_values() {
        let frame = WireFrame::Tensor {
            source: 1,
            destination: 0,
            tag: MessageTag(42),
            packet: TensorPacket::new(vec![2, 2], vec![1., 2., 3., 4.]).unwrap(),
        };
        let decoded = decode_wire(encode_wire(&frame, 1024).unwrap()).unwrap();
        match decoded {
            WireFrame::Tensor {
                source,
                destination,
                tag,
                packet,
            } => {
                assert_eq!((source, destination, tag), (1, 0, MessageTag(42)));
                assert_eq!(packet.shape(), &[2, 2]);
                assert_eq!(packet.values(), &[1., 2., 3., 4.]);
            }
            _ => panic!("expected tensor"),
        }
    }

    #[test]
    fn wire_control_round_trip_preserves_peers_tag_and_bytes() {
        let frame = WireFrame::Control {
            source: 1,
            destination: 0,
            tag: MessageTag(77),
            packet: ControlPacket::new(b"continue".to_vec()).unwrap(),
        };
        let decoded = decode_wire(encode_wire(&frame, 1024).unwrap()).unwrap();
        match decoded {
            WireFrame::Control {
                source,
                destination,
                tag,
                packet,
            } => {
                assert_eq!((source, destination, tag), (1, 0, MessageTag(77)));
                assert_eq!(packet.bytes(), b"continue");
            }
            _ => panic!("expected control frame"),
        }
    }

    #[test]
    fn wire_rejects_bad_magic_and_oversized_output() {
        let frame = WireFrame::Tensor {
            source: 0,
            destination: 1,
            tag: MessageTag(1),
            packet: TensorPacket::new(vec![4], vec![1.; 4]).unwrap(),
        };
        assert!(encode_wire(&frame, 8).is_err());
        let mut bytes = encode_wire(&frame, 1024).unwrap();
        bytes[0] = b'X';
        assert!(decode_wire(bytes).is_err());
        let mut truncated = encode_wire(&frame, 1024).unwrap();
        truncated.pop();
        assert!(decode_wire(truncated).is_err());
    }

    #[test]
    fn config_rejects_invalid_identity_and_missing_rank_zero_bind() {
        let base = TcpTransportConfig {
            rank: Rank::new(0, 2).unwrap(),
            run_id: "valid-run".to_owned(),
            rendezvous_addr: "127.0.0.1:1".to_owned(),
            rendezvous_bind_addr: None,
            listen_addr: "127.0.0.1:2".to_owned(),
            advertise_addr: "127.0.0.1:2".to_owned(),
            startup_timeout: Duration::from_millis(10),
            operation_timeout: Duration::from_millis(10),
            max_tensor_bytes: 1024,
        };
        assert!(base.validate().is_err());
        let mut invalid = base;
        invalid.rendezvous_bind_addr = Some("127.0.0.1:1".to_owned());
        invalid.run_id = "not valid!".to_owned();
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn rendezvous_rejects_duplicate_rank_registration() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        let server =
            thread::spawn(move || run_rendezvous_server(listener, "duplicate-test", 2, deadline));
        let registration = Registration {
            protocol_version: PROTOCOL_VERSION,
            run_id: "duplicate-test".to_owned(),
            rank: 0,
            world_size: 2,
            advertise_addr: "127.0.0.1:1000".to_owned(),
        };
        let mut first = connect_until(&address.to_string(), deadline, "test connect").unwrap();
        write_json(&mut first, &registration, deadline).unwrap();
        let mut duplicate = connect_until(&address.to_string(), deadline, "test connect").unwrap();
        write_json(&mut duplicate, &registration, deadline).unwrap();
        assert!(matches!(
            read_json::<RendezvousResponse>(&mut duplicate, deadline).unwrap(),
            RendezvousResponse::Error { .. }
        ));
        assert!(server.join().unwrap().is_err());
    }
}
