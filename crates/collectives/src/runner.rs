//! Thread runner for a complete in-memory rank world.

use crate::{
    CollectivesError, Communicator, InMemoryTransport, Result, Transport, create_in_memory_world,
};
use std::{any::Any, sync::Arc, thread, time::Duration};

/// Runs one worker thread per in-memory rank and returns successful results in rank order.
///
/// Every worker receives an exclusive transport endpoint. All workers are joined before the
/// function returns, even if one returns an error or panics.
pub fn run_in_memory<T, F>(
    world_size: usize,
    receive_timeout: Duration,
    worker: F,
) -> Result<Vec<T>>
where
    T: Send + 'static,
    F: Fn(Communicator<InMemoryTransport>) -> Result<T> + Send + Sync + 'static,
{
    let endpoints = create_in_memory_world(world_size, receive_timeout)?;
    let worker = Arc::new(worker);
    let handles = endpoints
        .into_iter()
        .map(|endpoint| {
            let rank = endpoint.rank().global_rank();
            let worker = Arc::clone(&worker);
            (
                rank,
                thread::spawn(move || worker(Communicator::new(endpoint))),
            )
        })
        .collect::<Vec<_>>();

    let mut results: Vec<Option<T>> = (0..world_size).map(|_| None).collect();
    let mut first_error = None;
    for (rank, handle) in handles {
        match handle.join() {
            Ok(Ok(value)) => results[rank] = Some(value),
            Ok(Err(source)) if first_error.is_none() => {
                first_error = Some(CollectivesError::WorkerFailed {
                    rank,
                    source: Box::new(source),
                });
            }
            Err(payload) if first_error.is_none() => {
                first_error = Some(CollectivesError::WorkerPanicked {
                    rank,
                    message: panic_message(payload),
                });
            }
            Ok(Err(_)) | Err(_) => {}
        }
    }

    if let Some(error) = first_error {
        return Err(error);
    }
    Ok(results
        .into_iter()
        .map(|result| result.expect("every successful rank produced a result"))
        .collect())
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_owned()
    }
}
