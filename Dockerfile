FROM rust:1.85-bookworm AS builder

WORKDIR /workspace
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/workspace/target,sharing=locked \
    cargo build --release --locked -p dlir-cli \
    && cp /workspace/target/release/dlir /tmp/dlir

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /tmp/dlir /usr/local/bin/dlir

ENTRYPOINT ["/usr/local/bin/dlir"]
