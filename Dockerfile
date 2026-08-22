FROM rust:1.85-bookworm AS builder

WORKDIR /workspace
COPY . .
RUN cargo build --release --locked -p dlir-cli

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /workspace/target/release/dlir /usr/local/bin/dlir

ENTRYPOINT ["/usr/local/bin/dlir"]
