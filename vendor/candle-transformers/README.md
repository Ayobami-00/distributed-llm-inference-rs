# Candle Transformers Llama oracle

This dev-only package contains Candle Transformers 0.11.0's unmodified `models/llama.rs`
and the minimal support surface it needs. `dlir-runtime` uses it only as an independent
synthetic-logit oracle; it is not linked into the `dlir` release binary.
