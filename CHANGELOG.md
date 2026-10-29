# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- The graph runtime: operator contract and static operator table, shape inference and constant evaluation per set of input shapes, execution plans with liveness-based buffer reuse, thread-safe sessions with fast and exact modes, and the `Model` happy path. Operators so far: Add, Mul, Relu, MatMul, Gemm, Reshape, Identity, Constant.
- The bar runner can time this library (`--side ours --mode fast|exact`).

- The solution skeleton: library, tests, NativeAOT and browser-wasm smoke apps, micro-benchmarks and a fuzz harness.
