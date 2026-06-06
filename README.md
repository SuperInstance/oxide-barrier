# oxide-barrier

Synchronization barriers for GPU kernel phases with ternary arrival states. {+1=all_arrived, 0=some_waiting, -1=timeout}. Counting, cyclic, and phase barriers.

## Overview

# oxide-barrier

Synchronization barriers for GPU kernel phases with ternary arrival states.

## Architecture

This crate sits within the **five-layer Oxide Stack**:

| Layer | Crate | Role |
|-------|-------|------|
| 1 | open-parallel | Async runtime (tokio fork) |
| 2 | pincher | "Vector DB as runtime, LLM as compiler" |
| 3 | flux-core | Bytecode VM + A2A agent protocol |
| 4 | cuda-oxide | Flux→MIR→Pliron→NVVM→PTX compiler |
| 5 | cudaclaw | Persistent GPU kernels, warp consensus, SmartCRDT |

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Stats

| Metric | Value |
|--------|-------|
| Tests | 8 |
| Lines of Code | 171 |
| Public API Surface | 20 items |
| License | Apache-2.0 |

## Installation

```toml
[dependencies]
oxide-barrier = "0.1.0"
```

## Usage

```rust
use oxide_barrier::*;
// See src/lib.rs tests for complete working examples
```

### Key Types

```
- pub enum ArrivalState { AllArrived = 1, SomeWaiting = 0, Timeout = -1 }
- pub struct CountingBarrier {
    pub fn new(parties: usize) -> Self {
    pub fn arrive(&mut self) -> ArrivalState {
    pub fn timeout(&mut self) -> ArrivalState {
    pub fn state(&self) -> ArrivalState {
    pub fn generation(&self) -> u64 { self.generation }
    pub fn total_waits(&self) -> u64 { self.total_waits }
    pub fn waiting(&self) -> usize { self.arrived }
- pub struct CyclicBarrier {
```

## Design Philosophy

This crate uses **ternary algebra** (Z₃) where every value is {-1, 0, +1}:

- **+1** → positive signal (healthy, allocated, converged, ready)
- **0** → neutral (pending, balanced, monitoring, degraded)
- **-1** → negative signal (failed, free, diverged, overloaded)

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary neural networks at 60% less power
2. **GPU warp voting** — hardware ballot instructions return ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity (what goes in must come out)

## Testing

```bash
git clone https://github.com/SuperInstance/oxide-barrier.git
cd oxide-barrier
cargo test
```

## License

Apache-2.0
