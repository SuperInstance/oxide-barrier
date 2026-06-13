# Oxide Barrier

**Oxide Barrier** provides GPU synchronization barriers with ternary arrival states — `+1` (all arrived), `0` (some waiting), `-1` (timeout) — enabling coordinated multi-kernel execution phases on GPU hardware.

## Why It Matters

GPU compute pipelines require explicit synchronization between kernel phases. Without barriers, a downstream kernel may read uninitialized memory from an upstream kernel that hasn't finished writing. CPU-side `cudaDeviceSynchronize()` is too coarse — it stalls the entire device. Oxide Barrier provides fine-grained, phase-level barriers with timeout detection, enabling robust error recovery in production GPU workloads. The ternary arrival state maps directly to the SuperInstance conservation framework.

## How It Works

### Counting Barrier

A `CountingBarrier` blocks until N parties have arrived:

```
arrive():
  arrived += 1
  if arrived >= parties:
    arrived = 0
    generation += 1
    return AllArrived (+1)
  else:
    return SomeWaiting (0)

timeout():
  arrived = 0
  generation += 1
  return Timeout (-1)
```

Operations: **O(1)**. The `generation` counter enables detection of stale waits — if a thread's expected generation differs from the current generation, another phase has already completed.

### Cyclic Barrier

A `CyclicBarrier` resets after all parties arrive, enabling repeated use across multiple kernel phases:

```
Phase 1: arrive() × N → AllArrived → reset
Phase 2: arrive() × N → AllArrived → reset
...
```

The barrier can be reused indefinitely without reallocation.

### Phase Barrier

A `PhaseBarrier` tracks a monotonically increasing phase counter:

```
phase_0 → phase_1 → phase_2 → ...
```

Each phase has its own arrival state. This enables pipelined execution: while phase N's barrier waits, phase N-1's data can be consumed.

### Ternary State Mapping

```
+1 (AllArrived) → phase complete, safe to proceed
 0 (SomeWaiting) → in progress, continue waiting
-1 (Timeout)     → failure, trigger recovery
```

This maps to the γ + η = C framework: γ = proceed (+1), neutral = wait (0), η = abort (-1).

## Quick Start

```rust
use oxide_barrier::{CountingBarrier, ArrivalState};

let mut barrier = CountingBarrier::new(4); // 4 parties
for _ in 0..3 {
    assert_eq!(barrier.arrive(), ArrivalState::SomeWaiting);
}
assert_eq!(barrier.arrive(), ArrivalState::AllArrived);

// Timeout scenario
let mut b2 = CountingBarrier::new(4);
b2.arrive();
assert_eq!(b2.timeout(), ArrivalState::Timeout);
```

## API

| Type | Description |
|------|-------------|
| `ArrivalState` | `AllArrived (+1)`, `SomeWaiting (0)`, `Timeout (-1)` |
| `CountingBarrier` | One-shot barrier for N parties |
| `CyclicBarrier` | Resettable barrier for repeated phases |
| `PhaseBarrier` | Monotonic phase tracking with per-phase state |

Key methods: `arrive()`, `timeout()`, `state()`, `generation()`, `total_waits()`.

## Architecture Notes

Oxide Barrier is part of the oxide-* GPU infrastructure stack in SuperInstance. In γ + η = C, barriers control γ (growth — synchronizing computation phases) and η (avoidance — timeouts that prevent deadlocks from blocking the entire system). The `oxide-ring` crate uses these barriers for event buffer synchronization, and `oxide-epoch` uses them for memory reclamation phase coordination.

See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md) for the GPU stack architecture.

## References

1. NVIDIA (2024). *CUDA C++ Programming Guide*. Chapter 7: "Memory Synchronization."
2. Herlihy, M. & Shavit, N. (2012). *The Art of Multiprocessor Programming*, 2nd ed. MIT Press. Chapter 17: Barriers.
3. Hoefler, T. et al. (2014). "Energy-efficient credit-based cooperative barrier." *IEEE Parallel and Distributed Processing Symposium*.

## License

MIT
