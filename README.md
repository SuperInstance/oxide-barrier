# Oxide Barrier

**Oxide Barrier** provides synchronization barriers for GPU kernel phases with ternary arrival states — `+1 (AllArrived)`, `0 (SomeWaiting)`, `-1 (Timeout)` — featuring counting barriers, cyclic barriers with maximum generations, and multi-phase barriers with per-phase party counts.

## Why It Matters

GPU computation involves thousands of threads that must synchronize at defined points: all threads must finish phase A before any thread starts phase B. Standard barriers (C++ `std::barrier`, Java `CyclicBarrier`) are binary — either waiting or released. Oxide Barrier adds a third state: Timeout, indicating that not all threads arrived within the deadline and the barrier was force-released. This ternary state enables more nuanced synchronization: kernel code can branch on timeout (skip computation, use fallback, or abort) rather than deadlock indefinitely.

## How It Works

### Counting Barrier

```
CountingBarrier { parties: N, arrived: 0, generation: 0 }

arrive() → ArrivalState:
    arrived += 1
    if arrived >= parties:
        arrived = 0
        generation += 1
        return AllArrived (+1)
    else:
        return SomeWaiting (0)

timeout() → ArrivalState:
    arrived = 0
    generation += 1
    return Timeout (-1)
```

`arrive()`: **O(1)** (increment + compare). `timeout()`: **O(1)** (reset + increment).

### Cyclic Barrier

Wraps `CountingBarrier` with a maximum cycle count:

```
CyclicBarrier { inner: CountingBarrier, max_cycles: N }

arrive() → ArrivalState:
    state = inner.arrive()
    if inner.generation() > max_cycles:
        return Timeout (-1)   // exhausted
    return state

is_complete() → inner.generation() >= max_cycles
```

Use case: limit GPU kernel iterations. After N barrier cycles, force termination to prevent infinite loops. Cost: **O(1)** per arrive.

### Phase Barrier

Multiple barriers sequenced as phases:

```
PhaseBarrier { phases: [CountingBarrier; P], current_phase: 0 }

arrive() → (phase_index, ArrivalState):
    state = phases[current_phase].arrive()
    if state == AllArrived && current_phase < P-1:
        current_phase += 1     // advance to next phase
    return (current_phase, state)

is_complete() → last_phase released
```

Example: Phase 0 (4 threads compute), Phase 1 (2 threads reduce), Phase 2 (1 thread writeback). Per-phase party count varies. Cost: **O(1)** per arrive.

### Generation Tracking

Each barrier release increments `generation`:
- Enables detection of stale threads (thread arrives at generation 3 but barrier is at generation 5)
- `total_waits()` counts successful barrier releases (not individual arrivals)
- Useful for debugging: if `total_waits × parties ≠ total_arrivals`, some threads skipped barriers

### Ternary State Diagram

```
    arrive()           arrive()           arrive()
     (1/N)              (2/N)              (N/N)
SomeWaiting ──→ SomeWaiting ──→ ... ──→ AllArrived
  (0)             (0)                    (+1)
                                           │
                                       timeout()
                                           ↓
                                      Timeout (-1)
```

## Quick Start

```rust
use oxide_barrier::{CountingBarrier, CyclicBarrier, PhaseBarrier, ArrivalState};

// Counting barrier: 3 threads
let mut barrier = CountingBarrier::new(3);
assert_eq!(barrier.arrive(), ArrivalState::SomeWaiting);
assert_eq!(barrier.arrive(), ArrivalState::SomeWaiting);
assert_eq!(barrier.arrive(), ArrivalState::AllArrived);
assert_eq!(barrier.generation(), 1);

// Cyclic barrier: 2 threads, max 3 cycles
let mut cyclic = CyclicBarrier::new(2, 3);
for _ in 0..3 { cyclic.arrive(); cyclic.arrive(); }
assert!(cyclic.is_complete());

// Phase barrier: phase 0 (2 parties), phase 1 (3 parties)
let mut phased = PhaseBarrier::new(vec![2, 3]);
phased.arrive(); phased.arrive(); // phase 0 complete
assert_eq!(phased.current_phase(), 1);
```

## API

| Type | Methods | Description |
|------|---------|-------------|
| `CountingBarrier` | `new(parties)`, `arrive() → ArrivalState`, `timeout()`, `state()`, `generation()`, `waiting()`, `total_waits()` | Basic N-party barrier |
| `CyclicBarrier` | `new(parties, max_cycles)`, `arrive()`, `is_complete()`, `cycle()` | Bounded-lifetime barrier |
| `PhaseBarrier` | `new(phase_parties: Vec<usize>)`, `arrive() → (usize, ArrivalState)`, `current_phase()`, `total_phases()`, `is_complete()` | Multi-phase sequenced barriers |
| `ArrivalState` | `AllArrived (+1)`, `SomeWaiting (0)`, `Timeout (-1)` | Ternary result enum |

## Architecture Notes

Oxide Barrier provides GPU kernel synchronization primitives for SuperInstance. In γ + η = C, AllArrived (+1) represents γ (growth — all threads completed their computation, enabling progress), Timeout (-1) represents η (avoidance — deadlock prevention by force-releasing stuck threads), and SomeWaiting (0) is the neutral state where the outcome is undecided. The generation counter enforces C: each generation is one complete cycle, and stale threads are detected by generation mismatch. Integrates with `oxide-ring` for event logging during synchronization and `oxide-epoch` for epoch-based memory management.

See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md) for GPU synchronization architecture.

## References

1. NVIDIA Corporation (2024). *CUDA C++ Programming Guide: Cooperative Groups*. Section 7.
2. Herlihy, M. & Shavit, N. (2012). *The Art of Multiprocessor Programming*, revised. Morgan Kaufmann. Chapter 3: Synchronization.
3. Mellor-Crummey, J. M. & Scott, M. L. (1991). "Algorithms for Scalable Synchronization on Shared-Memory Multiprocessors." *ACM TOCS*, 9(1), 21–65.

## License

Apache-2.0
