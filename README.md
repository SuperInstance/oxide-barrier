# oxide-barrier

*Synchronization barriers for GPU kernel phases. Ternary arrival states: which threads have arrived (+1), which are in transit (0), and which haven't started (-1).*

## Why This Exists

GPU barriers are expensive. Every thread in a block must reach the barrier before any can proceed. On NVIDIA hardware, `__syncthreads()` is a hardware instruction — fast but inflexible. For multi-kernel coordination (where different kernels run in sequence and need to synchronize), you need software barriers.

The ternary arrival state tells you *why* a barrier hasn't been satisfied:
- **+1 (Arrived):** Thread reached the barrier. Ready to proceed.
- **0 (InTransit):** Thread is running but hasn't reached the barrier yet.
- **-1 (NotStarted/Errored):** Thread hasn't begun, or it failed.

This information enables smart waiting: spin if most threads are in transit, block if many haven't started, abort if there are errors.

## Architecture

```
Phase 1 Kernel ──→ Barrier ──→ Phase 2 Kernel ──→ Barrier ──→ Phase 3
                    ↑                              ↑
              Check arrival states           Check arrival states
              +1: 192/256 threads arrived
               0:  48/256 in transit
              -1:  16/256 not started
```

### Key Types

- **`Barrier`** — N-thread synchronization point with ternary arrival tracking.
- **`ArrivalState`** — Arrived / InTransit / NotStarted / Errored per thread.
- **`PhaseTracker`** — Coordinate multiple barriers in sequence (phase 1 → 2 → 3 → ...).
- **`BarrierStats`** — Arrival counts by state, estimated wait time, timeout detection.

## Usage

```rust
use oxide_barrier::*;

// 256 threads, 3 phases
let mut tracker = PhaseTracker::new(256, 3);

// Simulate thread arrivals
tracker.arrive(0, ThreadId(42));  // Thread 42 arrives at phase 0
tracker.in_transit(0, ThreadId(100)); // Thread 100 still running

// Check barrier state
let stats = tracker.barrier_stats(0);
println!("Arrived: {}/{}", stats.arrived, stats.total);

// Wait for barrier satisfaction
if tracker.wait(0, timeout_ms) {
    // All arrived — proceed to next phase
    tracker.advance();
} else {
    // Timeout — check which threads are stuck
    let stuck = tracker.not_arrived(0);
}
```

## The Deeper Idea

Barrier arrival states map directly to `agent-sync`'s timing model. An agent that arrives early, on-time, or late has the same dynamics as a GPU thread reaching a barrier. The agent-sync experiment proved that timing beats quality (2.46× advantage) — the same principle applies to barrier design. A barrier that knows arrival states can make smarter decisions about waiting.

The connection to `ternary-fence` is structural: fences are memory-level barriers (ensure writes are visible), while this crate provides execution-level barriers (ensure threads have reached a point). Together they form the complete synchronization story.

## Related Crates

- `oxide-epoch` — Epoch management that coordinates with barrier phases
- `oxide-workflow` — DAG execution that uses barriers between kernel steps
- `ternary-fence` — Memory fences (the hardware-level counterpart)
- `agent-sync` — Agent timing coordination (the same pattern at fleet scale)
