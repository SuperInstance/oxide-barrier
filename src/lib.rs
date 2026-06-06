//! # oxide-barrier
//!
//! Synchronization barriers for GPU kernel phases with ternary arrival states.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrivalState { AllArrived = 1, SomeWaiting = 0, Timeout = -1 }

pub struct CountingBarrier {
    parties: usize,
    arrived: usize,
    generation: u64,
    total_waits: u64,
}

impl CountingBarrier {
    pub fn new(parties: usize) -> Self {
        Self { parties, arrived: 0, generation: 0, total_waits: 0 }
    }

    pub fn arrive(&mut self) -> ArrivalState {
        self.arrived += 1;
        if self.arrived >= self.parties {
            self.arrived = 0;
            self.generation += 1;
            self.total_waits += 1;
            ArrivalState::AllArrived
        } else {
            ArrivalState::SomeWaiting
        }
    }

    pub fn timeout(&mut self) -> ArrivalState {
        self.arrived = 0;
        self.generation += 1;
        ArrivalState::Timeout
    }

    pub fn state(&self) -> ArrivalState {
        if self.arrived == 0 { ArrivalState::AllArrived }
        else if self.arrived < self.parties { ArrivalState::SomeWaiting }
        else { ArrivalState::AllArrived }
    }

    pub fn generation(&self) -> u64 { self.generation }
    pub fn total_waits(&self) -> u64 { self.total_waits }
    pub fn waiting(&self) -> usize { self.arrived }
}

pub struct CyclicBarrier {
    inner: CountingBarrier,
    max_cycles: u64,
}

impl CyclicBarrier {
    pub fn new(parties: usize, max_cycles: u64) -> Self {
        Self { inner: CountingBarrier::new(parties), max_cycles }
    }

    pub fn arrive(&mut self) -> ArrivalState {
        let state = self.inner.arrive();
        if self.inner.generation() > self.max_cycles {
            ArrivalState::Timeout
        } else {
            state
        }
    }

    pub fn is_complete(&self) -> bool { self.inner.generation() >= self.max_cycles }
    pub fn cycle(&self) -> u64 { self.inner.generation() }
}

pub struct PhaseBarrier {
    phases: Vec<CountingBarrier>,
    current_phase: usize,
}

impl PhaseBarrier {
    pub fn new(phase_parties: Vec<usize>) -> Self {
        let phases = phase_parties.into_iter().map(CountingBarrier::new).collect();
        Self { phases, current_phase: 0 }
    }

    pub fn arrive(&mut self) -> (usize, ArrivalState) {
        if self.current_phase >= self.phases.len() {
            return (self.current_phase, ArrivalState::AllArrived);
        }
        let state = self.phases[self.current_phase].arrive();
        if state == ArrivalState::AllArrived && self.current_phase < self.phases.len() - 1 {
            self.current_phase += 1;
        }
        (self.current_phase, state)
    }

    pub fn current_phase(&self) -> usize { self.current_phase }
    pub fn total_phases(&self) -> usize { self.phases.len() }
    pub fn is_complete(&self) -> bool { self.current_phase >= self.phases.len() - 1
        && self.phases.last().map(|p| p.arrived == 0).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counting_all_arrive() {
        let mut b = CountingBarrier::new(3);
        assert_eq!(b.arrive(), ArrivalState::SomeWaiting);
        assert_eq!(b.arrive(), ArrivalState::SomeWaiting);
        assert_eq!(b.arrive(), ArrivalState::AllArrived);
    }

    #[test]
    fn test_counting_generation() {
        let mut b = CountingBarrier::new(2);
        b.arrive(); b.arrive(); // gen 1
        b.arrive(); b.arrive(); // gen 2
        assert_eq!(b.generation(), 2);
        assert_eq!(b.total_waits(), 2);
    }

    #[test]
    fn test_timeout() {
        let mut b = CountingBarrier::new(3);
        b.arrive();
        assert_eq!(b.timeout(), ArrivalState::Timeout);
        assert_eq!(b.waiting(), 0);
    }

    #[test]
    fn test_cyclic_complete() {
        let mut b = CyclicBarrier::new(2, 3);
        for _ in 0..3 {
            b.arrive(); b.arrive();
        }
        assert!(b.is_complete());
        assert_eq!(b.cycle(), 3);
    }

    #[test]
    fn test_cyclic_not_complete() {
        let mut b = CyclicBarrier::new(2, 3);
        b.arrive(); b.arrive();
        b.arrive(); b.arrive();
        assert!(!b.is_complete());
    }

    #[test]
    fn test_phase_advance() {
        let mut pb = PhaseBarrier::new(vec![2, 3]);
        pb.arrive(); // phase 0, 1/2
        assert_eq!(pb.arrive(), (0, ArrivalState::AllArrived)); // phase 0 done
        assert_eq!(pb.current_phase(), 1);
    }

    #[test]
    fn test_phase_complete() {
        let mut pb = PhaseBarrier::new(vec![2, 2]);
        pb.arrive(); pb.arrive(); // phase 0
        assert_eq!(pb.current_phase(), 1);
        pb.arrive(); pb.arrive(); // phase 1
        assert!(pb.is_complete());
    }

    #[test]
    fn test_phase_total() {
        let pb = PhaseBarrier::new(vec![2, 3, 1]);
        assert_eq!(pb.total_phases(), 3);
    }
}
