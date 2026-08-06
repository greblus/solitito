//! Minimal pseudo-random generator for exercise ordering.
//!
//! No `rand` dependency on purpose: picking the next chord or shuffling four
//! interval steps needs no statistical guarantees and no cryptographic quality.
//! xorshift64* passes the usual smoke tests and fits in twenty lines.
//!
//! Seeded from the system clock, so every run gives a different order.

pub struct Rng {
    state: u64,
}

impl Default for Rng {
    fn default() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x2545_F491_4F6C_DD1D);
        Self::with_seed(seed)
    }
}

impl Rng {
    /// A zero seed would make xorshift produce zeros forever, hence the fallback.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `0..n`. Returns 0 for n == 0 so callers need no special case.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next_u64() % n as u64) as usize }
    }

    /// Uniform in `0..n` but never `current` - the next exercise should differ
    /// from the one just finished. With n == 1 there is no choice, so it returns
    /// `current`.
    pub fn below_excluding(&mut self, n: usize, current: usize) -> usize {
        if n <= 1 { return current.min(n.saturating_sub(1)); }
        // Draw from n-1 slots and skip over `current`; no rejection loop needed.
        let r = self.below(n - 1);
        if r >= current { r + 1 } else { r }
    }

    /// Fisher-Yates, in place.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_same_sequence() {
        let mut a = Rng::with_seed(42);
        let mut b = Rng::with_seed(42);
        for _ in 0..50 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn zero_seed_does_not_stick_at_zero() {
        let mut r = Rng::with_seed(0);
        assert!(r.next_u64() != 0);
        assert!(r.next_u64() != 0);
    }

    #[test]
    fn below_stays_in_range_and_handles_zero() {
        let mut r = Rng::with_seed(7);
        assert_eq!(r.below(0), 0);
        for _ in 0..500 {
            assert!(r.below(5) < 5);
        }
    }

    #[test]
    fn below_excluding_never_returns_current() {
        let mut r = Rng::with_seed(9);
        for cur in 0..4 {
            for _ in 0..200 {
                assert_ne!(r.below_excluding(4, cur), cur);
            }
        }
    }

    #[test]
    fn below_excluding_survives_degenerate_sizes() {
        let mut r = Rng::with_seed(11);
        assert_eq!(r.below_excluding(1, 0), 0);   // one item: no alternative
        assert_eq!(r.below_excluding(0, 0), 0);   // empty list must not panic
    }

    #[test]
    fn shuffle_keeps_every_element() {
        let mut r = Rng::with_seed(3);
        let mut v: Vec<usize> = (0..12).collect();
        r.shuffle(&mut v);
        v.sort();
        assert_eq!(v, (0..12).collect::<Vec<_>>());
    }

    /// A shuffle that always returned the input would pass the test above.
    #[test]
    fn shuffle_actually_reorders() {
        let mut r = Rng::with_seed(5);
        let mut changed = 0;
        for _ in 0..20 {
            let mut v: Vec<usize> = (0..8).collect();
            r.shuffle(&mut v);
            if v != (0..8).collect::<Vec<_>>() { changed += 1; }
        }
        assert!(changed >= 18, "shuffle looks like a no-op ({changed}/20 reordered)");
    }
}
