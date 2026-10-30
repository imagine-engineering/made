use std::{hint::black_box, time::{Duration, Instant}};
#[inline(never)]
fn leaf(seed: u64) -> u64 {
    (0..10000).fold(seed, |value, next| black_box(value.wrapping_mul(31).wrapping_add(next)))
}
#[inline(never)]
fn branch(seed: u64) -> u64 { leaf(seed) }
fn main() {
    let started = Instant::now();
    let mut value = 7;
    while started.elapsed() < Duration::from_millis(300) { value = branch(value); }
    black_box(value);
}
