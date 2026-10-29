#![doc = include_str!("../README.md")]
#![deny(missing_docs)]

pub mod read;
pub mod write;

// ESP-IDF has a smaller default to accommodate limited RAM.
const DEFAULT_BUF_SIZE: usize = cfg_select! {
    target_os = "espidf" => 1 << 9,
    _ => 8 << 10,
};

// Shared allocation counter for buffer and reader tests. These tests check
// memory reuse and avoidable allocations that content or capacity checks alone
// cannot detect. This module provides test support rather than test cases.
#[cfg(test)]
mod test {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    // Count only on the calling thread so parallel tests do not affect each
    // other.
    thread_local! {
        static TRACK: Cell<bool> = const { Cell::new(false) };
        static COUNT: Cell<usize> = const { Cell::new(0) };
    }

    struct Allocator;

    // SAFETY: operations forward the caller's pointer/layout to System.
    // The counters use const thread-local Cells and do not allocate.
    unsafe impl GlobalAlloc for Allocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if TRACK.try_with(Cell::get).unwrap_or(false) {
                COUNT.with(|n| n.set(n.get() + 1));
            }
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            if TRACK.try_with(Cell::get).unwrap_or(false) {
                COUNT.with(|n| n.set(n.get() + 1));
            }
            unsafe { System.realloc(ptr, layout, size) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: Allocator = Allocator;

    // Return the result and number of allocation/reallocation calls during the
    // operation on this thread. Calls must not be nested: each resets the
    // counter.
    pub(crate) fn allocations<T>(operation: impl FnOnce() -> T) -> (T, usize) {
        struct Stop;
        impl Drop for Stop {
            fn drop(&mut self) {
                TRACK.with(|track| track.set(false));
            }
        }

        COUNT.with(|count| count.set(0));
        TRACK.with(|track| track.set(true));
        let stop = Stop;
        let result = operation();
        // Stop tracking even when an assertion inside operation unwinds.
        drop(stop);
        (result, COUNT.with(Cell::get))
    }
}
