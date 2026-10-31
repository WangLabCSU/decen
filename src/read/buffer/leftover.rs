use std::ops::Range;
use std::slice::SliceIndex;

use bytes::{Buf, BytesMut};

// buf: [ consumed | retained input ], with buffered <= buf.len().
// Keep the live length directly: empty leftover is the common per-Item test.
#[derive(Debug)]
pub(super) struct LeftOver {
    buf: BytesMut,
    buffered: usize,
}

impl LeftOver {
    #[inline]
    pub(super) fn new() -> Self {
        Self {
            buf: BytesMut::new(),
            buffered: 0,
        }
    }

    #[inline]
    pub(super) fn len(&self) -> usize {
        self.buffered
    }

    #[inline]
    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub(super) fn clear(&mut self) {
        self.buffered = 0;
    }

    #[inline]
    pub(super) fn buffer(&self) -> &[u8] {
        let start = self.buf.len() - self.buffered;
        // SAFETY: buffered <= buf.len() places start within initialized
        // storage. The borrow prevents changes to that storage.
        unsafe { self.buf.get_unchecked(start..) }
    }

    /// Consumes `amount` bytes without checking the input length.
    ///
    /// # Safety
    ///
    /// Requires `amount <= self.len()`.
    #[inline]
    pub(super) unsafe fn consume_uncheck(&mut self, amount: usize) {
        self.buffered -= amount;
    }

    /// Borrows `range` from the current input without consuming bytes.
    ///
    /// # Safety
    ///
    /// `range` must be a valid slice range for `self.buffer()`.
    #[inline]
    pub(super) unsafe fn view_unchecked(
        &self,
        range: impl SliceIndex<[u8], Output = [u8]>,
    ) -> &[u8] {
        // SAFETY: buffer() excludes consumed bytes; the caller guarantees
        // that range is valid within that input slice.
        unsafe { self.buffer().get_unchecked(range) }
    }

    /// Borrows `range` and consumes `consume` bytes, both relative to input
    /// before this call.
    ///
    /// # Safety
    ///
    /// Requires `range.start <= range.end <= self.len()` and
    /// `consume <= self.len()`.
    #[inline]
    pub(super) unsafe fn take_view_unchecked(
        &mut self,
        range: Range<usize>,
        consume: usize,
    ) -> &[u8] {
        let offset = self.buf.len() - self.buffered;
        let start = offset + range.start;
        let end = offset + range.end;
        // SAFETY: the caller bounds consumption by the current input length.
        // Only the live length changes, so the saved selection stays in place.
        unsafe { self.consume_uncheck(consume) };
        // SAFETY: the saved range lies within initialized storage, including
        // any selected bytes that were just consumed.
        unsafe { self.buf.get_unchecked(start..end) }
    }

    /// Transfers the first `at` bytes without checking the input length.
    /// May allocate, but does not copy bytes.
    ///
    /// # Safety
    ///
    /// Requires `at <= self.len()`.
    #[inline]
    pub(super) unsafe fn split_to_unchecked(&mut self, at: usize) -> BytesMut {
        let offset = self.buf.len() - self.buffered;
        // Split before changing the live length, since split_to can allocate.
        let mut bytes = self.buf.split_to(offset + at);
        bytes.advance(offset);
        self.buffered -= at;
        bytes
    }

    #[inline]
    pub(super) fn extend_from_slice(&mut self, other: &[u8]) {
        if self.is_empty() {
            self.buf.clear();
        } else {
            let offset = self.buf.len() - self.buffered;
            if offset != 0 {
                // Discard the consumed prefix before appending. BytesMut's
                // reserve decides when moving live bytes is worth reclaiming
                // its storage.
                self.buf.advance(offset);
            }
        }
        self.buf.extend_from_slice(other);
        self.buffered += other.len();
    }
}
