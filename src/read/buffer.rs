//! Buffering and frame extraction for the reader.

use std::hint;
use std::io::{self, Read};
use std::ops::{Range, RangeBounds};

use bytes::{Buf, BytesMut};

mod leftover;
mod read_chunk;

use self::leftover::LeftOver;
use self::read_chunk::ReadChunk;
use super::Segments;
use super::segments::bounds;

/// Buffered input that can be borrowed or transferred to a decoder.
///
/// Input may have two consecutive parts. Borrowing a frame across those parts
/// joins the selected bytes; borrowing within one part does not copy.
#[derive(Debug)]
pub struct Buffer {
    leftover: LeftOver,
    chunk: ReadChunk,
}

impl Buffer {
    /// Creates an empty buffer with at least `capacity` bytes for reading.
    /// Zero defers allocation until the first read.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            leftover: LeftOver::new(),
            chunk: ReadChunk::with_capacity(capacity),
        }
    }

    /// Returns the number of buffered bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.leftover.len() + self.chunk.len()
    }

    /// Returns `true` if no input is buffered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn clear(&mut self) {
        self.leftover.clear();
        self.chunk.clear();
    }

    /// Consumes the first `amount` bytes.
    ///
    /// # Panics
    ///
    /// Panics if `amount` exceeds the buffered input length.
    #[inline]
    pub fn consume(&mut self, amount: usize) {
        if amount == 0 {
            return;
        }
        assert!(amount <= self.len(), "cannot consume past buffered input");
        // SAFETY: the assertion bounds consumption by the buffered input.
        unsafe { self.consume_core(amount) };
    }

    /// Consumes `amount` bytes without checking the input length.
    ///
    /// # Safety
    ///
    /// Requires `amount <= self.len()`.
    #[inline]
    unsafe fn consume_unchecked(&mut self, amount: usize) {
        if amount == 0 {
            return;
        }
        unsafe { self.consume_core(amount) };
    }

    #[inline]
    unsafe fn consume_core(&mut self, mut amount: usize) {
        if self.leftover.len() >= amount {
            // SAFETY: this branch checks amount against the live prefix.
            unsafe { self.leftover.consume_uncheck(amount) };
            return;
        }
        amount -= self.leftover.len();
        self.leftover.clear();
        // SAFETY: the caller bounds consumption by the combined input. After
        // consuming the entire leftover, the remainder fits in the chunk.
        unsafe { self.chunk.consume_unchecked(amount) };
    }

    /// Borrows the buffered input without joining its parts.
    #[inline]
    pub fn buffer(&self) -> Segments<&[u8]> {
        match (self.leftover.is_empty(), self.chunk.is_empty()) {
            (true, _) => Segments::Contiguous(self.chunk.buffer()),
            (false, true) => Segments::Contiguous(self.leftover.buffer()),
            (false, false) => Segments::Split(self.leftover.buffer(), self.chunk.buffer()),
        }
    }

    /// Borrows `range` as a slice and consumes `consume` bytes.
    /// Selection and consumption both refer to input before this call.
    ///
    /// May allocate and copy when the selection crosses the two input parts.
    /// The selected bytes may overlap input retained for the next frame.
    ///
    /// # Panics
    ///
    /// Panics if the range or consumption exceeds the buffered input.
    #[inline(always)]
    pub fn take_view(&mut self, range: impl RangeBounds<usize>, consume: usize) -> &[u8] {
        let len = self.len();
        let range = bounds(range, len);
        assert!(consume <= len, "cannot consume past buffered input");
        // SAFETY: bounds() validates the range; the assertion bounds
        // consumption.
        unsafe { self.take_view_unchecked(range, consume) }
    }

    /// Borrows `range`, consuming through its end and another `addition`
    /// bytes. May allocate and copy when the range crosses input parts.
    ///
    /// # Panics
    ///
    /// Panics if the range or its end plus `addition` exceeds buffered input.
    #[inline(always)]
    pub fn take_view_to(&mut self, range: impl RangeBounds<usize>, addition: usize) -> &[u8] {
        let len = self.len();
        let range = bounds(range, len);
        // Subtraction is valid after bounds(); checking before addition also
        // rejects overflow without wrapping in release builds.
        assert!(
            addition <= len - range.end,
            "cannot consume past buffered input"
        );
        let consume = range.end + addition;
        // SAFETY: bounds() validates the range; the check ensures consume <=
        // len.
        unsafe { self.take_view_unchecked(range, consume) }
    }

    /// Borrows a validated range and consumes `consume` bytes.
    ///
    /// # Safety
    ///
    /// Requires `range.start <= range.end <= self.len()` and
    /// `consume <= self.len()`.
    #[inline(always)]
    unsafe fn take_view_unchecked(&mut self, range: Range<usize>, consume: usize) -> &[u8] {
        let prefix = self.leftover.len();
        if prefix == 0 {
            // Common case of view 3 below: bypass prefix adjustment when the
            // whole input is in the chunk. Inline this per-Item path so the
            // caller can optimize selection and consumption together.
            // SAFETY: with no leftover, the caller's bounds apply directly
            // to the chunk's input.
            return unsafe { self.chunk.take_view_unchecked(range, consume) };
        }
        if range.end <= prefix {
            // 1. Entirely in leftover: borrow one contiguous view.
            //
            // leftover: [ ... [start ... end) ... ] | chunk: [ ... ]
            //                 <---- view --->
            let prefix_consume = consume.min(prefix);
            // SAFETY: consumption beyond leftover fits within the chunk.
            unsafe { self.chunk.consume_unchecked(consume - prefix_consume) };
            // SAFETY: the selection and prefix_consume fit in leftover.
            // Its helper saves the selection before advancing consumption.
            return unsafe { self.leftover.take_view_unchecked(range, prefix_consume) };
        } else if range.start < prefix {
            // 2. Across leftover and chunk: join into one contiguous view.
            //
            // leftover: [ ... [start ... ] | chunk: [ ... end) ... ]
            //                 <------------- view -------->
            //
            // Copy the selected chunk prefix into leftover before consuming
            // either part. Remove that prefix from the chunk's input so
            // overlapping frames do not see it twice, even if consume is 0.
            let chunk_size = range.end - prefix;
            // SAFETY: the validated range bounds chunk_size by chunk.len().
            // The source belongs to the chunk, separate from leftover.
            self.leftover
                .extend_from_slice(unsafe { self.chunk.view_unchecked(..chunk_size) });
            // SAFETY: chunk_size bytes were copied into leftover.
            unsafe { self.chunk.consume_unchecked(chunk_size) };
            let prefix_consume = consume.min(range.end);
            // SAFETY: joining preserves the combined input and extends
            // leftover through range.end. Consumption beyond it fits in
            // the remaining chunk.
            unsafe { self.chunk.consume_unchecked(consume - prefix_consume) };
            // SAFETY: the selection and prefix_consume fit in the joined
            // leftover. Save the selection before advancing consumption.
            return unsafe { self.leftover.take_view_unchecked(range, prefix_consume) };
        }

        // 3. Entirely in chunk: borrow one contiguous view.
        //
        // leftover: [ ... ] | chunk: [ ... [start ... end) ... ]
        //                                  <---- view --->
        let prefix_consume = consume.min(prefix);
        // SAFETY: prefix_consume is bounded by leftover's current length.
        unsafe { self.leftover.consume_uncheck(prefix_consume) };
        // SAFETY: subtracting prefix places the selection in the chunk;
        // consumption beyond leftover fits within the chunk's input.
        unsafe {
            self.chunk.take_view_unchecked(
                range.start - prefix..range.end - prefix,
                consume - prefix_consume,
            )
        }
    }

    /// Borrows `range` without joining its parts and consumes `consume` bytes.
    /// Selection and consumption both refer to input before this call.
    ///
    /// # Panics
    ///
    /// Panics if the range or consumption exceeds the buffered input.
    #[inline(always)]
    pub fn take_view_segments(
        &mut self,
        range: impl RangeBounds<usize>,
        consume: usize,
    ) -> Segments<&[u8]> {
        let len = self.len();
        let range = bounds(range, len);
        assert!(consume <= len, "cannot consume past buffered input");
        // SAFETY: bounds() validates the range; the assertion bounds
        // consumption.
        unsafe { self.take_view_segments_unchecked(range, consume) }
    }

    /// Borrows `range` without joining its parts, consuming through its end
    /// and another `addition` bytes.
    ///
    /// # Panics
    ///
    /// Panics if the range or its end plus `addition` exceeds buffered input.
    #[inline(always)]
    pub fn take_view_segments_to(
        &mut self,
        range: impl RangeBounds<usize>,
        addition: usize,
    ) -> Segments<&[u8]> {
        let len = self.len();
        let range = bounds(range, len);
        assert!(
            addition <= len - range.end,
            "cannot consume past buffered input"
        );
        let consume = range.end + addition;
        // SAFETY: bounds() validates the range; the check bounds addition
        // before summing, so consume <= len without overflow.
        unsafe { self.take_view_segments_unchecked(range, consume) }
    }

    /// Borrows a validated range without joining and consumes `consume` bytes.
    ///
    /// # Safety
    ///
    /// Requires `range.start <= range.end <= self.len()` and
    /// `consume <= self.len()`.
    #[inline(always)]
    unsafe fn take_view_segments_unchecked(
        &mut self,
        range: Range<usize>,
        consume: usize,
    ) -> Segments<&[u8]> {
        let prefix = self.leftover.len();
        if prefix == 0 {
            // Common case of view 3: one chunk supplies the entire selection.
            // SAFETY: with no leftover, the caller's selection and consumption
            // bounds apply directly to the chunk's input.
            return Segments::Contiguous(unsafe { self.chunk.take_view_unchecked(range, consume) });
        }
        // SAFETY: the caller validated selection and consumption; prefix
        // is the current nonempty leftover length.
        unsafe { self.take_view_segments_with_leftover_unchecked(range, consume, prefix) }
    }

    /// Extracts a validated view when input includes leftover bytes.
    ///
    /// # Safety
    ///
    /// Requires `prefix == self.leftover.len()`, `prefix != 0`,
    /// `range.start <= range.end <= self.len()`, and `consume <= self.len()`.
    #[cold]
    #[inline(never)]
    unsafe fn take_view_segments_with_leftover_unchecked(
        &mut self,
        range: Range<usize>,
        consume: usize,
        prefix: usize,
    ) -> Segments<&[u8]> {
        if range.end <= prefix {
            // 1. Entirely in leftover: borrow one contiguous view.
            //
            // leftover: [ ... [start ... end) ... ] | chunk: [ ... ]
            //                 <---- view --->
            let prefix_consume = consume.min(prefix);
            // SAFETY: the caller bounds consumption by combined input;
            // the part beyond the prefix fits within the chunk.
            unsafe { self.chunk.consume_unchecked(consume - prefix_consume) };
            // SAFETY: the selection and prefix_consume fit in the prefix.
            return Segments::Contiguous(unsafe {
                self.leftover.take_view_unchecked(range, prefix_consume)
            });
        } else if range.start < prefix {
            // 2. Across leftover and chunk: borrow two separate views.
            //
            // leftover: [ ... [start ... ] | chunk: [ ... end) ... ]
            //                 <- first ->           <- second ->
            let prefix_consume = consume.min(prefix);
            // SAFETY: this branch places the start in the prefix and the
            // end in the chunk. Splitting consumption at prefix bounds
            // each part's consumption by its own input length.
            return unsafe {
                Segments::Split(
                    self.leftover
                        .take_view_unchecked(range.start..prefix, prefix_consume),
                    self.chunk
                        .take_view_unchecked(0..range.end - prefix, consume - prefix_consume),
                )
            };
        }

        // 3. Entirely in chunk: borrow one contiguous view.
        //
        // leftover: [ ... ] | chunk: [ ... [start ... end) ... ]
        //                                  <---- view --->
        let prefix_consume = consume.min(prefix);
        // SAFETY: prefix_consume is bounded by leftover's current length.
        unsafe { self.leftover.consume_uncheck(prefix_consume) };
        // SAFETY: subtracting prefix places the validated selection within the
        // chunk; the remaining consumption is bounded by its input length.
        Segments::Contiguous(unsafe {
            self.chunk.take_view_unchecked(
                range.start - prefix..range.end - prefix,
                consume - prefix_consume,
            )
        })
    }

    /// Copies the bytes in `range` and consumes `consume` bytes.
    /// Selection and consumption both refer to input before this call.
    ///
    /// Allocates for nonempty selections. The returned buffer can outlive
    /// this buffer and may include bytes retained for the next frame.
    ///
    /// # Panics
    ///
    /// Panics if the range or consumption exceeds the buffered input.
    #[inline(always)]
    pub fn take_copy_bytes(&mut self, range: impl RangeBounds<usize>, consume: usize) -> BytesMut {
        let len = self.len();
        let range = bounds(range, len);
        assert!(consume <= len, "cannot consume past buffered input");

        let prefix = self.leftover.len();
        if prefix == 0 {
            // Common case of view 3: copy before consuming, including when
            // the selection overlaps input retained for the next frame.
            // SAFETY: with no leftover, the checked range lies in the chunk.
            let bytes = BytesMut::from(unsafe { self.chunk.view_unchecked(range) });
            // SAFETY: the checked consumption fits in the chunk, and copying
            // has not changed its input length.
            unsafe { self.chunk.consume_unchecked(consume) };
            return bytes;
        }
        if range.end <= prefix {
            // 1. Entirely in leftover: copy one selected view.
            //
            // leftover: [ ... [start ... end) ... ] | chunk: [ ... ]
            //                 <---- copy --->
            // SAFETY: bounds() orders the range and this branch bounds it
            // by the current leftover input.
            let bytes = BytesMut::from(unsafe { self.leftover.view_unchecked(range) });
            // SAFETY: consumption was checked against the input length;
            // copying leaves that length unchanged.
            unsafe { self.consume_unchecked(consume) };
            return bytes;
        } else if range.start < prefix {
            // 2. Across leftover and chunk: copy both views into one result.
            //
            // leftover: [ ... [start ... ] | chunk: [ ... end) ... ]
            //                 <------------- copy -------->
            let mut bytes = BytesMut::with_capacity(range.len());
            let end = range.end - prefix;
            // SAFETY: range.start < prefix < range.end <= len() places both
            // selected parts within initialized input. bytes is new storage,
            // so neither source overlaps the destination.
            unsafe {
                bytes.extend_from_slice(self.leftover.view_unchecked(range.start..));
                bytes.extend_from_slice(self.chunk.view_unchecked(..end));
            }
            // SAFETY: consumption was checked against the input length;
            // copying leaves that length unchanged.
            unsafe { self.consume_unchecked(consume) };
            return bytes;
        }

        // 3. Entirely in chunk: copy one selected view.
        //
        // leftover: [ ... ] | chunk: [ ... [start ... end) ... ]
        //                                  <---- copy --->
        let start = range.start - prefix;
        let end = range.end - prefix;
        // SAFETY: bounds() checked the selection against the total input;
        // subtracting prefix places it within the chunk's input.
        let bytes = BytesMut::from(unsafe { self.chunk.view_unchecked(start..end) });
        // SAFETY: the assertion bounds consume by the input length. Copying
        // leaves that length unchanged, even when the selection overlaps it.
        unsafe { self.consume_unchecked(consume) };
        bytes
    }

    /// Transfers the bytes in `range`, consuming through its end and another
    /// `addition` bytes. The returned buffer can outlive this buffer.
    ///
    /// May allocate. Selections within one input part are transferred without
    /// copying. Selections crossing parts are joined, which copies bytes.
    ///
    /// # Panics
    ///
    /// Panics if the range or its end plus `addition` exceeds buffered input.
    #[inline(always)]
    pub fn take_split_bytes(
        &mut self,
        range: impl RangeBounds<usize>,
        addition: usize,
    ) -> BytesMut {
        let len = self.len();
        let range = bounds(range, len);
        assert!(
            addition <= len - range.end,
            "cannot consume past buffered input"
        );
        let prefix = self.leftover.len();
        if prefix == 0 {
            // Common case of view 3: transfer directly from the chunk.
            // SAFETY: with no leftover, the checked range and its end plus
            // addition are bounded by the chunk's input length.
            return unsafe { self.chunk.take_split_bytes_unchecked(range, addition) };
        }
        if range.end <= prefix {
            // 1. Entirely in leftover: transfer one selected view.
            //
            // leftover: [ ... [start ... end) ... ] | chunk: [ ... ]
            //                 <--- first -->
            // SAFETY: this branch checks range.end <= leftover.len().
            let mut first = unsafe { self.leftover.split_to_unchecked(range.end) };
            first.advance(range.start);
            // SAFETY: splitting removed range.end bytes from the input.
            // The addition check bounds the remaining consumption.
            unsafe { self.consume_unchecked(addition) };
            return first;
        } else if range.start < prefix {
            // 2. Across leftover and chunk: transfer and join both views.
            //
            // leftover: [ ... [start ... ] | chunk: [ ... end) ... ]
            //                 <- first ->           <- second ->
            //
            // Append second to first to return one contiguous buffer. This
            // copies second and may also move first when its storage grows.
            // SAFETY: prefix is the current leftover length.
            let mut first = unsafe { self.leftover.split_to_unchecked(prefix) };
            first.advance(range.start);
            // SAFETY: removing prefix from the validated range places its
            // end and additional consumption within the chunk's input.
            let second = unsafe {
                self.chunk
                    .take_split_bytes_unchecked(0..range.end - prefix, addition)
            };
            first.extend_from_slice(&second);
            return first;
        }

        // 3. Entirely in chunk: transfer one selected view.
        //
        // leftover: [ ... ] | chunk: [ ... [start ... end) ... ]
        //                                  <--- bytes -->
        //
        // Subtract prefix to express the range within the chunk. Consumption
        // reaches range.end, so it also discards all of leftover.
        // SAFETY: the selection is entirely in the chunk. The addition check
        // bounds consumption through its end plus addition by the input length.
        let bytes = unsafe {
            self.chunk
                .take_split_bytes_unchecked(range.start - prefix..range.end - prefix, addition)
        };
        self.leftover.clear();
        bytes
    }

    /// Appends bytes from one read and returns the number read.
    ///
    /// Makes room as needed and always supplies a nonempty destination.
    /// Consumed bytes may be discarded before reading.
    ///
    /// # Errors
    ///
    /// Returns reader errors without retrying. Bytes from a failed read are
    /// not included in the buffered input.
    ///
    /// # Panics
    ///
    /// Panics if the reader panics or reports more bytes than the destination
    /// can hold.
    // Keep refilling out of the per-Item decoding loop's generated code.
    #[inline(never)]
    pub fn write_buf<R: ?Sized + Read>(&mut self, reader: &mut R) -> io::Result<usize> {
        self.make_room();
        self.chunk.read_from(reader)
    }

    // Preserve pending input before preparing the main chunk for another read.
    #[inline]
    fn make_room(&mut self) {
        if !self.chunk.is_empty() {
            self.leftover.extend_from_slice(self.chunk.buffer());
        }
        self.chunk.clear();

        let mut cap = self.chunk.buffer_size();
        if self.leftover.len() >= cap {
            // Grow for accumulated input, not merely for another short read
            // or a small tail left by splitting an owned frame.
            cap = cap.saturating_mul(2).min(isize::MAX as usize);
            // Discard input and its initialization bound
            self.chunk.clear_buf();
            self.chunk.reserve(cap);
        } else if !self.chunk.try_reclaim()
            && (self.chunk.capacity() == 0 || !self.leftover.is_empty())
        {
            // Discard input and its initialization bound
            self.chunk.clear_buf();
            // Reclamation can fail while another Item owns the storage.
            // Without pending input, keep a nonempty tail to avoid allocating
            // just to probe EOF. Otherwise restore the target read capacity.
            self.chunk.reserve(cap);
        }
    }
}

#[cfg(test)]
mod test {
    use std::io::{self, Read};
    use std::ops::Bound::{Excluded, Included, Unbounded};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::{Buffer, Segments};
    use crate::read::{Frame, FramedRead, Framer, OwnedDecoder, ViewDecoder};
    use crate::test::allocations;

    #[test]
    fn short_reads_reuse_prefix_storage_with_continuous_lookahead() {
        struct Source(u8);
        impl Read for Source {
            fn read(&mut self, dst: &mut [u8]) -> io::Result<usize> {
                dst[0] = self.0;
                self.0 = self.0.wrapping_add(1);
                Ok(1)
            }
        }
        struct Lookahead;
        impl Framer for Lookahead {
            type Error = io::Error;

            fn frame(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
                Ok(if src.len() < 3 {
                    Frame::need_more(0)
                } else {
                    Frame::ready_split(..1, 0)
                })
            }
        }
        impl ViewDecoder for Lookahead {
            type Item<'a> = &'a [u8];

            fn decode_view<'a>(&mut self, frame: &'a [u8]) -> io::Result<Self::Item<'a>> {
                Ok(frame)
            }
        }
        impl OwnedDecoder for Lookahead {
            type Item = u8;

            fn decode_owned(&mut self, frame: Segments<&[u8]>) -> io::Result<u8> {
                Ok(*frame.iter().next().unwrap())
            }
        }

        // Both paths consume one byte while retaining two bytes of lookahead.
        // Varying input detects misplaced bytes when the prefix is reclaimed.
        for view in [true, false] {
            let mut reader = FramedRead::with_capacity(Source(0), Lookahead, 64);
            let mut check_next = |index: usize| {
                let item = if view {
                    reader.decode_view().unwrap().unwrap()[0]
                } else {
                    reader.decode_owned().unwrap().unwrap()
                };
                assert_eq!(item, index as u8);
            };
            for index in 0..32 {
                check_next(index);
            }
            let count = if cfg!(miri) { 64 } else { 4096 };
            let (_, calls) = allocations(|| {
                for index in 32..32 + count {
                    check_next(index);
                }
            });
            assert_eq!(calls, 0, "bounded lookahead must reuse prefix storage");
            assert_eq!(
                reader.buffer().to_vec(),
                [(32 + count) as u8, (33 + count) as u8]
            );
        }
    }

    #[test]
    fn selections_preserve_overlap_and_ownership() {
        let bytes: &[u8] = if cfg!(miri) { b"abcd" } else { b"abcdefgh" };
        for boundary in 0..=bytes.len() {
            for start in 0..=bytes.len() {
                for end in start..=bytes.len() {
                    for consume in 0..=bytes.len() {
                        for mode in 0..6 {
                            if matches!(mode, 2 | 4 | 5) && consume < end {
                                continue;
                            }
                            let mut buffer = Buffer::with_capacity(16);
                            // Exercise consumed prefix storage as well as the
                            // boundary between the two live parts.
                            let input = [b"?", &bytes[boundary..]].concat();
                            buffer.write_buf(&mut &input[..]).unwrap();
                            buffer.consume(1);
                            buffer.leftover.extend_from_slice(b"!");
                            buffer.leftover.extend_from_slice(&bytes[..boundary]);
                            buffer.consume(1);
                            let expected = &bytes[start..end];
                            match mode {
                                0 | 5 => {
                                    let selected = if mode == 0 {
                                        buffer.take_view_segments(start..end, consume)
                                    } else {
                                        buffer.take_view_segments_to(start..end, consume - end)
                                    };
                                    assert_eq!(selected.to_vec(), expected);
                                }
                                1 => assert_eq!(buffer.take_view(start..end, consume), expected),
                                2 => assert_eq!(
                                    buffer.take_view_to(start..end, consume - end),
                                    expected,
                                ),
                                _ => {
                                    let original = buffer.buffer().slice(start..end);
                                    let pointers: Vec<_> =
                                        original.iter().map(|byte| byte as *const u8).collect();
                                    let item = if mode == 3 {
                                        buffer.take_copy_bytes(start..end, consume)
                                    } else {
                                        buffer.take_split_bytes(start..end, consume - end)
                                    };
                                    assert_eq!(item.to_vec(), expected);
                                    for (byte, ptr) in item.iter().zip(pointers) {
                                        if mode == 3 {
                                            assert!(!std::ptr::eq(byte, ptr));
                                        } else if end <= boundary || start >= boundary {
                                            assert!(std::ptr::eq(byte, ptr));
                                        }
                                    }
                                    buffer.write_buf(&mut &b"ijk"[..]).unwrap();
                                    assert_eq!(item.to_vec(), expected);
                                    assert_eq!(
                                        buffer.buffer().to_vec(),
                                        [&bytes[consume..], b"ijk"].concat()
                                    );
                                    continue;
                                }
                            }
                            assert_eq!(buffer.buffer().to_vec(), &bytes[consume..]);
                            // A second borrow after joining must see neither
                            // duplicated lookahead nor an omitted byte.
                            assert_eq!(buffer.take_view(.., 0), &bytes[consume..]);
                            buffer.write_buf(&mut &b"ijk"[..]).unwrap();
                            assert_eq!(
                                buffer.buffer().to_vec(),
                                [&bytes[consume..], b"ijk"].concat()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_ranges_and_consumption_leave_both_parts_unchanged() {
        for boundary in 0..=4 {
            let mut buffer = Buffer::with_capacity(8);
            buffer.leftover.extend_from_slice(&b"abcd"[..boundary]);
            buffer.chunk.read_from(&mut &b"abcd"[boundary..]).unwrap();
            for range in [
                (Included(3), Excluded(2)),
                (Unbounded, Included(usize::MAX)),
                (Excluded(usize::MAX), Unbounded),
                (Included(0), Excluded(5)),
            ] {
                for mode in 0..6 {
                    assert!(
                        catch_unwind(AssertUnwindSafe(|| match mode {
                            0 => {
                                buffer.take_view_segments(range, 0);
                            }
                            1 => {
                                buffer.take_view(range, 0);
                            }
                            2 => {
                                buffer.take_view_to(range, 0);
                            }
                            3 => {
                                buffer.take_copy_bytes(range, 0);
                            }
                            4 => {
                                buffer.take_split_bytes(range, 0);
                            }
                            _ => {
                                buffer.take_view_segments_to(range, 0);
                            }
                        }))
                        .is_err()
                    );
                    assert_eq!(buffer.buffer().to_vec(), b"abcd");
                }
            }
            for amount in [5, usize::MAX] {
                assert!(catch_unwind(AssertUnwindSafe(|| buffer.consume(amount))).is_err());
                assert!(
                    catch_unwind(AssertUnwindSafe(|| {
                        buffer.take_view(..2, amount);
                    }))
                    .is_err()
                );
                assert!(
                    catch_unwind(AssertUnwindSafe(|| {
                        buffer.take_view_to(..2, amount);
                    }))
                    .is_err()
                );
                assert!(
                    catch_unwind(AssertUnwindSafe(|| {
                        buffer.take_view_segments(..2, amount);
                    }))
                    .is_err()
                );
                assert!(
                    catch_unwind(AssertUnwindSafe(|| {
                        buffer.take_view_segments_to(..2, amount);
                    }))
                    .is_err()
                );
                assert!(
                    catch_unwind(AssertUnwindSafe(|| {
                        buffer.take_copy_bytes(..2, amount);
                    }))
                    .is_err()
                );
                assert!(
                    catch_unwind(AssertUnwindSafe(|| {
                        buffer.take_split_bytes(..2, amount);
                    }))
                    .is_err()
                );
                assert_eq!(buffer.buffer().to_vec(), b"abcd");
            }
        }
    }

    #[test]
    fn refills_keep_prefixes_and_reuse_the_whole_chunk() {
        let mut buffer = Buffer::with_capacity(4);
        buffer.leftover.extend_from_slice(&[0; 32]);
        buffer.leftover.clear();
        buffer.write_buf(&mut &b"ab"[..]).unwrap();
        buffer.write_buf(&mut &b"cd"[..]).unwrap();
        let capacity = buffer.chunk.capacity();
        assert_eq!(capacity, 4);
        buffer.consume(1);
        let address = buffer.leftover.buffer().as_ptr();
        buffer.write_buf(&mut &b"ef"[..]).unwrap();
        assert_eq!(buffer.buffer().to_vec(), b"bcdef");
        // Appending discards the consumed prefix without moving the retained
        // bytes when tail space remains. Short reads keep the same chunk size.
        assert_eq!(buffer.leftover.buffer(), b"bcd");
        assert_eq!(buffer.leftover.buffer().as_ptr(), address);
        assert_eq!(buffer.chunk.capacity(), capacity);
        buffer.write_buf(&mut &b"gh"[..]).unwrap();
        assert_eq!(buffer.buffer().to_vec(), b"bcdefgh");
        assert!(buffer.chunk.capacity() >= 2 * capacity);

        let mut buffer = Buffer::with_capacity(4);
        assert_eq!(buffer.write_buf(&mut &b"abcd"[..]).unwrap(), 4);
        let owned = buffer.take_split_bytes(..4, 0);
        assert_eq!(buffer.chunk.capacity(), 0);
        assert_eq!(buffer.write_buf(&mut &b"ef"[..]).unwrap(), 2);
        assert_eq!(&owned[..], b"abcd");
        assert_eq!(buffer.buffer().to_vec(), b"ef");

        for capacity in [0, 1, 4] {
            let mut buffer = Buffer::with_capacity(capacity);
            let mut source = &b"abcdefghi"[..];
            let mut expected = Vec::new();
            while !source.is_empty() {
                let before = source;
                let n = buffer.write_buf(&mut source).unwrap();
                assert!(n > 0);
                expected.extend_from_slice(&before[..n]);
                assert_eq!(buffer.buffer().to_vec(), expected);
                assert_eq!(buffer.chunk.len(), n);
            }
            // EOF probes retain input without repeatedly growing the chunk.
            assert_eq!(buffer.write_buf(&mut io::empty()).unwrap(), 0);
            let capacity = buffer.chunk.capacity();
            for _ in 0..3 {
                assert_eq!(buffer.write_buf(&mut io::empty()).unwrap(), 0);
                assert_eq!(buffer.buffer().to_vec(), expected);
                assert_eq!(buffer.chunk.capacity(), capacity);
            }
            buffer.consume(expected.len());
            assert!(buffer.is_empty());
            buffer.write_buf(&mut &b"x"[..]).unwrap();
            assert_eq!(buffer.buffer().to_vec(), b"x");
            assert!(matches!(buffer.buffer(), Segments::Contiguous(_)));
        }
    }
}
