//! Buffering and frame extraction for the reader.

#[cfg(feature = "nightly")]
use std::io::BorrowedBuf;
use std::io::{self, Read};
use std::ops::{Range, RangeBounds};
#[cfg(not(feature = "nightly"))]
use std::ptr;
use std::slice;

use bytes::{Buf, BytesMut};

use super::Segments;
use super::segments::bounds;

/// Buffered input that can be borrowed or transferred to a decoder.
///
/// Input may have two consecutive parts. Borrowing a frame across those parts
/// joins the selected bytes; borrowing within one part does not copy.
//
// leftover.buf: [ consumed | retained input ]
//                          ^ leftover.offset
// buf:       [ consumed | buffered | initialized spare | spare capacity ]
//
// leftover.offset <= leftover.buf.len().
// consumed + buffered <= buf.len() <= capacity.
// buf.len() tracks initialization separately from committed input. Failed
// reads never add input, even when they initialize or write their destination.
#[derive(Debug)]
pub struct Buffer {
    buf: BytesMut,
    consumed: usize,
    buffered: usize,

    leftover: LeftOver,

    // Keep the refill size after owned extraction shrinks the BytesMut view.
    buffer_size: usize,
}

#[derive(Debug)]
struct LeftOver {
    buf: BytesMut,
    offset: usize,
}

impl LeftOver {
    #[inline]
    fn new() -> Self {
        Self {
            buf: BytesMut::new(),
            offset: 0,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.buf.len() - self.offset
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    fn clear(&mut self) {
        self.offset = self.buf.len();
    }

    #[inline]
    fn buffer(&self) -> &[u8] {
        // SAFETY: offset never exceeds buf.len(). Consumption advances it
        // within the live prefix; clearing sets it to the end of buf.
        // The returned borrow prevents changes to the storage while it is used.
        unsafe { self.buf.get_unchecked(self.offset..) }
    }

    #[inline]
    fn consume(&mut self, amount: usize) {
        if amount == 0 {
            return;
        }
        assert!(amount <= self.len(), "cannot consume past buffered input");
        // SAFETY: the check keeps the new offset within buf.len().
        unsafe { self.consume_uncheck(amount) };
    }

    // The caller must ensure amount <= len().
    #[inline]
    unsafe fn consume_uncheck(&mut self, amount: usize) {
        self.offset += amount;
    }

    #[inline]
    fn extend_from_slice(&mut self, other: &[u8]) {
        if self.is_empty() {
            self.buf.clear();
            self.offset = 0;
        }
        self.buf.extend_from_slice(other);
    }
}

impl Buffer {
    /// Creates an empty buffer with at least `capacity` bytes for reading.
    /// Zero defers allocation until the first read.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        let buf = BytesMut::with_capacity(capacity);
        Self {
            buffer_size: buf.capacity(),
            buf,
            consumed: 0,
            buffered: 0,
            leftover: LeftOver::new(),
        }
    }

    /// Returns the number of buffered bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.leftover.len() + self.buffered
    }

    /// Returns `true` if no input is buffered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    fn clear(&mut self) {
        self.leftover.clear();
        self.consumed = 0;
        self.buffered = 0;
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
        unsafe { self.consume_uncheck(amount) };
    }

    /// Consumes `amount` bytes without checking the input length.
    ///
    /// # Safety
    ///
    /// Requires `amount <= self.len()`.
    #[inline]
    unsafe fn consume_uncheck(&mut self, mut amount: usize) {
        // Most frames use only the main chunk; avoid prefix bookkeeping.
        if self.leftover.is_empty() {
            self.consumed += amount;
            self.buffered -= amount;
            return;
        }
        if amount == 0 {
            return;
        }
        if self.leftover.len() >= amount {
            // SAFETY: this branch checks amount against the live prefix.
            unsafe { self.leftover.consume_uncheck(amount) };
            return;
        }
        amount -= self.leftover.len();
        self.leftover.clear();
        self.consumed += amount;
        self.buffered -= amount;
    }

    /// Borrows the buffered input without joining its parts.
    #[inline]
    pub fn buffer(&self) -> Segments<&[u8]> {
        let leftover = self.leftover.buffer();
        // SAFETY: consumed + buffered <= buf.len(), so the slice covers only
        // initialized input. The borrow prevents mutation of its storage.
        let buf =
            unsafe { slice::from_raw_parts(self.buf.as_ptr().add(self.consumed), self.buffered) };
        match (leftover.is_empty(), buf.is_empty()) {
            (true, _) => Segments::Contiguous(buf),
            (false, true) => Segments::Contiguous(leftover),
            (false, false) => Segments::Split(leftover, buf),
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
        unsafe { self.take_view_uncheck(range, consume) }
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
        unsafe { self.take_view_uncheck(range, consume) }
    }

    /// Borrows a validated range and consumes `consume` bytes.
    ///
    /// # Safety
    ///
    /// Requires `range.start <= range.end <= self.len()` and
    /// `consume <= self.len()`.
    #[inline]
    unsafe fn take_view_uncheck(&mut self, range: Range<usize>, consume: usize) -> &[u8] {
        let prefix = self.leftover.len();
        // With no leftover, go directly to the main-chunk selection below.
        if prefix != 0 {
            if range.end <= prefix {
                // 1. The selection is entirely in leftover; borrow it directly.
                //
                // leftover: [ ... [start ... end) ... ] | buf: [ ... ]
                //                 <--- view --->
                let start = self.leftover.offset + range.start;
                let end = self.leftover.offset + range.end;
                // SAFETY: the caller bounds consume by the total input length.
                unsafe { self.consume_uncheck(consume) };
                // SAFETY: range.end <= prefix keeps these storage offsets
                // within
                // leftover.buf, since prefix excludes leftover.offset.
                // Consumption changes counters only, leaving these bytes
                // intact.
                return unsafe { self.leftover.buf.get_unchecked(start..end) };
            } else if range.start < prefix {
                // 2. The selection crosses the boundary; join it in leftover.
                //
                // leftover: [ ... [start ... ] | buf: [ ... end) ... ]
                //                 <----------- view -------->
                //                                    <- copy ->
                //
                // Move buf's prefix through end into leftover, then consume.
                // Removing those bytes from buf's input keeps overlapping
                // frames
                // from seeing them twice, including when consume is zero.
                // SAFETY: the caller validates the range and consumption; this
                // branch selects bytes from both input parts.
                return unsafe { self.take_view_cross(range, consume) };
            }
        }

        // 3. The selection is entirely in buf; borrow it directly.
        //
        // leftover: [ ... ] | buf: [ ... [start ... end) ... ]
        //                                <--- view --->
        //
        // Range offsets include leftover, so subtract its length before
        // translating them to buf's storage offsets.
        let start = self.consumed + (range.start - prefix);
        let end = self.consumed + (range.end - prefix);
        // SAFETY: the caller bounds consume by the total input length.
        unsafe { self.consume_uncheck(consume) };
        // SAFETY: the caller guarantees the range is within both parts;
        // range.start >= prefix places it within buf's initialized input.
        // Consumption changes counters without moving these bytes.
        unsafe { self.buf.get_unchecked(start..end) }
    }

    // The caller validates the range and consumption and ensures the range
    // crosses from leftover into buf.
    #[inline(never)]
    unsafe fn take_view_cross(&mut self, range: Range<usize>, consume: usize) -> &[u8] {
        let prefix = self.leftover.len();
        let append = range.end - prefix;
        // SAFETY: range.end <= len() ensures append <= buffered. The
        // source is initialized input in buf, separate from leftover.
        self.leftover.extend_from_slice(unsafe {
            self.buf
                .get_unchecked(self.consumed..self.consumed + append)
        });
        self.consumed += append;
        self.buffered -= append;
        let start = self.leftover.offset + range.start;
        let end = self.leftover.offset + range.end;
        // SAFETY: the caller bounds consume by the total input length.
        // Joining parts, if needed, does not change that length.
        unsafe { self.consume_uncheck(consume) };
        // SAFETY: appending through range.end puts the entire selection
        // in leftover.buf. Consumption leaves the stored bytes intact.
        unsafe { self.leftover.buf.get_unchecked(start..end) }
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
        unsafe { self.take_view_segments_uncheck(range, consume) }
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
        unsafe { self.take_view_segments_uncheck(range, consume) }
    }

    /// Borrows a validated range without joining and consumes `consume` bytes.
    ///
    /// # Safety
    ///
    /// Requires `range.start <= range.end <= self.len()` and
    /// `consume <= self.len()`.
    #[inline]
    unsafe fn take_view_segments_uncheck(
        &mut self,
        range: Range<usize>,
        consume: usize,
    ) -> Segments<&[u8]> {
        let prefix = self.leftover.len();
        let offset = self.leftover.offset;
        let consumed = self.consumed;
        // SAFETY: the caller bounds consumption by the input length.
        // Consumption changes counters only; the saved offsets still locate
        // the original bytes, including any consumed part of the selection.
        unsafe { self.consume_uncheck(consume) };

        // With no leftover, go directly to the main-chunk selection below.
        if prefix != 0 {
            if range.end <= prefix {
                // 1. The selection is entirely in leftover; borrow one part.
                //
                // leftover: [ ... [start ... end) ... ] | buf: [ ... ]
                //                 <--- first -->
                let start = offset + range.start;
                let end = offset + range.end;
                // SAFETY: range.end <= prefix keeps both storage offsets within
                // leftover.buf. The caller guarantees start <= end.
                return Segments::Contiguous(unsafe {
                    self.leftover.buf.get_unchecked(start..end)
                });
            } else if range.start < prefix {
                // 2. The selection crosses the boundary; borrow both parts.
                //
                // leftover: [ ... [start ... ] | buf: [ ... end) ... ]
                //                 <- first ->        <- second ->
                let start = offset + range.start;
                let end = consumed + (range.end - prefix);
                // SAFETY: start is within leftover's original input, which
                // still
                // ends at leftover.buf.len(). The validated range.end places
                // end
                // within buf's initialized input. Consumption moved neither
                // part.
                return unsafe {
                    Segments::Split(
                        self.leftover.buf.get_unchecked(start..),
                        self.buf.get_unchecked(consumed..end),
                    )
                };
            }
        }

        // 3. The selection is entirely in buf; borrow one part.
        //
        // leftover: [ ... ] | buf: [ ... [start ... end) ... ]
        //                                <--- bytes -->
        let start = consumed + (range.start - prefix);
        let end = consumed + (range.end - prefix);
        // SAFETY: the caller bounds the selection by the total input;
        // subtracting prefix places it within buf's initialized input.
        Segments::Contiguous(unsafe { self.buf.get_unchecked(start..end) })
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
    pub fn take_bytes(&mut self, range: impl RangeBounds<usize>, consume: usize) -> BytesMut {
        let len = self.len();
        let range = bounds(range, len);
        assert!(consume <= len, "cannot consume past buffered input");

        let prefix = self.leftover.len();
        let bytes = if range.end <= prefix {
            // 1. The selection is entirely in leftover; copy that range.
            //
            // leftover: [ ... [start ... end) ... ] | buf: [ ... ]
            //                 <---- copy --->
            let start = self.leftover.offset + range.start;
            let end = self.leftover.offset + range.end;
            // SAFETY: range.end <= prefix keeps the storage offsets within
            // leftover.buf; bounds() checked that start <= end.
            BytesMut::from(unsafe { self.leftover.buf.get_unchecked(start..end) })
        } else if range.start < prefix {
            // 2. The selection crosses the boundary; copy both into one result.
            //
            // leftover: [ ... [start ... ] | buf: [ ... end) ... ]
            //                 <-------------- copy ----->
            let mut bytes = BytesMut::with_capacity(range.end - range.start);
            let start = self.leftover.offset + range.start;
            let end = self.consumed + (range.end - prefix);
            // SAFETY: range.start < prefix < range.end <= len() places both
            // selected parts within initialized input. bytes is new storage,
            // so neither source overlaps the destination.
            unsafe {
                bytes.extend_from_slice(self.leftover.buf.get_unchecked(start..));
                bytes.extend_from_slice(self.buf.get_unchecked(self.consumed..end));
            }
            bytes
        } else {
            // 3. The selection is entirely in buf; copy that range.
            //
            // leftover: [ ... ] | buf: [ ... [start ... end) ... ]
            //                                <---- copy --->
            let start = self.consumed + (range.start - prefix);
            let end = self.consumed + (range.end - prefix);
            // SAFETY: bounds() checked the selection against the total input;
            // subtracting prefix places it within buf's initialized input.
            BytesMut::from(unsafe { self.buf.get_unchecked(start..end) })
        };
        // SAFETY: the assertion bounds consume by the input length. Copying
        // leaves that length unchanged, even when the selection overlaps it.
        unsafe { self.consume_uncheck(consume) };
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
    pub fn take_bytes_to(&mut self, range: impl RangeBounds<usize>, addition: usize) -> BytesMut {
        let len = self.len();
        let range = bounds(range, len);
        assert!(
            addition <= len - range.end,
            "cannot consume past buffered input"
        );
        let consume = range.end + addition;
        if range.is_empty() {
            // SAFETY: the addition check bounds consume by the input length.
            unsafe { self.consume_uncheck(consume) };
            return BytesMut::new();
        }
        let prefix = self.leftover.len();
        // With no leftover, go directly to the main-chunk selection below.
        if prefix != 0 {
            if range.end <= prefix {
                // 1. The selection is entirely in leftover; transfer one part.
                //
                // leftover: [ ... [start ... end) ... ] | buf: [ ... ]
                //                 <--- first -->
                let offset = self.leftover.offset;
                // Split before resetting offset, since split_to can allocate.
                let mut first = self.leftover.buf.split_to(offset + range.end);
                self.leftover.offset = 0;
                first.advance(offset + range.start);
                // SAFETY: splitting removed range.end bytes from the input.
                // The addition check ensures range.end <= consume <= the old
                // length.
                unsafe { self.consume_uncheck(consume - range.end) };
                return first;
            } else if range.start < prefix {
                // 2. The selection crosses the boundary; join the selected
                // parts.
                //
                // leftover: [ ... [start ... ] | buf: [ ... end) ... ]
                //                 <- first ->        <- second ->
                //
                // Append second to first to return one contiguous buffer. This
                // copies second and may also move first when its storage grows.
                let offset = self.leftover.offset;
                let mut first = self.leftover.buf.split_to(offset + prefix);
                self.leftover.offset = 0;
                first.advance(offset + range.start);
                let second = self.split_buf(0..range.end - prefix, consume - range.end);
                first.extend_from_slice(&second);
                return first;
            }
        }

        // 3. The selection is entirely in buf; transfer one part.
        //
        // leftover: [ ... ] | buf: [ ... [start ... end) ... ]
        //                                <--- bytes -->
        //
        // Subtract prefix to express the range within buf. Consumption
        // reaches range.end, so it also discards all of leftover.
        let bytes = self.split_buf(
            range.start - prefix..range.end - prefix,
            consume - range.end,
        );
        self.leftover.clear();
        bytes
    }

    // The caller checked the complete selection before splitting either part.
    #[inline]
    fn split_buf(&mut self, range: Range<usize>, addition: usize) -> BytesMut {
        let skip = self.consumed + range.start;
        let mut bytes = self.buf.split_to(self.consumed + range.end);
        self.buffered -= range.end + addition;
        self.consumed = addition;
        bytes.advance(skip);
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
        #[cfg(feature = "nightly")]
        {
            self.write_uninit_buf(reader)
        }
        #[cfg(not(feature = "nightly"))]
        {
            self.write_init_buf(reader)
        }
    }

    // Retain the pending input separately. Existing leftover triggers doubling.
    #[inline]
    fn make_room(&mut self) {
        let double = !self.leftover.is_empty();
        if self.buffered > 0 {
            // SAFETY: consumed + buffered <= buf.len(); this range contains
            // only initialized input. Extending leftover cannot mutate buf.
            let pending = unsafe {
                self.buf
                    .get_unchecked(self.consumed..self.consumed + self.buffered)
            };
            self.leftover.extend_from_slice(pending);
        }
        self.consumed = 0;
        self.buffered = 0;

        debug_assert_eq!(self.consumed, 0);
        debug_assert_eq!(self.buffered, 0);
        if !double {
            // Splitting off the whole chunk can leave no writable capacity.
            // A zero initial capacity also needs space before the first read.
            if self.buf.capacity() == 0 {
                self.buf.reserve(self.buffer_size.max(1));
                self.buffer_size = self.buffer_size.max(self.buf.capacity());
            }
            return;
        }

        // Pending input is in leftover; reserve need not retain the old chunk.
        self.buf.clear();
        self.buf
            .reserve(self.buffer_size.saturating_mul(2).min(isize::MAX as usize));
        self.buffer_size = self.buffer_size.max(self.buf.capacity());
    }

    #[cfg(not(feature = "nightly"))]
    #[inline(always)]
    fn write_init_buf<R: ?Sized + Read>(&mut self, reader: &mut R) -> io::Result<usize> {
        let initialized = self.buf.len();
        let capacity = self.buf.capacity();
        if initialized < capacity {
            // SAFETY: spare_capacity_mut covers capacity - initialized writable
            // bytes. Writing all of them makes the whole allocation
            // initialized;
            // no input is overwritten, and set_len exposes only those bytes.
            unsafe {
                ptr::write_bytes(
                    self.buf.spare_capacity_mut().as_mut_ptr(),
                    0,
                    capacity - initialized,
                );
                self.buf.set_len(capacity);
            }
        }
        let nread = reader.read(&mut self.buf)?;
        assert!(
            nread <= capacity,
            "Read returned {nread} bytes for a buffer of {capacity} bytes"
        );
        self.buffered = nread;
        Ok(nread)
    }

    #[cfg(feature = "nightly")]
    #[inline(always)]
    fn write_uninit_buf<R: ?Sized + Read>(&mut self, reader: &mut R) -> io::Result<usize> {
        let capacity = self.buf.capacity();
        let filled = self.consumed + self.buffered;
        // buf.len() includes initialized spare bytes; filled ends the input.
        // Neither boundary implies that those bytes contain zeros.
        let mut dst = if self.buf.len() == capacity {
            // From<&mut [u8]> marks the destination as initialized without
            // writing to it. The default read_buf can then call read() without
            // clearing space that was initialized by an earlier operation.
            // SAFETY: consumed + buffered <= buf.len() by the buffer invariant.
            // The tail is initialized and exclusively borrowed through self.
            BorrowedBuf::from(unsafe { self.buf.get_unchecked_mut(filled..) })
        } else {
            // This nightly's BorrowedBuf records initialization with a bool,
            // so it cannot describe an initialized prefix within the tail.
            // Treat the whole tail as potentially uninitialized to keep the
            // full read size. The read() fallback may clear that prefix again;
            // a native read_buf implementation can fill the tail directly.
            //
            // SAFETY: filled <= buf.len(); shortening leaves only initialized
            // input exposed if the reader panics while filling spare capacity.
            unsafe { self.buf.set_len(filled) };
            BorrowedBuf::from(self.buf.spare_capacity_mut())
        };

        // read_buf may advance its cursor and then return an error. Keep the
        // result until initialization has been recorded, but do not change
        // buffered yet: an error or panic must not append that partial input.
        let result = reader.read_buf(dst.unfilled());
        let nread = dst.len();
        // Remember a fully initialized tail even after a short read or error,
        // so a later read() fallback need not initialize it again. Otherwise,
        // only the cursor's filled prefix is known to be initialized; do not
        // restore the partially initialized tail discarded before the call.
        let len = if dst.is_init() {
            capacity
        } else {
            filled + nread
        };
        // SAFETY: the existing input and BorrowedBuf's filled prefix form a
        // contiguous initialized range within capacity. If is_init is true,
        // the remaining tail is initialized as well.
        unsafe { self.buf.set_len(len) };
        result?;
        // Initialization alone does not make bytes input. Commit only the
        // bytes reported by a successful read, including zero at EOF.
        self.buffered += nread;
        Ok(nread)
    }
}

#[cfg(test)]
mod test {
    use std::io::{self, Read};
    use std::ops::Bound::{Excluded, Included, Unbounded};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::{Buffer, Segments};

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
                            buffer.leftover.buf.extend_from_slice(b"!");
                            buffer.leftover.buf.extend_from_slice(&bytes[..boundary]);
                            buffer.leftover.offset = 1;
                            buffer.buf.extend_from_slice(b"?");
                            buffer.buf.extend_from_slice(&bytes[boundary..]);
                            buffer.consumed = 1;
                            buffer.buffered = bytes.len() - boundary;
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
                                        buffer.take_bytes(start..end, consume)
                                    } else {
                                        buffer.take_bytes_to(start..end, consume - end)
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
            buffer.leftover.buf.extend_from_slice(&b"abcd"[..boundary]);
            buffer.buf.extend_from_slice(&b"abcd"[boundary..]);
            buffer.buffered = 4 - boundary;
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
                                buffer.take_bytes(range, 0);
                            }
                            4 => {
                                buffer.take_bytes_to(range, 0);
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
                        buffer.take_bytes(..2, amount);
                    }))
                    .is_err()
                );
                assert!(
                    catch_unwind(AssertUnwindSafe(|| {
                        buffer.take_bytes_to(..2, amount);
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
        buffer.leftover.buf.reserve(32);
        buffer.write_buf(&mut &b"ab"[..]).unwrap();
        buffer.write_buf(&mut &b"cd"[..]).unwrap();
        let capacity = buffer.buf.capacity();
        assert_eq!(capacity, 4);
        buffer.consume(1);
        let address = buffer.leftover.buf.as_ptr();
        buffer.write_buf(&mut &b"ef"[..]).unwrap();
        assert_eq!(buffer.buffer().to_vec(), b"bcdef");
        // Keep the consumed prefix in place instead of copying live bytes
        // over it. Only the main read capacity grows for the longer input.
        assert_eq!(buffer.leftover.offset, 1);
        assert_eq!(&buffer.leftover.buf[..], b"abcd");
        assert_eq!(buffer.leftover.buf.as_ptr(), address);
        assert!(buffer.buf.capacity() >= 2 * capacity);

        let mut buffer = Buffer::with_capacity(4);
        assert_eq!(buffer.write_buf(&mut &b"abcd"[..]).unwrap(), 4);
        let owned = buffer.take_bytes_to(..4, 0);
        assert_eq!(buffer.buf.capacity(), 0);
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
                assert_eq!(buffer.consumed, 0);
            }
            // EOF probes retain input; existing leftover still triggers the
            // same capacity growth as any other read attempt.
            assert_eq!(buffer.write_buf(&mut io::empty()).unwrap(), 0);
            for _ in 0..3 {
                assert_eq!(buffer.write_buf(&mut io::empty()).unwrap(), 0);
                assert_eq!(buffer.buffer().to_vec(), expected);
            }
            buffer.consume(expected.len());
            assert!(buffer.is_empty());
            buffer.write_buf(&mut &b"x"[..]).unwrap();
            assert_eq!(buffer.buffer().to_vec(), b"x");
            assert!(matches!(buffer.buffer(), Segments::Contiguous(_)));
        }
    }

    #[test]
    fn failed_reads_keep_input_and_allow_retry() {
        struct Source(usize);
        impl Read for Source {
            fn read(&mut self, dst: &mut [u8]) -> io::Result<usize> {
                assert!(!dst.is_empty());
                dst.fill(0xa5);
                match self.0 {
                    0 => Ok(0),
                    1 => Ok(2),
                    2 => Err(io::ErrorKind::Interrupted.into()),
                    3 => Ok(dst.len() + 1),
                    _ => panic!("source panic"),
                }
            }
        }
        for initialized in [false, true] {
            for outcome in 0..5 {
                let mut buffer = Buffer::with_capacity(8);
                buffer.buf.extend_from_slice(b"ab");
                buffer.buffered = 2;
                if initialized {
                    buffer.buf.resize(8, b'x');
                }
                let result =
                    catch_unwind(AssertUnwindSafe(|| buffer.write_buf(&mut Source(outcome))));
                match outcome {
                    0 | 1 => assert_eq!(result.unwrap().unwrap(), outcome * 2),
                    2 => assert_eq!(
                        result.unwrap().unwrap_err().kind(),
                        io::ErrorKind::Interrupted
                    ),
                    _ => assert!(result.is_err()),
                }
                let expected: &[u8] = if outcome == 1 { b"ab\xa5\xa5" } else { b"ab" };
                assert_eq!(buffer.buffer().to_vec(), expected);
                buffer.write_buf(&mut &b"!"[..]).unwrap();
                assert_eq!(buffer.buffer().to_vec(), [expected, b"!"].concat());
            }
        }
    }

    #[test]
    fn initialized_space_is_reused_after_owned_storage_is_released() {
        struct Inspect;
        impl Read for Inspect {
            fn read(&mut self, dst: &mut [u8]) -> io::Result<usize> {
                assert_eq!(dst, b"abcdefgh");
                dst[..2].copy_from_slice(b"xy");
                Ok(2)
            }
        }
        let mut buffer = Buffer::with_capacity(8);
        buffer.buf.extend_from_slice(b"abcdefgh");
        buffer.buffered = 8;
        buffer.consume(8);
        buffer.write_buf(&mut Inspect).unwrap();
        assert_eq!(buffer.buffer().to_vec(), b"xy");
        // Clear and poison an owned part before returning its storage. Reuse
        // must not expose it as input or trust its previous initialization.
        let mut owned = buffer.take_bytes_to(..2, 0);
        owned.clear();
        for byte in owned.spare_capacity_mut() {
            *byte = std::mem::MaybeUninit::uninit();
        }
        drop(owned);
        buffer.write_buf(&mut &b"next"[..]).unwrap();
        assert_eq!(buffer.buffer().to_vec(), b"next");
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn native_read_buf_tracks_initialization_without_committing_failed_bytes() {
        struct Source(bool);
        impl Read for Source {
            fn read_buf(&mut self, mut dst: io::BorrowedCursor<'_, u8>) -> io::Result<()> {
                assert!(!dst.is_init());
                dst.append(b"cd");
                if self.0 {
                    Ok(())
                } else {
                    Err(io::ErrorKind::Interrupted.into())
                }
            }
        }
        for success in [false, true] {
            let mut buffer = Buffer::with_capacity(8);
            buffer.buf.extend_from_slice(b"ab");
            buffer.buffered = 2;
            assert_eq!(buffer.write_buf(&mut Source(success)).is_ok(), success);
            let expected: &[u8] = if success { b"abcd" } else { b"ab" };
            assert_eq!(buffer.buffer().to_vec(), expected);
            assert_eq!(buffer.buf.len(), 2);
            buffer.write_buf(&mut &b"!"[..]).unwrap();
            assert_eq!(buffer.buffer().to_vec(), [expected, b"!"].concat());
        }
    }
}
