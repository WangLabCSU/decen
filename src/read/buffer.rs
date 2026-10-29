//! Reusable byte buffering and storage management.

use std::io::{self, Read};
use std::mem;
use std::ops::{Bound, Index, RangeBounds};
use std::{ptr, slice};

use bytes::buf::UninitSlice;
use bytes::{Buf, BufMut, Bytes, BytesMut};

/// A byte buffer that reuses space after its data has been consumed.
///
/// Borrow selected bytes without copying them, or split off owned bytes.
/// Consumed space can be reused for subsequent input.
///
/// # Storage
///
/// The current `BytesMut` view is divided into four regions:
///
/// ```text
/// +----------+----------+-----------+----------------+
/// | consumed | buffered | spare_len | spare_capacity |
/// +----------+----------+-----------+----------------+
///            | buffer() |
///                       |          spare()           |
/// |            buf.len()            |
/// |                 buf.capacity()                   |
/// ```
///
/// `consumed` counts bytes already consumed, and `buffered` counts the bytes
/// returned by [`buffer`](Self::buffer). `spare_len` is initialized space
/// after those bytes; `spare_capacity` is the remaining uninitialized capacity.
/// Together, the last two regions make up [`spare`](Self::spare).
///
/// This view excludes storage held by previously split results. `buffer_size`
/// tracks the refill size across splits, not a boundary in the diagram.
///
/// # Examples
///
/// ```text
/// use crate::read::buffer::Buffer;
///
/// let mut buffer = Buffer::with_capacity(8);
/// buffer.extend_from_slice(b"abcdef");
///
/// let view = buffer.take_view(..4, 4);
/// assert_eq!(view, b"abcd");
/// assert_eq!(buffer.buffer(), b"ef");
///
/// buffer.extend_from_slice(b"ghij");
/// assert_eq!(buffer.buffer(), b"efghij");
/// assert_eq!(buffer.capacity(), 8);
/// ```
//
// Space accounting within the current view: consumed | buffered | spare.
// BytesMut owns the accounting for storage outside this view. Even after a
// reallocation it can retain a prefix offset, so capacity() need not describe
// the whole allocation.
//
// Between operations, consumed + buffered <= buf.len() <= buf.capacity().
// buf.len() includes the initialized portion of spare space. Raw appends
// temporarily lower it to borrow capacity as MaybeUninit<u8>, then restore it
// once all bytes up to the new length are initialized.
// write_buf deliberately bypasses this initialization invariant while calling
// Read. A source panic also bypasses that method's length restoration.
//
// Split results own disjoint views that may outlive this buffer. They can clear
// and uninitialize their storage. Reclaiming it must not expose those bytes as
// buffered data.
#[derive(Debug)]
pub struct Buffer {
    // Holds consumed and buffered bytes, plus initialized spare space.
    buf: BytesMut,
    consumed: usize,
    buffered: usize,
    // Refill size retained across splitting; updated when storage is acquired
    // or grows.
    buffer_size: usize,
}

impl Default for Buffer {
    fn default() -> Self {
        Self::new()
    }
}

impl Buffer {
    /// Creates an empty buffer without allocating.
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Creates an empty buffer with at least `capacity` bytes of capacity. Zero
    /// defers allocation until needed.
    ///
    /// # Panics
    ///
    /// Panics if the requested capacity exceeds the slice-size limit.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        let buf = BytesMut::with_capacity(capacity);
        Self {
            buffer_size: buf.capacity(),
            buf,
            consumed: 0,
            buffered: 0,
        }
    }

    /// Returns the buffer's capacity in bytes, including consumed space.
    ///
    /// Use [`spare`](Self::spare) for the space currently available to append
    /// data. Splitting reduces this capacity; reclaiming space can increase it.
    /// It does not measure memory held by previously split results.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    /// Returns the number of bytes in the buffer.
    ///
    /// Bytes removed by [`consume`](Self::consume) are not included.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffered
    }

    /// Returns `true` if the buffer has no bytes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffered == 0
    }

    /// Returns the number of consumed bytes before the data returned by
    /// [`buffer`](Self::buffer).
    ///
    /// This describes the current buffer, not the total number of bytes
    /// consumed over its lifetime.
    #[inline]
    pub fn consumed(&self) -> usize {
        self.consumed
    }

    /// Returns the number of buffered bytes waiting to be processed.
    #[inline]
    pub fn buffered(&self) -> usize {
        self.buffered
    }

    /// Returns the number of bytes available for appending without reserving
    /// space.
    ///
    /// Excludes space occupied by consumed bytes. Call
    /// [`reserve`](Self::reserve) to request more.
    #[inline]
    pub fn spare(&self) -> usize {
        self.buf.capacity() - self.consumed - self.buffered
    }

    // Initialized bytes after the buffered data, available for the next read.
    #[inline]
    fn spare_len(&self) -> usize {
        self.buf.len() - self.consumed - self.buffered
    }

    // Capacity beyond the initialized bytes.
    #[inline]
    fn spare_capacity(&self) -> usize {
        self.buf.capacity() - self.buf.len()
    }

    /// Removes all bytes from the buffer, keeping its capacity for reuse.
    ///
    /// # Examples
    ///
    /// ```text
    /// use crate::read::buffer::Buffer;
    ///
    /// let mut buf = Buffer::with_capacity(32);
    /// buf.extend_from_slice(b"hello");
    /// let capacity = buf.capacity();
    /// buf.clear();
    ///
    /// assert!(buf.is_empty());
    /// assert_eq!(buf.capacity(), capacity);
    /// ```
    #[inline]
    pub fn clear(&mut self) {
        self.consumed = 0;
        self.buffered = 0;
    }

    /// Returns a slice containing the buffer's bytes.
    ///
    /// # Examples
    ///
    /// ```text
    /// use crate::read::buffer::Buffer;
    ///
    /// let mut buf = Buffer::new();
    /// buf.extend_from_slice(b"hello world");
    /// buf.consume(6);
    ///
    /// assert_eq!(buf.buffer(), b"world");
    /// ```
    #[inline]
    pub fn buffer(&self) -> &[u8] {
        // SAFETY: consumed + buffered <= buf.len() places these buffered
        // bytes within initialized buf. BytesMut supplies a non-null
        // pointer even for an empty view; u8 requires only byte alignment.
        // Borrowing self ties the result to the owner and prevents mutation.
        unsafe { slice::from_raw_parts(self.buf.as_ptr().add(self.consumed), self.len()) }
    }

    /// Returns a slice of the bytes in `range` and consumes the first `consume`
    /// bytes of the buffer.
    ///
    /// Both arguments refer to the buffered data before this call. Afterwards,
    /// the buffer contains the bytes from `consume` onwards. The returned slice
    /// can overlap those remaining bytes.
    ///
    /// No bytes are copied. The slice borrows the buffer, so it must no longer
    /// be used before the buffer can be modified again.
    ///
    /// # Examples
    ///
    /// ```text
    /// use crate::read::buffer::Buffer;
    ///
    /// let mut buf = Buffer::new();
    /// buf.extend_from_slice(b"hello world");
    /// let view = buf.take_view(6..11, 6);
    ///
    /// assert_eq!(view, b"world");
    /// assert_eq!(buf.buffer(), b"world");
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `range` is invalid or `consume > self.len()`. The buffer is
    /// unchanged in either case.
    #[inline]
    pub fn take_view(&mut self, range: impl RangeBounds<usize>, consume: usize) -> &[u8] {
        let len = self.len();
        let end = match range.end_bound() {
            Bound::Included(&end) => end.checked_add(1).expect("range end overflow"),
            Bound::Excluded(&end) => end,
            Bound::Unbounded => len,
        };
        let start = match range.start_bound() {
            Bound::Included(&start) => start,
            Bound::Excluded(&start) => start.checked_add(1).expect("range start overflow"),
            Bound::Unbounded => 0,
        };
        assert!(start <= end && end <= len, "view range out of bounds");
        assert!(consume <= len, "cannot consume view past buffered input");
        let offset = self.consumed + start;
        self.consumed += consume;
        self.buffered -= consume;
        // SAFETY: the checked range lies within the previously buffered data.
        // Consuming changes only the counts, so those initialized bytes remain
        // in place. The returned borrow prevents further mutation.
        unsafe { slice::from_raw_parts(self.buf.as_ptr().add(offset), end - start) }
    }

    /// Returns the bytes in `range` as a [`BytesMut`], consuming everything
    /// through the end of that range and another `addition` bytes after it.
    ///
    /// For `start..end`, the result contains the original bytes `[start, end)`,
    /// and the buffer keeps the bytes from `end + addition` onwards. Bytes
    /// before `start` and the `addition` bytes are discarded. All indices are
    /// relative to the buffered data before this call.
    ///
    /// May allocate, but does not copy the bytes. The returned bytes can be
    /// modified independently and remain valid after the buffer is dropped.
    ///
    /// # Examples
    ///
    /// ```text
    /// use crate::read::buffer::Buffer;
    ///
    /// let mut buf = Buffer::new();
    /// buf.extend_from_slice(b"<hello>tail");
    /// let bytes = buf.take_bytes(1..6, 1);
    ///
    /// assert_eq!(&bytes[..], b"hello");
    /// assert_eq!(buf.buffer(), b"tail");
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `range` is invalid or its end plus `addition` exceeds
    /// `self.len()`. The buffer is unchanged in either case.
    // Inline so constant frame bounds can eliminate unused range-handling
    // branches.
    #[inline(always)]
    pub fn take_bytes(&mut self, range: impl RangeBounds<usize>, addition: usize) -> BytesMut {
        let len = self.len();
        let end = match range.end_bound() {
            Bound::Included(&end) => end.checked_add(1).expect("range end overflow"),
            Bound::Excluded(&end) => end,
            Bound::Unbounded => len,
        };
        let start = match range.start_bound() {
            Bound::Included(&start) => start,
            Bound::Excluded(&start) => start.checked_add(1).expect("range start overflow"),
            Bound::Unbounded => 0,
        };
        assert!(start <= end && end <= len, "bytes range out of bounds");
        // end <= len above makes subtraction safe; addition could overflow a
        // sum.
        assert!(addition <= len - end, "cannot consume past buffered input");
        // Split before discarding prefixes: promoting BytesMut to shared
        // storage
        // can allocate. An allocation panic must leave these counts consistent.
        let skip = self.consumed + start;
        let mut bytes = self.buf.split_to(self.consumed + end);
        self.buffered -= end + addition;
        self.consumed = addition;
        if skip != 0 {
            // The checked range puts skip within bytes. split_to has already
            // promoted both halves to shared storage, so advance cannot
            // allocate.
            bytes.advance(skip);
        }
        bytes
    }

    /// Consumes the first `amt` bytes from the buffer.
    ///
    /// # Examples
    ///
    /// ```text
    /// use crate::read::buffer::Buffer;
    ///
    /// let mut buf = Buffer::new();
    /// buf.extend_from_slice(b"hello world");
    /// buf.consume(6);
    /// assert_eq!(buf.buffer(), b"world");
    ///
    /// buf.consume(5);
    /// assert!(buf.is_empty());
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `amt > self.len()`.
    #[inline]
    pub fn consume(&mut self, amt: usize) {
        if amt > 0 {
            assert!(amt <= self.buffered, "cannot consume past buffered input");
            self.consumed += amt;
            self.buffered -= amt;
        }
    }

    /// Discards consumed space.
    ///
    /// May copy buffered bytes to make that space available for appending.
    ///
    /// # Examples
    ///
    /// ```text
    /// use crate::read::buffer::Buffer;
    ///
    /// let mut buf = Buffer::with_capacity(16);
    /// buf.extend_from_slice(b"hello world");
    /// buf.consume(6);
    /// assert_eq!(buf.spare(), 5);
    ///
    /// buf.compact();
    ///
    /// assert_eq!(buf.buffer(), b"world");
    /// assert_eq!(buf.spare(), 11);
    /// ```
    #[inline]
    pub fn compact(&mut self) {
        if self.is_empty() {
            self.clear();
        } else if self.consumed != 0 {
            // Source and destination overlap when less than half the data
            // has been consumed. copy_within supports that overlap.
            self.buf
                .copy_within(self.consumed..(self.consumed + self.buffered), 0);
            self.consumed = 0;
        }
    }

    /// Reserves space to append at least `additional` more bytes.
    ///
    /// More space may be reserved than requested. Automatically discards
    /// consumed bytes when necessary.
    ///
    /// If enough space is already available, this does not allocate or copy.
    /// Otherwise, it may allocate and copy existing bytes.
    ///
    /// # Examples
    ///
    /// ```text
    /// use crate::read::buffer::Buffer;
    ///
    /// let mut buf = Buffer::new();
    /// buf.extend_from_slice(b"old:hello");
    /// buf.consume(4);
    /// buf.reserve(64);
    ///
    /// assert_eq!(buf.buffer(), b"hello");
    /// assert!(buf.spare() >= 64);
    /// buf.extend_from_slice(b" world");
    /// assert_eq!(buf.buffer(), b"hello world");
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the required capacity exceeds the slice-size limit. Consumed
    /// bytes may have been discarded before the panic.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.reuse_storage(additional, true);
    }

    /// Tries to make room for at least `additional` more bytes using existing
    /// storage.
    ///
    /// Returns `true` if enough space is available after the attempt. Returns
    /// `false` if the attempt does not provide enough space. Use
    /// [`reserve`](Self::reserve) if space must be provided.
    ///
    /// Automatically discards consumed bytes when necessary, even if the
    /// attempt returns `false`.
    ///
    /// May copy existing bytes. Does not allocate new byte storage, although
    /// discarding consumed bytes can itself allocate on large buffers.
    /// Retaining split results can prevent reclamation. A failed attempt can
    /// leave less reusable space; call [`compact`](Self::compact) first if
    /// consumed space must remain available.
    ///
    /// # Examples
    ///
    /// ```text
    /// use crate::read::buffer::Buffer;
    ///
    /// let mut buf = Buffer::with_capacity(8);
    /// buf.extend_from_slice(b"abcdefgh");
    /// buf.consume(4);
    ///
    /// assert!(buf.try_reclaim(2));
    /// assert_eq!(buf.buffer(), b"efgh");
    /// assert_eq!(buf.capacity(), 8);
    ///
    /// buf.consume(1);
    /// assert!(!buf.try_reclaim(usize::MAX));
    /// assert_eq!(buf.buffer(), b"fgh");
    /// ```
    #[inline]
    pub fn try_reclaim(&mut self, additional: usize) -> bool {
        self.reuse_storage(additional, false)
    }

    #[inline(always)]
    fn reuse_storage(&mut self, additional: usize, allocate: bool) -> bool {
        if additional <= self.spare() {
            return true;
        }

        // Compact only if it supplies the whole request and moves no more
        // bytes than have been consumed. This prevents repeated small appends
        // from copying the same large remainder on every reservation.
        if additional <= self.spare() + self.consumed && self.consumed >= self.buffered {
            self.compact();
            return true;
        }

        // Keep accounting valid if a subsequent reservation panics.
        // advance can allocate BytesMut's shared header when its inline offset
        // overflows (past 128 MiB - 1 on 32-bit targets). Thus even the
        // allocate=false path does not guarantee zero allocator calls.
        if self.consumed != 0 {
            self.buf.advance(self.consumed);
            self.consumed = 0;
        }

        // BytesMut uses its length to calculate the required capacity and the
        // cost of copying data back into released space. Exclude the spare tail
        // so it does not inflate either: reserve could otherwise allocate more
        // memory, or try_reclaim could fail despite room for buffered +
        // additional.
        // This also discards initialization tracking for the tail.
        //
        // SAFETY: advancing past consumed bytes leaves the initialized buffered
        // bytes at the start of buf. buffered <= buf.len() <= buf.capacity(),
        // so this only shortens the length and exposes no new bytes.
        unsafe { self.buf.set_len(self.buffered) };
        if allocate {
            self.buf.reserve(additional);
            self.buffer_size = self.buffer_size.max(self.capacity());
            true
        } else {
            let reclaimed = self.buf.try_reclaim(additional);
            if reclaimed {
                // Reclaiming does not grow the allocation, but can restore
                // capacity
                // outside the current view. For adopted storage, that can
                // exceed the
                // size recorded by put; keep buffer_size >= capacity()
                // afterwards.
                self.buffer_size = self.buffer_size.max(self.capacity())
            };
            reclaimed
        }
    }

    /// Copies buffered bytes into new storage with room for `additional` more
    /// bytes and returns the previous storage.
    ///
    /// The returned bytes include consumed bytes and initialized spare space.
    /// Consumed bytes are discarded from the buffer.
    ///
    /// # Panics
    ///
    /// Panics if the required capacity exceeds the slice-size limit.
    fn allocate(&mut self, additional: usize) -> BytesMut {
        let len = self.len();
        let capacity = len
            .checked_add(additional)
            .expect("buffer capacity overflow");
        let mut allocated = BytesMut::with_capacity(capacity);
        // SAFETY: capacity is at least len, and buffer() contains len
        // initialized
        // bytes. For nonzero len, the new allocation is disjoint from the still
        // live source allocation. The copy initializes exactly the new length.
        unsafe {
            ptr::copy_nonoverlapping(
                self.buffer().as_ptr(),
                allocated.spare_capacity_mut().as_mut_ptr().cast::<u8>(),
                len,
            );
            allocated.set_len(len);
        }
        self.consumed = 0;
        self.buffered = len;
        self.buffer_size = self.buffer_size.max(allocated.capacity());
        mem::replace(&mut self.buf, allocated)
    }

    /// Ensures at least `len` initialized bytes of append space.
    ///
    /// Fills newly initialized bytes with `val`, leaving already initialized
    /// bytes unchanged. Does not allocate or append data.
    ///
    /// # Panics
    ///
    /// Panics if there is not enough space for the request.
    #[inline]
    pub fn try_init(&mut self, val: u8, len: usize) {
        assert!(len <= self.spare(), "insufficient append space");
        let end = (self.consumed + self.buffered) + len;
        if end > self.buf.len() {
            let additional = end - self.buf.len();
            let spare = self.buf.spare_capacity_mut();

            // SAFETY: the check places end within capacity. Filling exactly
            // additional bytes extends the initialized prefix through end
            // without holes or overwriting existing initialized bytes.
            unsafe {
                ptr::write_bytes(spare.as_mut_ptr(), val, additional);
                self.buf.set_len(end);
            }
        }
    }

    /// Appends bytes from one [`Read::read`] call and returns the number read.
    ///
    /// Makes room as needed before reading. The reader always receives a
    /// nonempty slice.
    ///
    /// # Read contract
    ///
    /// This contract is temporary. It will be removed after migrating to stable
    /// standard-library support for reading into uninitialized buffers.
    ///
    /// `reader` and all its adapters must treat the destination as write-only:
    /// they must not read its existing contents, even when returning an error
    /// or panicking. A successful read must write every byte included in its
    /// returned count.
    ///
    /// # Errors
    ///
    /// Returns errors from the reader, including
    /// [`io::ErrorKind::Interrupted`], without retrying. Returns
    /// [`io::ErrorKind::InvalidData`] if the reader reports more bytes than the
    /// supplied space can hold. The error includes the reported count and the
    /// supplied length. Bytes from a failed read are not appended. Consumed
    /// bytes may be discarded when making room, even if reading fails.
    ///
    /// # Panics
    ///
    /// Panics if buffer capacity overflows. If the reader panics, bytes written
    /// by that call are not appended.
    #[inline(never)]
    pub fn write_buf<R: ?Sized + Read>(&mut self, reader: &mut R) -> io::Result<usize> {
        self.make_room();
        let filled = self.consumed + self.buffered;
        let capacity = self.buf.capacity();
        let available = capacity - filled;

        // Deliberately expose spare capacity without initializing it. This
        // bypasses the initialization requirements of Read.
        unsafe { self.buf.set_len(capacity) };
        let result = reader.read(&mut self.buf[filled..]);

        let nread = match result {
            Ok(nread) if nread <= available => {
                // SAFETY: nread <= capacity - filled bounds the sum by capacity
                // and prevents overflow. This unchecked path assumes Read wrote
                // the reported bytes; retain them after the initialized prefix,
                // excluding the rest of the temporary tail.
                unsafe { self.buf.set_len(filled + nread) };
                nread
            }
            Ok(nread) => {
                // SAFETY: filled <= capacity and 0..filled was initialized
                // before
                // Read borrowed only the tail. Restore that prefix without
                // using
                // the invalid count, excluding all bytes from this failed read.
                unsafe { self.buf.set_len(filled) };
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Read returned {nread} bytes for a buffer of {available} bytes"),
                ));
            }
            Err(error) => {
                // SAFETY: filled bounds the existing initialized prefix within
                // the unchanged capacity. Read could modify only the tail;
                // exclude all of it because an error commits no new bytes.
                unsafe { self.buf.set_len(filled) };
                return Err(error);
            }
        };
        self.buffered += nread;
        Ok(nread)
    }

    // Prepare append space for one read, growing for long frames or recovering
    // space for another batch. write_buf exposes the resulting tail to Read.
    #[inline]
    fn make_room(&mut self) {
        if self.is_empty() {
            // All input was consumed; resetting the counts recovers that space
            // without copying data or changing the initialized length.
            self.clear();
        }

        // Target half the recorded size as append space. Splitting can shrink
        // the current view without lowering this target. This is a reservation
        // threshold, not a prediction of how many bytes Read will return; even
        // a small shortfall can trigger reserve. Zero-capacity buffers request
        // one byte.
        let additional = self.buffer_size.div_ceil(2).max(1);

        // Keep the current view and initialized tail when they supply the
        // batch.
        if additional <= self.spare() {
            return;
        }

        // Compact only when it supplies the whole batch and copies no more
        // bytes than have been consumed. Keep buf.len() unchanged so the
        // initialized space after the moved data remains available for reads.
        if additional <= self.spare() + self.consumed && self.consumed >= self.buffered {
            self.buf
                .copy_within(self.consumed..(self.consumed + self.buffered), 0);
            self.consumed = 0;
            return;
        }

        // Compaction cannot supply the batch cheaply. Exclude consumed bytes
        // before asking BytesMut to reclaim released split space or allocate.
        if self.consumed != 0 {
            self.buf.advance(self.consumed);
            self.consumed = 0;
        }

        // Exclude the spare tail from BytesMut's data length before reserving.
        // Counting it as data would overstate the amount to retain and copy,
        // potentially forcing allocation when released space could be reused.
        // This also discards initialization tracking for the tail.
        //
        // SAFETY: consumed bytes have been removed, so 0..buffered contains the
        // initialized data. buffered <= buf.len() <= buf.capacity(), making
        // this a length reduction that exposes no new bytes.
        unsafe { self.buf.set_len(self.buffered) };

        // With too little tail space and a frame larger than the normal batch,
        // target twice the recorded capacity. BytesMut can satisfy the request
        // by reclaiming released split space or growing the allocation.
        let extra = if self.buffered > additional {
            let capacity = self.buffer_size.saturating_mul(2).min(isize::MAX as usize);
            capacity - self.buf.len()
        } else {
            additional
        };
        self.buf.reserve(extra);
        // Record reservation results without lowering the size after a split.
        // A larger returned capacity also raises future half-size batch
        // requests.
        self.buffer_size = self.buffer_size.max(self.buf.capacity());
    }

    /// Returns the buffered data, reading once from `reader` if the buffer is
    /// empty. May allocate or copy data to make room. See
    /// [`write_buf`](Self::write_buf) for errors and the read contract.
    #[inline]
    pub fn fill_buf<R: ?Sized + Read>(&mut self, reader: &mut R) -> io::Result<&[u8]> {
        if self.is_empty() {
            self.write_buf(reader)?;
        }
        Ok(self.buffer())
    }

    /// Splits the bytes at `at`, returning the first part.
    ///
    /// The returned [`BytesMut`] contains the original bytes `[0, at)`, and the
    /// buffer keeps `[at, len)`, where `len` is the buffer's length before this
    /// call.
    ///
    /// May allocate, but does not copy the bytes. Both parts can be modified
    /// independently. The returned bytes remain valid after the buffer is
    /// dropped.
    ///
    /// # Examples
    ///
    /// ```text
    /// use crate::read::buffer::Buffer;
    ///
    /// let mut buf = Buffer::new();
    /// buf.extend_from_slice(b"hello world");
    /// let first = buf.split_to(5);
    ///
    /// assert_eq!(&first[..], b"hello");
    /// assert_eq!(buf.buffer(), b" world");
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `at > self.len()`.
    #[inline]
    pub fn split_to(&mut self, at: usize) -> BytesMut {
        assert!(
            at <= self.len(),
            "split_to out of bounds: {:?} <= {:?}",
            at,
            self.len(),
        );
        let offset = self.consumed;
        let end = offset + at;

        let mut other = self.buf.split_to(end);

        // Exclude the consumed prefix from the returned bytes.
        if offset != 0 {
            other.advance(offset);
        }

        self.buffered -= at;
        self.consumed = 0;

        other
    }

    /// Splits the bytes at `at`, returning the second part.
    ///
    /// The buffer keeps the original bytes `[0, at)`, and the returned
    /// [`BytesMut`] contains `[at, len)`, where `len` is the buffer's length
    /// before this call.
    ///
    /// May allocate, but does not copy the bytes. Both parts can be modified
    /// independently. The returned bytes remain valid after the buffer is
    /// dropped.
    ///
    /// # Examples
    ///
    /// ```text
    /// use crate::read::buffer::Buffer;
    ///
    /// let mut buf = Buffer::new();
    /// buf.extend_from_slice(b"hello world");
    /// let second = buf.split_off(5);
    ///
    /// assert_eq!(buf.buffer(), b"hello");
    /// assert_eq!(&second[..], b" world");
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `at > self.len()`.
    #[inline]
    pub fn split_off(&mut self, at: usize) -> BytesMut {
        assert!(
            at <= self.len(),
            "split_off out of bounds: {:?} <= {:?}",
            at,
            self.len(),
        );

        let split = self.consumed + at;
        let mut other = self.buf.split_off(split);
        other.truncate(self.buffered - at);
        self.buffered = at;
        other
    }

    /// Copies the bytes in `extend` to the end of the buffer.
    ///
    /// Reserves additional space if needed.
    ///
    /// # Examples
    ///
    /// ```text
    /// use crate::read::buffer::Buffer;
    ///
    /// let mut buf = Buffer::new();
    /// buf.extend_from_slice(b"hello");
    /// buf.extend_from_slice(b" world");
    ///
    /// assert_eq!(buf.buffer(), b"hello world");
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the required capacity exceeds the slice-size limit.
    #[inline]
    pub fn extend_from_slice(&mut self, extend: &[u8]) {
        let cnt = extend.len();
        self.reserve(cnt);
        let len = self.buf.len();
        let filled = self.consumed + self.buffered;

        // SAFETY: reserve guarantees filled + cnt <= capacity. extend is a
        // valid &[u8] containing cnt readable bytes. It cannot borrow this
        // view while this method holds &mut self. If it comes from a previously
        // split Item sharing the allocation, that Item's region is excluded
        // from this view's capacity. In both cases it is disjoint from the
        // destination filled..filled+cnt.
        // set_len(0) changes only metadata; it does not erase the old bytes.
        // The spare slice then covers the full capacity as MaybeUninit<u8>,
        // giving the destination pointer access to the entire reserved range.
        // No allocation or user code runs between taking the pointer and
        // copying.
        // Old bytes 0..len remain initialized, and the copy initializes
        // filled..filled+cnt. Since filled <= len, their union has no gap,
        // so restoring max(len, filled+cnt) exposes only initialized bytes.
        unsafe {
            self.buf.set_len(0);
            let dst = self.buf.spare_capacity_mut().as_mut_ptr().cast::<u8>();
            ptr::copy_nonoverlapping(extend.as_ptr(), dst.add(filled), cnt);
            self.buf.set_len(len.max(filled + cnt));
        }
        self.buffered += cnt;
    }

    /// Copies the bytes in `range` and appends them to the end of the buffer.
    ///
    /// The range is relative to [`buffer`](Self::buffer), so index zero is its
    /// first byte. Reserves additional space if needed.
    ///
    /// # Examples
    ///
    /// ```text
    /// use crate::read::buffer::Buffer;
    ///
    /// let mut buf = Buffer::new();
    /// buf.extend_from_slice(b"old:abc");
    /// buf.consume(4);
    /// buf.extend_from_within(1..3);
    ///
    /// assert_eq!(buf.buffer(), b"abcbc");
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `range` is invalid or the required capacity exceeds the
    /// slice-size limit.
    #[inline]
    pub fn extend_from_within(&mut self, range: impl RangeBounds<usize>) {
        let start = match range.start_bound() {
            Bound::Included(&start) => start,
            Bound::Excluded(&start) => start.checked_add(1).expect("range start overflow"),
            Bound::Unbounded => 0,
        };
        let extend = self
            .buffer()
            .index((Bound::Included(start), range.end_bound().cloned()));
        // Keep offsets, not pointers, across reserve: it can move
        // the buffered input or replace the allocation.
        let cnt = extend.len();
        self.reserve(cnt);
        let len = self.buf.len();
        let src_start = self.consumed + start;
        let dst_start = self.consumed + self.buffered;

        // SAFETY: indexing input with the bounds tuple checks the range and
        // returns a subslice, so start + cnt <= buffered.
        // reserve preserves buffered data and its length, although it may move
        // the data, so offsets are computed using the updated consumed value.
        // Source:      consumed+start .. consumed+start+cnt
        // Destination: dst_start .. dst_start+cnt
        // src_start + cnt <= dst_start, so these ranges cannot overlap.
        // The source is initialized and reserve provides enough destination
        // space.
        // set_len(0) lets both pointers derive from the same full-capacity
        // borrow;
        // no operation in between can move the allocation. Old bytes 0..len
        // stay
        // initialized. Since dst_start <= len, the copy extends initialization
        // without a gap before restoring the larger length.
        unsafe {
            self.buf.set_len(0);
            let ptr = self.buf.spare_capacity_mut().as_mut_ptr().cast::<u8>();
            ptr::copy_nonoverlapping(ptr.add(src_start), ptr.add(dst_start), cnt);
            self.buf.set_len(len.max(dst_start + cnt));
        }
        self.buffered += cnt;
    }
}

impl Buf for Buffer {
    #[inline]
    fn advance(&mut self, cnt: usize) {
        assert!(
            cnt <= self.remaining(),
            "cannot advance past remaining input"
        );
        self.consumed += cnt;
        self.buffered -= cnt;
    }

    #[inline]
    fn remaining(&self) -> usize {
        self.len()
    }

    #[inline]
    fn chunk(&self) -> &[u8] {
        self.buffer()
    }

    /// Returns the first `len` bytes as an immutable [`Bytes`], removing them
    /// from the buffer.
    ///
    /// The buffer keeps the bytes from `len` onwards. May allocate, but does
    /// not copy the bytes. The result remains valid after the buffer is
    /// dropped. Passing zero returns an empty result and leaves the buffer
    /// unchanged.
    ///
    /// # Examples
    ///
    /// ```text
    /// use bytes::Buf;
    ///
    /// use crate::read::buffer::Buffer;
    ///
    /// let mut buf = Buffer::new();
    /// buf.extend_from_slice(b"hello world");
    /// let bytes = buf.copy_to_bytes(5);
    ///
    /// assert_eq!(&bytes[..], b"hello");
    /// assert_eq!(buf.buffer(), b" world");
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if `len > self.len()`.
    #[inline]
    fn copy_to_bytes(&mut self, len: usize) -> Bytes {
        if len == 0 {
            return Bytes::new();
        }
        self.split_to(len).freeze()
    }
}

// SAFETY: chunks cover only spare storage owned by this buffer. UninitSlice
// prevents safe callers from reading uninitialized bytes or uninitializing
// existing bytes. advance_mut checks capacity and exposes only the prefix
// initialized by its caller; remaining_mut includes space obtainable by growth.
unsafe impl BufMut for Buffer {
    /// Returns the number of bytes that can be appended, allowing the buffer to
    /// grow.
    #[inline]
    fn remaining_mut(&self) -> usize {
        // BufMut reports how much can be appended with growth, not just the
        // current tail. Consumed bytes can be discarded before growing.
        isize::MAX as usize - self.len()
    }

    /// Appends the next `cnt` bytes previously written into the spare space.
    ///
    /// # Safety
    ///
    /// The caller must initialize these bytes before committing them and must
    /// not move or discard them between writing and this call.
    ///
    /// # Panics
    ///
    /// Panics if `cnt` exceeds the available spare space.
    #[inline]
    unsafe fn advance_mut(&mut self, cnt: usize) {
        // Keep panic formatting outside the inlined commit path, as BytesMut
        // does.
        #[cold]
        #[inline(never)]
        fn panic_cold(cnt: usize, available: usize) -> ! {
            panic!(
                "cannot advance past spare capacity: {:?} <= {:?}",
                cnt, available
            );
        }

        if cnt <= self.spare_len() {
            // This range already lies within buf.len(), hence within capacity.
            // The initialized spare count bounds the increase in buffered
            // bytes. The initialized byte count itself stays unchanged.
            self.buffered += cnt;
            return;
        }
        let available = self.spare();
        if cnt > available {
            panic_cold(cnt, available);
        }
        let end = (self.consumed + self.buffered) + cnt;
        // SAFETY: the check keeps end within buf.capacity(). The caller
        // initialized cnt bytes immediately after consumed + buffered, which
        // does not exceed the old buf.len(). Together with the old bytes,
        // these cover every byte exposed by the new length.
        unsafe { self.buf.set_len(end) };
        self.buffered += cnt;
    }

    /// Returns spare space for appending bytes; its length may be less than
    /// [`spare`](Self::spare). Write the bytes before committing them with
    /// [`BufMut::advance_mut`]. May allocate or copy data to make room.
    ///
    /// ```text
    /// use bytes::BufMut;
    ///
    /// use crate::read::buffer::Buffer;
    ///
    /// let mut buffer = Buffer::with_capacity(8);
    /// buffer.extend_from_slice(b"ab");
    /// buffer.chunk_mut()[..2].copy_from_slice(b"cd");
    /// assert_eq!(buffer.buffer(), b"ab"); // Written, but not committed yet.
    /// // SAFETY: the next two spare bytes were initialized above, and the
    /// // buffer's positions and storage have not changed since the write.
    /// unsafe { buffer.advance_mut(2) };
    /// assert_eq!(buffer.buffer(), b"abcd");
    /// ```
    #[inline]
    fn chunk_mut(&mut self) -> &mut UninitSlice {
        if self.spare() == 0 {
            self.reserve(64);
        }

        if self.spare_len() != 0 {
            // SAFETY: consumed + buffered <= buf.len() bounds the initialized
            // tail. &mut self gives exclusive access to this view; split Items
            // own
            // disjoint regions of the allocation.
            unsafe {
                self.buf
                    .get_unchecked_mut((self.consumed + self.buffered)..)
                    .into()
            }
        } else {
            self.buf.spare_capacity_mut().into()
        }
    }

    /// Appends all remaining bytes from `src`. May allocate and copy bytes, or
    /// reuse owned source storage.
    fn put<T: Buf>(&mut self, mut src: T)
    where
        Self: Sized,
    {
        if self.buf.capacity() == 0 && src.has_remaining() {
            let bytes = src.copy_to_bytes(src.remaining());
            // An adapter such as Take<Bytes> may still own the unused suffix.
            // Drop it before conversion so it does not prevent unique
            // ownership.
            drop(src);
            // buf.capacity() == 0 implies consumed == buffered == buf.len() ==
            // 0:
            // there is no buffered input or initialized tail to preserve here.
            // From<Bytes> handles both adoption and copying without a separate
            // is_unique / try_into_mut check and its extra virtual call.
            self.buf = BytesMut::from(bytes);
            self.buffer_size = self.buffer_size.max(self.capacity());
            self.buffered = self.buf.len();
            return;
        }
        self.reserve(src.remaining());
        while src.has_remaining() {
            let chunk = src.chunk();
            let len = chunk.len();
            self.extend_from_slice(chunk);
            src.advance(len);
        }
    }

    #[inline]
    fn put_slice(&mut self, src: &[u8]) {
        self.extend_from_slice(src);
    }

    #[inline]
    fn put_bytes(&mut self, val: u8, cnt: usize) {
        self.reserve(cnt);
        let start = self.consumed + self.buffered;
        let end = start + cnt;
        let len = self.buf.len();
        // SAFETY: reserve ensures end <= capacity. Temporarily setting length
        // to zero lets spare_capacity_mut lend the full capacity as
        // MaybeUninit<u8>; it does not erase existing bytes. The write covers
        // start..end, entirely inside that exclusive slice. Old bytes 0..len
        // outside the write remain initialized. Since start <= len, these
        // regions cover 0..max(len,end) without gaps, so restoring it is valid.
        unsafe {
            self.buf.set_len(0);
            let dst = self.buf.spare_capacity_mut().as_mut_ptr().cast::<u8>();
            ptr::write_bytes(dst.add(start), val, cnt);
            self.buf.set_len(len.max(end));
        }
        self.buffered += cnt;
    }
}

#[cfg(test)]
mod test {
    use std::io::{self, Read};
    use std::mem::MaybeUninit;
    use std::ops::Bound::{Excluded, Included, Unbounded};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use bytes::{Buf, BufMut, BytesMut};

    use super::Buffer;
    use crate::test::allocations;

    fn segments(buffer: &Buffer) -> [usize; 3] {
        let parts = [buffer.consumed(), buffer.buffered(), buffer.spare()];
        assert_eq!(parts.iter().sum::<usize>(), buffer.capacity());
        parts
    }

    #[test]
    fn make_room_prepares_zero_capacity_and_grows_full_buffers() {
        let mut buffer = Buffer::new();
        buffer.make_room();
        assert!(buffer.spare() > 0);
        assert_eq!(buffer.spare_len(), 0);
        for _ in 0..3 {
            let data = vec![b'x'; buffer.capacity()];
            buffer.clear();
            buffer.extend_from_slice(&data);
            buffer.make_room();
            assert!(buffer.spare() >= data.len());
            assert_eq!(buffer.buffer(), data);
            assert_eq!(buffer.spare(), buffer.capacity() - data.len());
            assert_eq!(buffer.spare_len(), 0);
        }
    }

    #[test]
    fn make_room_reuses_consumed_space_after_splitting() {
        let mut buffer = Buffer::with_capacity(8);
        buffer.extend_from_slice(b"abcdefgh");
        let address = buffer.buf.as_ptr();
        buffer.consume(4);
        buffer.make_room();
        assert_eq!(buffer.buf.as_ptr(), address);
        assert_eq!(buffer.buffer(), b"efgh");
        assert_eq!(buffer.spare(), 4);

        let retained = buffer.split_to(4);
        buffer.make_room();
        assert_eq!(buffer.capacity(), 4);
        buffer.extend_from_slice(b"ijkl");
        buffer.consume(4);
        buffer.make_room();
        assert_eq!(buffer.capacity(), 4);
        assert_eq!(buffer.spare(), 4);
        assert_eq!(&retained[..], b"efgh");
    }

    #[test]
    fn make_room_recovers_released_split_space() {
        let mut buffer = Buffer::with_capacity(64);
        buffer.extend_from_slice(&[b'x'; 64]);
        let address = buffer.buf.as_ptr();
        drop(buffer.split_off(8));
        assert_eq!(buffer.capacity(), 8);
        buffer.make_room();
        assert_eq!(buffer.buf.as_ptr(), address);
        assert_eq!(buffer.buffer(), &[b'x'; 8]);
        assert!(buffer.spare() >= 8);
    }

    #[test]
    fn write_buf_refills_shared_views_with_short_or_exhausted_tails() {
        for (filled, split) in [(64, 32), (63, 32), (64, 56)] {
            let mut buffer = Buffer::with_capacity(64);
            buffer.extend_from_slice(&vec![b'x'; filled]);
            let retained = buffer.split_to(split);
            let mut source = &[b'y'; 32][..];
            assert_eq!(buffer.write_buf(&mut source).unwrap(), 32);
            assert!(source.is_empty());
            assert_eq!(
                &buffer.buffer()[..filled - split],
                vec![b'x'; filled - split]
            );
            assert_eq!(&buffer.buffer()[filled - split..], &[b'y'; 32]);
            assert_eq!(&retained[..], vec![b'x'; split]);
        }
    }

    #[test]
    fn make_room_doubles_the_remaining_shared_view() {
        let data: Vec<_> = (0..64).collect();
        for filled in [63, 64] {
            for (split, consumed) in [(24, 24), (32, 16), (32, 24), (40, 16)] {
                let mut buffer = Buffer::with_capacity(64);
                buffer.extend_from_slice(&data[..filled]);
                let retained = buffer.split_to(split);
                buffer.consume(consumed);
                let capacity = buffer.capacity() - consumed;

                let (_, count) = allocations(|| {
                    buffer.make_room();
                    buffer.make_room();
                });
                assert_eq!(count, 1);
                assert!(buffer.capacity() >= capacity * 2);
                assert_eq!(buffer.buffer_size, 64);
                assert!(buffer.spare() >= capacity);
                assert_eq!(buffer.buffer(), &data[split + consumed..filled]);
                assert_eq!(&retained[..], &data[..split]);
            }
        }
    }

    #[test]
    fn make_room_grows_when_a_released_suffix_cannot_supply_a_read_batch() {
        let mut buffer = Buffer::with_capacity(64);
        buffer.extend_from_slice(&[b'x'; 64]);
        let retained = buffer.split_to(48);
        buffer.reserve(24);
        buffer.extend_from_slice(&[b'y'; 8]);
        assert_eq!(buffer.capacity(), 40);
        assert_eq!(buffer.buffer_size, 64);

        let mut suffix = buffer.split_off(24);
        suffix.spare_capacity_mut().fill(MaybeUninit::uninit());
        drop(suffix);
        let (_, count) = allocations(|| {
            buffer.make_room();
            buffer.make_room();
        });
        assert_eq!(count, 1);
        assert!(buffer.capacity() >= 56);
        assert_eq!(buffer.buffer_size, buffer.capacity());
        assert!(buffer.spare() >= 32);
        assert_eq!(buffer.spare_len(), 0);
        assert_eq!(&buffer.buffer()[..16], &[b'x'; 16]);
        assert_eq!(&buffer.buffer()[16..], &[b'y'; 8]);
        assert_eq!(&retained[..], &[b'x'; 48]);
    }

    #[test]
    fn acquired_storage_sets_the_refill_size_after_splitting() {
        for requested in [8, 128] {
            for operation in 0..3 {
                let mut buffer = Buffer::new();
                match operation {
                    0 => buffer.reserve(requested),
                    1 => drop(buffer.allocate(requested)),
                    _ => {
                        let mut source = BytesMut::with_capacity(requested);
                        source.extend_from_slice(&vec![b'x'; requested - 1]);
                        buffer.put(source);
                    }
                }
                let capacity = buffer.capacity();
                buffer.extend_from_slice(&vec![b'x'; capacity - 1 - buffer.len()]);
                let retained = buffer.split_to(buffer.len());
                assert_eq!(buffer.capacity(), 1);
                let batch = vec![b'y'; capacity.div_ceil(2)];
                assert_eq!(
                    buffer.write_buf(&mut batch.as_slice()).unwrap(),
                    batch.len()
                );
                assert_eq!(buffer.buffer_size, capacity);
                assert_eq!(buffer.buffer(), batch);
                assert_eq!(&retained[..], vec![b'x'; capacity - 1]);
            }
        }
    }

    #[test]
    fn write_buf_refills_space_released_by_a_modified_split() {
        let mut buffer = Buffer::with_capacity(64);
        buffer.extend_from_slice(&[b'x'; 64]);
        let mut prefix = buffer.split_to(32);
        prefix.clear();
        // The split owner may write through spare_capacity_mut, so its former
        // bytes cannot be assumed initialized after reclamation.
        prefix.spare_capacity_mut()[..32].fill(MaybeUninit::uninit());
        drop(prefix);
        buffer.make_room();
        assert_eq!(buffer.buffer(), &[b'x'; 32]);
        assert_eq!(buffer.spare(), 32);
        assert_eq!(buffer.write_buf(&mut &b"yy"[..]).unwrap(), 2);
        assert_eq!(&buffer.buffer()[..32], &[b'x'; 32]);
        assert_eq!(&buffer.buffer()[32..], b"yy");
        assert_eq!(buffer.buf.len(), 34);
        assert_eq!(buffer.spare_len(), 0);
    }

    #[test]
    fn segments_exclude_space_transferred_by_splitting() {
        let mut buffer = Buffer::with_capacity(32);
        buffer.extend_from_slice(b"abcdefghijkl");
        assert_eq!(segments(&buffer), [0, 12, 20]);
        buffer.consume(2);
        assert_eq!(segments(&buffer), [2, 10, 20]);
        let item = buffer.take_bytes(1..4, 1);
        assert_eq!(&item[..], b"def");
        assert_eq!(segments(&buffer), [1, 5, 20]);
        drop(item);
        assert_eq!(segments(&buffer), [1, 5, 20]);
        buffer.advance(2);
        assert_eq!(segments(&buffer), [3, 3, 20]);
        let suffix = buffer.split_off(1);
        assert_eq!(&suffix[..], b"kl");
        assert_eq!(segments(&buffer), [3, 1, 0]);
        buffer.reserve(0);
        assert_eq!(segments(&buffer), [3, 1, 0]);
        buffer.compact();
        assert_eq!(segments(&buffer), [0, 1, 3]);
        assert_eq!(buffer.buffer(), b"j");
        drop(buffer.allocate(8));
        assert_eq!(segments(&buffer), [0, 1, 8]);
        assert_eq!(&suffix[..], b"kl");
    }

    #[test]
    fn reserve_after_advancing_allows_later_prefix_recovery() {
        let mut buffer = Buffer::with_capacity(64);
        buffer.extend_from_slice(&[0x41; 64]);
        buffer.consume(8);
        buffer.reserve(100);
        assert_eq!(buffer.consumed(), 0);
        let capacity = buffer.capacity();
        let address = buffer.buf.as_ptr();
        drop(buffer.split_to(48));
        assert!(buffer.try_reclaim(capacity - buffer.len()));
        assert_eq!(buffer.buf.as_ptr(), address.wrapping_sub(8));
        assert_eq!(buffer.capacity(), capacity + 8);
        assert_eq!(buffer.buffer(), &[0x41; 8]);
    }

    #[test]
    fn suffix_reclamation_increases_capacity_without_copying() {
        for reserve in [false, true] {
            let mut buffer = Buffer::with_capacity(100);
            buffer.extend_from_slice(&[0x41; 8]);
            let suffix = buffer.split_off(8);
            let address = buffer.buf.as_ptr();
            assert_eq!(segments(&buffer), [0, 8, 0]);
            assert!(!buffer.try_reclaim(1));
            assert_eq!(segments(&buffer), [0, 8, 0]);
            drop(suffix);
            for additional in [1, 8, 32, 92] {
                if reserve {
                    buffer.reserve(additional);
                } else {
                    assert!(buffer.try_reclaim(additional));
                }
                assert_eq!(buffer.buf.as_ptr(), address);
                assert!(buffer.capacity() >= 8 + additional);
                assert_eq!(buffer.capacity(), buffer.len() + buffer.spare());
                assert!(buffer.spare() >= additional);
                let parts = segments(&buffer);
                buffer.reserve(0);
                assert!(buffer.try_reclaim(0));
                assert_eq!(segments(&buffer), parts);
            }
            assert_eq!(buffer.buffer(), &[0x41; 8]);
        }
    }

    #[test]
    fn adopted_storage_can_reclaim_a_prefix_before_its_initial_view() {
        let mut source = BytesMut::with_capacity(64);
        source.extend_from_slice(&[0x41; 64]);
        let address = source.as_ptr();
        drop(source.split_to(48));
        let mut buffer = Buffer::new();
        buffer.put(source);
        assert_eq!(buffer.buffer(), &[0x41; 16]);
        let item = buffer.split_to(8);
        assert!(!buffer.try_reclaim(56));
        drop(item);
        assert!(buffer.try_reclaim(56));
        assert_eq!(buffer.buf.as_ptr(), address);
        assert_eq!(segments(&buffer), [0, 8, 56]);
    }

    #[test]
    fn buf_mut_chunks_commit_written_bytes_and_reuse_tail_space() {
        for initialized in [0, 4, 8] {
            let mut buffer = Buffer::with_capacity(8);
            buffer.try_init(0xa5, initialized);
            buffer.extend_from_slice(b"ab");
            let address = buffer.buffer().as_ptr();
            buffer.consume(2);

            assert_eq!(
                buffer.chunk_mut().as_mut_ptr(),
                address.wrapping_add(2).cast_mut()
            );
            assert_eq!(buffer.capacity(), 8);
            assert!(buffer.is_empty());
            let initialized = buffer.buf.len();
            // SAFETY: advancing by zero requires no initialized bytes.
            unsafe { buffer.advance_mut(0) };
            assert_eq!(buffer.buf.len(), initialized);

            let input: Vec<u8> = (0..80).collect();
            let mut written = 0;
            while written < input.len() {
                let remaining = buffer.remaining_mut();
                let chunk = buffer.chunk_mut();
                assert!(chunk.len() > 0);
                assert!(chunk.len() <= remaining);
                let count = chunk.len().min(input.len() - written);
                chunk[..count].copy_from_slice(&input[written..written + count]);
                assert_eq!(buffer.buffer(), &input[..written]);
                // SAFETY: the next count bytes were initialized through
                // chunk_mut;
                // no intervening operation has moved or discarded them.
                unsafe { buffer.advance_mut(count) };
                written += count;
                assert_eq!(buffer.remaining_mut(), remaining - count);
                assert_eq!(buffer.buffer(), &input[..written]);
                assert!((buffer.consumed + buffer.buffered) <= buffer.buf.len());
            }
        }
        let mut empty = Buffer::new();
        assert!(empty.chunk_mut().len() >= 64);
        assert!(empty.is_empty());
    }

    #[test]
    fn buf_mut_put_methods_preserve_initialized_tail_and_split_data() {
        for initialized in [false, true] {
            let mut buffer = Buffer::with_capacity(16);
            if initialized {
                buffer.try_init(0xa5, 16);
            }
            buffer.extend_from_slice(b"old:ab");
            let retained = buffer.split_to(4);
            buffer.consume(1);

            buffer.put_slice(b"cd");
            buffer.put_bytes(b'!', 2);
            assert_eq!(buffer.buffer(), b"bcd!!");
            if initialized {
                assert_eq!(&buffer.buf[buffer.consumed + buffer.buffered..], &[0xa5; 6]);
            }
            buffer.put_bytes(b'?', 20);
            buffer.put(Buf::chain(&b"ef"[..], &b"gh"[..]));
            buffer.put_u16(0x696a);
            assert_eq!(
                buffer.buffer(),
                [b"bcd!!".as_slice(), &[b'?'; 20], b"efghij"].concat()
            );
            let state = (
                buffer.consumed,
                (buffer.consumed + buffer.buffered),
                buffer.buf.len(),
                buffer.capacity(),
            );
            buffer.put_slice(b"");
            buffer.put_bytes(0, 0);
            buffer.put(&b""[..]);
            assert_eq!(
                (
                    buffer.consumed,
                    (buffer.consumed + buffer.buffered),
                    buffer.buf.len(),
                    buffer.capacity()
                ),
                state
            );
            drop(buffer);
            assert_eq!(&retained[..], b"old:");
        }
    }

    #[test]
    fn buf_mut_put_reuses_owned_source_and_accepts_shared_source() {
        for shared in [false, true] {
            let mut source = BytesMut::with_capacity(16);
            source.extend_from_slice(b"abcdef");
            let source = source.freeze();
            let retained = shared.then(|| source.clone());
            let address = source.as_ptr();
            let mut buffer = Buffer::new();
            // Take retains the suffix until dropped; put must release it before
            // attempting to take ownership of the selected bytes.
            buffer.put(source.take(3));
            assert_eq!(buffer.buffer(), b"abc");
            if !shared {
                assert_eq!(buffer.buffer().as_ptr(), address);
            }
            buffer.put_slice(b"!");
            assert_eq!(buffer.buffer(), b"abc!");
            if let Some(retained) = retained {
                assert_eq!(&retained[..], b"abcdef");
            }
        }
    }

    #[test]
    fn take_bytes_selects_without_copying_and_consumes_the_extra_prefix() {
        use std::ops::Bound::{Excluded, Included};

        for initialized in [false, true] {
            let mut buffer = Buffer::with_capacity(32);
            if initialized {
                let spare = buffer.spare();
                buffer.try_init(0, spare);
                buffer.write_buf(&mut io::empty()).unwrap();
            }
            buffer.extend_from_slice(b"old:abcdefgh");
            buffer.consume(4);
            let address = buffer.buffer().as_ptr();
            let initialized_spare = buffer.spare_len();

            let bytes = buffer.take_bytes((Excluded(0), Included(3)), 2);
            assert_eq!(&bytes[..], b"bcd");
            assert_eq!(bytes.as_ptr(), address.wrapping_add(1));
            assert_eq!(buffer.buffer(), b"gh");
            assert_eq!(buffer.spare_len(), initialized_spare);

            buffer.clear();
            buffer.extend_from_slice(&[b'!'; 128]);
            drop(buffer);
            assert_eq!(&bytes[..], b"bcd");
        }
    }

    #[test]
    fn take_bytes_handles_empty_full_and_invalid_ranges() {
        use std::ops::Bound::{Excluded, Included, Unbounded};

        assert!(Buffer::new().take_bytes(.., 0).is_empty());
        let mut buffer = Buffer::new();
        buffer.extend_from_slice(b"abcdef");
        assert!(buffer.take_bytes(2..2, 1).is_empty());
        assert_eq!(buffer.buffer(), b"def");
        assert_eq!(&buffer.take_bytes(.., 0)[..], b"def");
        assert!(buffer.is_empty());

        for (range, addition) in [
            ((Included(3), Excluded(2)), 0),
            ((Unbounded, Excluded(7)), 0),
            ((Excluded(usize::MAX), Unbounded), 0),
            ((Unbounded, Included(usize::MAX)), 0),
            ((Unbounded, Excluded(2)), 5),
            ((Unbounded, Excluded(1)), usize::MAX),
            ((Unbounded, Excluded(2)), usize::MAX - 1),
        ] {
            let mut buffer = Buffer::new();
            buffer.extend_from_slice(b"abcdef");
            let address = buffer.buffer().as_ptr();
            let state = (
                buffer.consumed,
                (buffer.consumed + buffer.buffered),
                buffer.buf.len(),
                buffer.capacity(),
            );
            assert!(catch_unwind(AssertUnwindSafe(|| buffer.take_bytes(range, addition))).is_err());
            assert_eq!(buffer.buffer(), b"abcdef");
            assert_eq!(buffer.buffer().as_ptr(), address);
            assert_eq!(
                (
                    buffer.consumed,
                    (buffer.consumed + buffer.buffered),
                    buffer.buf.len(),
                    buffer.capacity()
                ),
                state,
            );
        }
    }

    #[test]
    fn copy_to_bytes_shares_only_buffered_input_and_preserves_initialized_tail() {
        for initialized in [false, true] {
            for count in [1, 3, 6] {
                let mut buffer = Buffer::with_capacity(16);
                if initialized {
                    let spare = buffer.spare();
                    buffer.try_init(0, spare);
                    buffer.write_buf(&mut io::empty()).unwrap();
                }
                buffer.extend_from_slice(b"abcdefgh");
                buffer.consume(2);
                let address = buffer.buffer().as_ptr();
                let initialized_spare = buffer.spare_len();

                let bytes = buffer.copy_to_bytes(count);
                assert_eq!(bytes.as_ptr(), address, "extraction must not copy input");
                assert_eq!(&bytes[..], &b"cdefgh"[..count]);
                assert_eq!(buffer.buffer(), &b"cdefgh"[count..]);
                assert_eq!(buffer.consumed(), 0);
                assert_eq!(buffer.spare_len(), initialized_spare);

                // A retained view must survive both reuse and growth of the
                // owner.
                buffer.extend_from_slice(b"ij");
                buffer.reserve(128);
                buffer.extend_from_slice(&[b'!'; 64]);
                drop(buffer);
                assert_eq!(&bytes[..], &b"cdefgh"[..count]);
            }
        }
    }

    #[test]
    fn copy_to_bytes_zero_and_invalid_lengths_preserve_buffer_state() {
        let mut buffer = Buffer::with_capacity(16);
        let spare = buffer.spare();
        buffer.try_init(0, spare);
        buffer.write_buf(&mut &b"abcdefgh"[..]).unwrap();
        buffer.consume(2);
        let before = (
            buffer.buffer().as_ptr(),
            buffer.consumed,
            (buffer.consumed + buffer.buffered),
            buffer.buf.len(),
            buffer.capacity(),
        );
        assert!(buffer.copy_to_bytes(0).is_empty());
        for count in [7, usize::MAX] {
            assert!(catch_unwind(AssertUnwindSafe(|| buffer.copy_to_bytes(count))).is_err());
        }
        assert_eq!(buffer.buffer(), b"cdefgh");
        assert_eq!(
            (
                buffer.buffer().as_ptr(),
                buffer.consumed,
                (buffer.consumed + buffer.buffered),
                buffer.buf.len(),
                buffer.capacity(),
            ),
            before
        );
        assert!(Buffer::new().copy_to_bytes(0).is_empty());
    }

    #[test]
    fn reclaim_of_an_empty_view_discards_initialized_tail() {
        let mut slab = Buffer::with_capacity(8192);
        slab.extend_from_slice(&vec![0xa5; 2000]);
        drop(slab.split_to(1500));
        slab.clear();
        assert!(slab.try_reclaim(8192));
        assert!(slab.buffer().is_empty());
        assert!(slab.buf.is_empty());
        assert_eq!(slab.spare_len(), 0);
        assert_eq!(slab.spare(), 8192);
        assert_eq!(slab.capacity(), 8192);
        slab.extend_from_slice(b"next");
        assert_eq!(slab.buffer(), b"next");
        assert_eq!(slab.buf.len(), 4);
    }

    #[test]
    fn reserve_preserves_input_after_shared_prefix_is_uninitialized() {
        let mut buffer = Buffer::with_capacity(128);
        buffer.extend_from_slice(&[b'e'; 128]);
        let mut prefix = buffer.split_to(64);
        assert_eq!(buffer.capacity(), 64);
        // Previously split bytes need not remain initialized after their owner
        // drops. Reclaiming storage must not expose that region as initialized.
        prefix.clear();
        prefix.spare_capacity_mut().fill(MaybeUninit::uninit());
        drop(prefix);
        buffer.reserve(64);
        assert!(buffer.spare() >= 64);
        assert_eq!(buffer.buffer(), &[b'e'; 64]);
        assert_eq!(buffer.buf.len(), 64);
        let spare = buffer.spare();
        buffer.try_init(0, spare);
        buffer.write_buf(&mut &b"ij"[..]).unwrap();
        assert_eq!(&buffer.buffer()[..64], &[b'e'; 64]);
        assert_eq!(&buffer.buffer()[64..], b"ij");
    }

    #[test]
    fn reserved_capacity_is_separate_from_initialized_spare() {
        let mut buffer = Buffer::with_capacity(8);
        buffer.extend_from_slice(b"ab");
        let spare = buffer.spare();
        buffer.try_init(0, spare);
        buffer.reserve(16);
        assert_eq!(buffer.buf.len(), 2);
        assert_eq!(buffer.spare_len(), 0);
        assert!(buffer.spare() >= 16);
        buffer.try_init(0, 6);
        assert_eq!(buffer.spare_len(), 6);
        buffer.buf[buffer.consumed + buffer.buffered..].copy_from_slice(b"cdefgh");
        assert_eq!(buffer.spare_len(), 6);
        let spare = buffer.buf.spare_capacity_mut();
        // Safe callers may leave these bytes uninitialized. Neither buffered
        // data
        // nor the initialized tail may expose uninitialized bytes.
        spare.fill(MaybeUninit::uninit());
        spare[0].write(b'i');
        spare[1].write(b'j');
        assert_eq!(buffer.buffer(), b"ab");
        assert_eq!(&buffer.buf[buffer.consumed + buffer.buffered..], b"cdefgh");
        buffer.extend_from_slice(b"cdefghij");
        assert_eq!(buffer.buffer(), b"abcdefghij");

        assert_eq!(buffer.write_buf(&mut &b"k"[..]).unwrap(), 1);
        assert_eq!(buffer.buffer(), b"abcdefghijk");
        assert_eq!(buffer.buf.len(), 11);
        assert_eq!(buffer.spare_len(), 0);
    }

    #[test]
    fn reclamation_handles_poisoned_prefixes_with_empty_and_consumed_input() {
        for split in 1..=8 {
            for consumed in 0..=8 - split {
                let mut buffer = Buffer::with_capacity(8);
                buffer.extend_from_slice(b"abcdefgh");
                let mut prefix = buffer.split_to(split);
                prefix.clear();
                prefix.spare_capacity_mut().fill(MaybeUninit::uninit());
                drop(prefix);
                buffer.consume(consumed);

                // Reclamation may fail when moving live bytes would cost too
                // much.
                // Either result must keep buffered input and initialized spare
                // valid.
                buffer.try_reclaim(split);
                assert_eq!(buffer.buffer(), &b"abcdefgh"[split + consumed..]);
                // Read every spare byte so Miri checks initialization,
                // including
                // cases where there is no buffered input left to inspect.
                std::hint::black_box(
                    buffer.buf[buffer.consumed + buffer.buffered..]
                        .iter()
                        .map(|&b| u64::from(b))
                        .sum::<u64>(),
                );
                buffer.extend_from_slice(b"!");
                let mut expected = b"abcdefgh"[split + consumed..].to_vec();
                expected.push(b'!');
                assert_eq!(buffer.buffer(), expected);
            }
        }
    }

    #[test]
    fn reserve_overflow_preserves_data_after_discarding_initialized_tail() {
        for shared in [false, true] {
            let mut buffer = Buffer::with_capacity(8);
            buffer.extend_from_slice(b"abcdef");
            let spare = buffer.spare();
            buffer.try_init(0, spare);
            buffer.buf[buffer.consumed + buffer.buffered..].copy_from_slice(b"XY");
            let owned = shared.then(|| buffer.split_to(1));
            buffer.consume(if shared { 1 } else { 2 });

            let address = buffer.buffer().as_ptr();
            assert!(catch_unwind(AssertUnwindSafe(|| buffer.reserve(usize::MAX))).is_err());
            assert_eq!(buffer.consumed(), 0);
            assert_eq!((buffer.consumed() + buffer.buffered()), 4);
            assert_eq!(buffer.buffer().as_ptr(), address);
            assert_eq!(buffer.buffer(), b"cdef");
            assert_eq!(buffer.buf.len(), 4);
            assert_eq!(buffer.spare_len(), 0);
            assert!(buffer.buffer_size >= buffer.capacity());
            buffer.extend_from_slice(b"gh");
            assert_eq!(buffer.buffer(), b"cdefgh");
            if let Some(owned) = owned {
                assert_eq!(&owned[..], b"a");
            }
        }
    }

    #[test]
    fn small_consumption_does_not_repeatedly_shift_the_remaining_data() {
        for initialized in [false, true] {
            let mut buffer = Buffer::with_capacity(128);
            if initialized {
                buffer.try_init(0, 128);
            }
            buffer.extend_from_slice(&[0x55; 128]);
            buffer.consume(1);
            buffer.reserve(1);
            buffer.extend_from_slice(&[0x55]);
            let capacity = buffer.capacity();
            let address = buffer.buffer().as_ptr();
            assert!(buffer.spare() >= 64);

            for consumed in 1..=64 {
                buffer.consume(1);
                buffer.reserve(1);
                buffer.extend_from_slice(&[0x55]);
                assert_eq!(buffer.buffer().as_ptr(), address.wrapping_add(consumed));
                assert_eq!(buffer.capacity(), capacity);
                assert_eq!(buffer.buffer(), &[0x55; 128]);
            }
        }
    }

    #[test]
    fn try_reclaim_defers_small_prefixes_without_losing_data() {
        for retain in [false, true] {
            let mut buffer = Buffer::with_capacity(16);
            buffer.extend_from_slice(b"0123456789abcdef");
            let held = retain.then(|| buffer.split_to(4));
            buffer.consume(1);
            let address = buffer.buffer().as_ptr();
            let expected = buffer.buffer().to_vec();
            assert!(!buffer.try_reclaim(1));
            assert_eq!(buffer.buffer(), expected);
            assert_eq!(buffer.buffer().as_ptr(), address);
            assert_eq!(buffer.spare(), 0);

            // Consuming more data makes the next reclamation worthwhile.
            buffer.consume(buffer.len() / 2 + 1);
            let expected = buffer.buffer().to_vec();
            assert!(buffer.try_reclaim(1));
            assert_eq!(buffer.buffer(), expected);
            if let Some(held) = held {
                assert_eq!(&held[..], b"0123");
            }
        }
    }

    #[test]
    fn reserve_and_try_reclaim_keep_existing_space_unchanged() {
        for initialized in [false, true] {
            for consumed in [0, 2, 6] {
                let mut buffer = Buffer::with_capacity(16);
                buffer.extend_from_slice(b"abcdef");
                if initialized {
                    let spare = buffer.spare();
                    buffer.try_init(0xa5, spare);
                }
                buffer.consume(consumed);
                let address = buffer.buffer().as_ptr();
                let state = (
                    buffer.consumed,
                    (buffer.consumed + buffer.buffered),
                    buffer.buf.len(),
                    buffer.capacity(),
                );
                let contents = buffer.buf.to_vec();
                for additional in [0, 1, buffer.spare()] {
                    for reclaim in [false, true] {
                        if reclaim {
                            assert!(buffer.try_reclaim(additional));
                        } else {
                            buffer.reserve(additional);
                        }
                        assert_eq!(buffer.buffer().as_ptr(), address);
                        assert_eq!(
                            (
                                buffer.consumed,
                                (buffer.consumed + buffer.buffered),
                                buffer.buf.len(),
                                buffer.capacity()
                            ),
                            state
                        );
                        assert_eq!(buffer.buf.as_ref(), contents);
                    }
                }
            }
        }
    }

    #[test]
    fn try_reclaim_recovers_released_splits_without_exposing_their_uninitialized_bytes() {
        let mut buffer = Buffer::with_capacity(16);
        buffer.extend_from_slice(b"0123456789abcdef");
        let address = buffer.buffer().as_ptr();
        let mut prefix = buffer.split_to(8);
        prefix.clear();
        prefix.spare_capacity_mut().fill(MaybeUninit::uninit());
        drop(prefix);
        buffer.consume(1);

        assert!(buffer.try_reclaim(9));
        assert_eq!(buffer.buffer(), b"9abcdef");
        assert_eq!(buffer.buffer().as_ptr(), address);
        assert_eq!(buffer.capacity(), 16);
        assert_eq!(buffer.spare(), 9);
        assert_eq!(buffer.spare_len(), 0);
        buffer.try_init(0xa5, 9);
        assert_eq!(&buffer.buf[buffer.consumed + buffer.buffered..], &[0xa5; 9]);
        buffer.extend_from_slice(b"gh");
        assert_eq!(buffer.buffer(), b"9abcdefgh");
    }

    #[test]
    fn failed_try_reclaim_preserves_data_and_discards_initialized_tail() {
        for shared in [false, true] {
            for additional in [16, isize::MAX as usize, usize::MAX] {
                let mut buffer = Buffer::with_capacity(8);
                buffer.extend_from_slice(b"abcdef");
                buffer.try_init(0xa5, 2);
                let retained = shared.then(|| buffer.split_to(1));
                buffer.consume(if shared { 1 } else { 2 });

                assert!(!buffer.try_reclaim(additional));
                assert_eq!(buffer.buffer(), b"cdef");
                assert_eq!(buffer.consumed(), 0);
                assert_eq!(buffer.buf.len(), 4);
                assert_eq!(buffer.spare_len(), 0);
                assert!(buffer.buffer_size >= buffer.capacity());
                buffer.extend_from_slice(b"gh");
                assert_eq!(buffer.buffer(), b"cdefgh");
                if let Some(retained) = retained {
                    assert_eq!(&retained[..], b"a");
                }
            }
        }
        let mut empty = Buffer::new();
        assert!(empty.try_reclaim(0));
        assert!(!empty.try_reclaim(1));
        assert_eq!(empty.capacity(), 0);
    }

    #[test]
    fn reserve_growth_preserves_data_with_released_and_retained_splits() {
        for retain in [false, true] {
            let mut buffer = Buffer::with_capacity(16);
            buffer.extend_from_slice(b"0123456789abcdef");
            let prefix = buffer.split_to(4);
            let retained = if retain {
                Some(prefix)
            } else {
                drop(prefix);
                None
            };
            buffer.consume(2);
            buffer.reserve(100);
            assert_eq!(buffer.buffer(), b"6789abcdef");
            assert!(buffer.spare() >= 100);
            assert_eq!(buffer.consumed(), 0);
            if let Some(prefix) = retained {
                assert_eq!(&prefix[..], b"0123");
            }

            buffer.extend_from_slice(&[b'!'; 100]);
            assert_eq!(&buffer.buffer()[..10], b"6789abcdef");
            assert_eq!(&buffer.buffer()[10..], &[b'!'; 100]);
            let buf: Vec<u8> = buffer.buf.freeze().into();
            assert_eq!(&buf[..10], b"6789abcdef");
            assert_eq!(&buf[10..], &[b'!'; 100]);
        }
    }

    #[test]
    fn reserve_allocates_when_compact_cannot_satisfy_the_request() {
        // With three bytes remaining, compacting can supply at most five spare
        // bytes.
        for additional in [5, 6, 32] {
            let mut buffer = Buffer::with_capacity(8);
            buffer.extend_from_slice(b"abcdef");
            let spare = buffer.spare();
            buffer.try_init(0, spare);
            buffer.buf[buffer.consumed + buffer.buffered..].copy_from_slice(b"XY");
            buffer.consume(3);
            buffer.reserve(additional);
            assert_eq!(buffer.buffer(), b"def");
            assert!(buffer.spare() >= additional);
            if additional == 5 {
                assert_eq!(buffer.consumed(), 0);
                assert_eq!((buffer.consumed() + buffer.buffered()), 3);
                assert_eq!(buffer.capacity(), 8);
                assert_eq!(&buffer.buf[buffer.consumed + buffer.buffered..], b"defXY");
                assert_eq!(buffer.spare_len(), 5);
            } else {
                assert_eq!(buffer.consumed(), 0);
                assert_eq!((buffer.consumed() + buffer.buffered()), 3);
                assert_eq!(buffer.buf.len(), 3);
                assert_eq!(buffer.spare_len(), 0);
            }
            buffer.extend_from_slice(b"gh");
            assert_eq!(buffer.buffer(), b"defgh");
        }
    }

    #[test]
    fn write_buf_commits_only_successful_reads_and_allows_retry() {
        struct Source(usize, usize);
        impl Read for Source {
            fn read(&mut self, dst: &mut [u8]) -> io::Result<usize> {
                assert!(dst.len() >= 2);
                self.1 = dst.len();
                dst.fill(0xa5);
                match self.0 {
                    0 => Ok(0),
                    1 => Ok(2),
                    2 => Err(io::ErrorKind::Interrupted.into()),
                    3 => Ok(dst.len() + 1),
                    4 => Ok(usize::MAX),
                    _ => panic!("source panicked after writing"),
                }
            }
        }
        // Existing tail, growth, and growth while an owned prefix remains
        // alive.
        for (capacity, filled, consumed, split) in [(16, 6, 4, 0), (64, 40, 0, 0), (64, 48, 0, 8)] {
            for outcome in 0..6 {
                let mut buffer = Buffer::with_capacity(capacity);
                buffer.extend_from_slice(&vec![0x55; filled]);
                let retained = buffer.split_to(split);
                buffer.consume(consumed);
                let mut expected = vec![0x55; filled - split - consumed];
                let mut source = Source(outcome, 0);
                let result = catch_unwind(AssertUnwindSafe(|| buffer.write_buf(&mut source)));
                match outcome {
                    0 => assert_eq!(result.unwrap().unwrap(), 0),
                    1 => {
                        assert_eq!(result.unwrap().unwrap(), 2);
                        expected.extend_from_slice(&[0xa5; 2]);
                    }
                    2 => assert_eq!(
                        result.unwrap().unwrap_err().kind(),
                        io::ErrorKind::Interrupted
                    ),
                    3 | 4 => {
                        let error = result.unwrap().unwrap_err();
                        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
                        let available = source.1;
                        let reported = if outcome == 3 {
                            available + 1
                        } else {
                            usize::MAX
                        };
                        assert_eq!(
                            error.to_string(),
                            format!(
                                "Read returned {reported} bytes for a buffer of {available} bytes"
                            )
                        );
                    }
                    _ => assert!(result.is_err()),
                }
                assert_eq!(buffer.buffer(), expected);
                if outcome != 5 {
                    assert_eq!(buffer.spare_len(), 0);
                }
                assert_eq!(buffer.write_buf(&mut &b"!"[..]).unwrap(), 1);
                expected.push(b'!');
                assert_eq!(buffer.buffer(), expected);
                assert_eq!(buffer.write_buf(&mut io::empty()).unwrap(), 0);
                assert_eq!(buffer.buffer(), expected);
                assert_eq!(&retained[..], vec![0x55; split]);
            }
        }
    }

    #[test]
    fn write_buf_reuses_prepared_space_without_initializing_it() {
        let mut buffer = Buffer::with_capacity(8);
        buffer.extend_from_slice(b"ab");
        buffer.try_init(b'x', buffer.spare());
        // Keep known bytes in spare memory while dropping initialized-tail
        // tracking, as write_buf does after a short read.
        buffer.buf.truncate(buffer.len());
        struct Inspect;
        impl Read for Inspect {
            fn read(&mut self, dst: &mut [u8]) -> io::Result<usize> {
                assert_eq!(dst, b"xxxxxx");
                dst[..4].copy_from_slice(b"cdef");
                Ok(4)
            }
        }
        assert_eq!(buffer.write_buf(&mut Inspect).unwrap(), 4);
        assert_eq!(buffer.buffer(), b"abcdef");
        assert_eq!(buffer.capacity(), 8);
        assert_eq!(buffer.buf.len(), 6);
        assert_eq!(buffer.spare_len(), 0);
    }

    #[test]
    fn allocate_returns_old_storage_and_prepares_new_capacity() {
        for consumed in [0, 2, 6] {
            for additional in [0, 1, 32] {
                let mut buffer = Buffer::with_capacity(16);
                buffer.extend_from_slice(b"--abcdef");
                let retained = buffer.split_to(2);
                buffer.try_init(b'x', 4);
                buffer.consume(consumed);
                let expected = &b"abcdef"[consumed..];
                let address = buffer.buf.as_ptr();

                let mut old = buffer.allocate(additional);
                assert_eq!(old.as_ptr(), address);
                assert_eq!(&old[..], b"abcdefxxxx");
                assert_eq!(buffer.buffer(), expected);
                assert_eq!(buffer.consumed(), 0);
                assert!(buffer.spare() >= additional);
                assert_eq!(buffer.spare_len(), 0);

                old.fill(b'z');
                assert_eq!(buffer.buffer(), expected);
                assert_eq!(&retained[..], b"--");
                assert_eq!(buffer.write_buf(&mut &b"!"[..]).unwrap(), 1);
                assert_eq!(buffer.buffer(), [expected, b"!"].concat());
                drop(buffer);
                assert!(old.iter().all(|&byte| byte == b'z'));
            }
        }
    }

    #[test]
    fn allocate_rejects_overflow_without_changing_buffer() {
        let mut buffer = Buffer::with_capacity(16);
        buffer.extend_from_slice(b"abcdef");
        buffer.consume(2);
        buffer.try_init(b'x', 4);
        let address = buffer.buffer().as_ptr();

        for additional in [usize::MAX, isize::MAX as usize] {
            assert!(catch_unwind(AssertUnwindSafe(|| buffer.allocate(additional))).is_err());
            assert_eq!(buffer.buffer(), b"cdef");
            assert_eq!(buffer.buffer().as_ptr(), address);
            assert_eq!(buffer.consumed(), 2);
            assert_eq!(buffer.capacity(), 16);
            assert_eq!(&buffer.buf[buffer.consumed + buffer.buffered..], b"xxxx");
        }
    }

    #[test]
    fn try_init_only_fills_missing_space_and_rejects_excess() {
        let mut buffer = Buffer::with_capacity(16);
        buffer.extend_from_slice(b"abcdef");
        buffer.consume(2);
        let address = buffer.buffer().as_ptr();
        for (value, len, tail) in [
            (0xa5, 3, &[0xa5; 3][..]),
            (0, 1, &[0xa5; 3][..]),
            (0, 5, &[0xa5, 0xa5, 0xa5, 0, 0][..]),
        ] {
            buffer.try_init(value, len);
            assert_eq!(&buffer.buf[6..], tail);
        }
        for len in [11, usize::MAX] {
            assert!(catch_unwind(AssertUnwindSafe(|| buffer.try_init(0, len))).is_err());
            assert_eq!(&buffer.buf[6..], &[0xa5, 0xa5, 0xa5, 0, 0]);
        }
        assert_eq!(buffer.buffer(), b"cdef");
        assert_eq!(buffer.buffer().as_ptr(), address);
        assert_eq!(buffer.capacity(), 16);
        let mut empty = Buffer::new();
        empty.try_init(0, 0);
        assert_eq!(empty.capacity(), 0);
    }

    #[test]
    fn mixed_operations_match_a_byte_queue() {
        for capacity in [0, 1, 8, 32] {
            let mut buffer = Buffer::with_capacity(capacity);
            let mut expected = Vec::new();
            let mut retained = Vec::new();
            let mut seed = 1_u64;
            let mut next = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed as usize
            };

            for _ in 0..1000 {
                let amount = next() % 17;
                match next() % 11 {
                    0 => {
                        let bytes = vec![next() as u8; amount];
                        buffer.extend_from_slice(&bytes);
                        expected.extend_from_slice(&bytes);
                    }
                    1 => {
                        let count = amount.min(expected.len());
                        buffer.consume(count);
                        expected.drain(..count);
                    }
                    2 => {
                        let at = amount.min(expected.len());
                        buffer.advance(at);
                        expected.drain(..at);
                    }
                    3 => {
                        let at = amount.min(expected.len());
                        retained.push((
                            buffer.take_bytes(..at, 0),
                            expected.drain(..at).collect::<Vec<_>>(),
                        ));
                    }
                    4 => {
                        let at = amount.min(expected.len());
                        retained.push((buffer.split_off(at), expected.split_off(at)));
                    }
                    5 => {
                        let end = amount.min(expected.len());
                        buffer.extend_from_within(..end);
                        expected.extend_from_within(..end);
                    }
                    6 => {
                        let bytes = vec![next() as u8; amount];
                        let count = buffer.write_buf(&mut bytes.as_slice()).unwrap();
                        expected.extend_from_slice(&bytes[..count]);
                    }
                    7 => buffer.reserve(amount),
                    8 => buffer.compact(),
                    9 => {
                        let start = next() % (expected.len() + 1);
                        let end = start + next() % (expected.len() - start + 1);
                        let consume = amount.min(expected.len());
                        assert_eq!(buffer.take_view(start..end, consume), &expected[start..end]);
                        expected.drain(..consume);
                    }
                    _ => {
                        buffer.clear();
                        expected.clear();
                    }
                }
                assert!(buffer.consumed <= (buffer.consumed + buffer.buffered));
                assert!((buffer.consumed + buffer.buffered) <= buffer.buf.len());
                assert!(buffer.buf.len() <= buffer.capacity());
                assert_eq!(buffer.len(), expected.len());
                assert_eq!(buffer.buffer(), expected);
                assert_eq!(buffer.is_empty(), expected.is_empty());
                for (bytes, saved) in &retained {
                    assert_eq!(bytes.as_ref(), saved);
                }
            }
            drop(buffer);
            for (bytes, saved) in retained {
                assert_eq!(bytes.as_ref(), saved);
            }
        }
    }

    #[test]
    fn taking_a_view_validates_selection_and_consumption_before_advancing() {
        let data = b"abcdef";
        for start in 0..=data.len() + 1 {
            for end in 0..=data.len() + 1 {
                for consume in (0..=data.len() + 1).chain([usize::MAX]) {
                    let mut buffer = Buffer::with_capacity(32);
                    buffer.extend_from_slice(b"old:");
                    buffer.extend_from_slice(data);
                    buffer.advance(4);
                    let original = buffer.buffer().as_ptr();
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        let view = buffer.take_view(start..end, consume);
                        assert_eq!(view, &data[start..end]);
                        assert_eq!(view.as_ptr(), original.wrapping_add(start));
                    }));
                    let valid = start <= end && end <= data.len() && consume <= data.len();
                    assert_eq!(result.is_ok(), valid, "{start}..{end}, consume {consume}");
                    if valid {
                        assert_eq!(buffer.buffer(), &data[consume..]);
                    } else {
                        assert_eq!(buffer.buffer(), data);
                    }
                }
            }
        }
    }

    #[test]
    fn extend_from_within_matches_vec_for_remaining_data() {
        for capacity in [0, 4, 8, 16] {
            for initialized in [false, true] {
                for consumed in 0..=4 {
                    for start in 0..=4 - consumed {
                        for end in start..=4 - consumed {
                            let mut buffer = Buffer::with_capacity(capacity);
                            let input = b"a\0\xffz";
                            buffer.extend_from_slice(input);
                            if initialized {
                                buffer.try_init(0, buffer.spare());
                            }
                            buffer.consume(consumed);
                            let mut expected = input[consumed..].to_vec();

                            buffer.extend_from_within(start..end);
                            expected.extend_from_within(start..end);
                            assert_eq!(buffer.buffer(), expected);
                            assert_eq!(buffer.len(), expected.len());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn extend_from_within_accepts_range_bounds_and_empty_ranges() {
        let mut buffer = Buffer::with_capacity(0);
        buffer.extend_from_within(..);
        assert!(buffer.is_empty());
        assert_eq!(buffer.capacity(), 0);

        buffer.extend_from_slice(b"abc");
        buffer.consume(1);
        buffer.extend_from_within(..=0);
        assert_eq!(buffer.buffer(), b"bcb");
        buffer.extend_from_within(1..=2);
        assert_eq!(buffer.buffer(), b"bcbcb");
        buffer.extend_from_within(4..);
        assert_eq!(buffer.buffer(), b"bcbcbb");
        buffer.extend_from_within(..1);
        assert_eq!(buffer.buffer(), b"bcbcbbb");
        buffer.extend_from_within((Excluded(0), Included(1)));
        assert_eq!(buffer.buffer(), b"bcbcbbbc");

        let len = buffer.len();
        buffer.extend_from_within(len..len);
        assert_eq!(buffer.buffer(), b"bcbcbbbc");
        buffer.clear();
        buffer.extend_from_within(..);
        assert!(buffer.is_empty());
    }

    #[test]
    fn extend_from_within_rejects_invalid_ranges_before_mutating() {
        let mut buffer = Buffer::with_capacity(16);
        buffer.try_init(0, buffer.spare());
        buffer.write_buf(&mut &b"abcdef"[..]).unwrap();
        buffer.consume(2);
        let address = buffer.buffer().as_ptr();
        let buffered_end = buffer.consumed() + buffer.buffered();
        let capacity = buffer.capacity();

        for range in [
            (Included(3), Excluded(2)),
            (Included(0), Excluded(5)),
            (Included(5), Unbounded),
            (Included(5), Excluded(5)),
            (Unbounded, Included(usize::MAX)),
            (Excluded(usize::MAX), Unbounded),
        ] {
            assert!(catch_unwind(AssertUnwindSafe(|| buffer.extend_from_within(range))).is_err());
            assert_eq!(buffer.buffer(), b"cdef");
            assert_eq!(buffer.buffer().as_ptr(), address);
            assert_eq!((buffer.consumed() + buffer.buffered()), buffered_end);
            assert_eq!(buffer.capacity(), capacity);
        }
        buffer.extend_from_within(..);
        assert_eq!(buffer.buffer(), b"cdefcdef");
    }

    #[test]
    fn extensions_reuse_space_and_preserve_shared_storage() {
        for capacity in [8, 32] {
            for initialized in [0, 1, 2] {
                for split in 0..3 {
                    for within in [false, true] {
                        let mut buffer = Buffer::with_capacity(capacity);
                        buffer.extend_from_slice(b"abcdefgh");
                        buffer.consume(2);
                        let retained = match split {
                            1 => buffer.split_to(2),
                            2 => buffer.split_off(2),
                            _ => BytesMut::new(),
                        };
                        let saved = retained.to_vec();
                        let mut expected = buffer.buffer().to_vec();
                        let init = match initialized {
                            0 => 0,
                            1 => buffer.spare().min(2),
                            _ => buffer.spare(),
                        };
                        buffer.try_init(0xa5, init);
                        for _ in 0..4 {
                            let address = buffer.buffer().as_ptr();
                            let fits = expected.len() <= buffer.spare();
                            if within {
                                buffer.extend_from_within(..);
                            } else {
                                buffer.extend_from_slice(&expected);
                            }
                            expected.extend_from_within(..);
                            assert_eq!(buffer.buffer(), expected);
                            if fits {
                                assert_eq!(buffer.buffer().as_ptr(), address);
                            }
                        }
                        // Appending from another view of the same allocation is
                        // valid.
                        buffer.extend_from_slice(&retained);
                        expected.extend_from_slice(&saved);
                        assert_eq!(buffer.buffer(), expected);
                        drop(buffer);
                        assert_eq!(&retained[..], saved);
                    }
                }
            }
        }
    }

    #[test]
    fn fill_buf_reads_only_when_empty_and_can_resume_after_eof() {
        let mut source = io::Cursor::new(b"abcdef");
        let mut buffer = Buffer::with_capacity(4);
        assert_eq!(buffer.fill_buf(&mut source).unwrap(), b"abcd");
        buffer.consume(2);
        assert_eq!(buffer.fill_buf(&mut source).unwrap(), b"cd");
        assert_eq!(source.position(), 4);
        buffer.consume(2);
        assert_eq!(buffer.fill_buf(&mut source).unwrap(), b"ef");
        buffer.consume(2);
        for _ in 0..2 {
            assert!(buffer.fill_buf(&mut io::empty()).unwrap().is_empty());
        }
        assert_eq!(buffer.fill_buf(&mut &b"\xff\0"[..]).unwrap(), b"\xff\0");
    }

    #[test]
    fn fill_buf_errors_do_not_expose_consumed_or_failed_input() {
        struct Reject;

        impl Read for Reject {
            fn read(&mut self, dst: &mut [u8]) -> io::Result<usize> {
                assert!(!dst.is_empty());
                dst.fill(0xa5);
                Err(io::ErrorKind::Interrupted.into())
            }
        }

        let mut buffer = Buffer::with_capacity(4);
        assert_eq!(buffer.fill_buf(&mut &b"old"[..]).unwrap(), b"old");
        // Buffered input is returned without consulting even a failing source.
        assert_eq!(buffer.fill_buf(&mut Reject).unwrap(), b"old");
        buffer.consume(3);
        assert_eq!(
            buffer.fill_buf(&mut Reject).unwrap_err().kind(),
            io::ErrorKind::Interrupted
        );
        assert!(buffer.buffer().is_empty());
        let mut source = &b"ok"[..];
        // Empty buffers reuse consumed space after the failed read.
        assert_eq!(buffer.fill_buf(&mut source).unwrap(), b"ok");
        assert!(source.is_empty());
    }

    #[test]
    fn fill_buf_allocates_on_demand_and_preserves_split_items() {
        let input: Vec<u8> = (0..64).collect();
        for capacity in [0, 4] {
            let mut buffer = Buffer::with_capacity(capacity);
            let mut source = &input[..];
            let first_len = buffer.fill_buf(&mut source).unwrap().len();
            assert!(first_len > 0);
            let owned = buffer.take_bytes(..first_len, 0);
            let mut output = owned.to_vec();
            loop {
                let chunk = buffer.fill_buf(&mut source).unwrap();
                if chunk.is_empty() {
                    break;
                }
                output.extend_from_slice(chunk);
                let consumed = chunk.len();
                buffer.consume(consumed);
            }
            assert_eq!(output, input);
            drop(buffer);
            assert_eq!(&owned[..], &input[..first_len]);
        }
    }

    #[test]
    fn buf_cursor_operations_agree_with_buffered_bytes() {
        fn header<B: Buf>(input: &mut B) -> (u16, u16) {
            (input.get_u16(), input.get_u16())
        }

        let mut buffer = Buffer::with_capacity(16);
        buffer.try_init(0, buffer.spare());
        buffer.write_buf(&mut &b"\x01\x02\x03\x04tail"[..]).unwrap();
        assert_eq!(header(&mut buffer), (0x0102, 0x0304));
        assert_eq!(buffer.chunk(), b"tail");
        assert_eq!(buffer.remaining(), 4);

        let mut tail = [0; 4];
        buffer.copy_to_slice(&mut tail);
        assert_eq!(&tail, b"tail");
        assert!(!buffer.has_remaining());
        assert!(buffer.chunk().is_empty());
        buffer.advance(0);
        assert!(buffer.chunk().is_empty());
    }

    #[test]
    fn advance_and_consume_reject_excess_without_changing_input() {
        let mut buffer = Buffer::with_capacity(8);
        buffer.try_init(0, buffer.spare());
        buffer.write_buf(&mut &b"abcdef"[..]).unwrap();
        buffer.consume(2);
        for count in [5, usize::MAX] {
            let consumed = buffer.consumed();
            assert!(catch_unwind(AssertUnwindSafe(|| buffer.advance(count))).is_err());
            assert!(catch_unwind(AssertUnwindSafe(|| buffer.consume(count))).is_err());
            assert_eq!(buffer.consumed(), consumed);
            assert_eq!(buffer.chunk(), b"cdef");
            assert_eq!(buffer.remaining(), 4);
        }
        buffer.consume(0);
        assert_eq!(buffer.chunk(), b"cdef");
        buffer.consume(4);
        assert!(buffer.is_empty());
        buffer.consume(0);
        assert!(buffer.is_empty());
        assert!(catch_unwind(AssertUnwindSafe(|| buffer.consume(1))).is_err());
        assert_eq!(buffer.consumed(), 6);

        buffer.compact();
        buffer.try_init(0, buffer.spare());
        assert_eq!(buffer.write_buf(&mut &b"next"[..]).unwrap(), 4);
        assert_eq!(buffer.chunk(), b"next");
    }

    #[test]
    fn reserve_preserves_pending_input_and_makes_room_for_the_full_request() {
        for capacity in [1, 8, 16] {
            let input: Vec<_> = (0..capacity).map(|i| i as u8).collect();
            for end in 0..=input.len() {
                for start in 0..=end {
                    for additional in [0, 1, capacity / 2, capacity, capacity * 3] {
                        let mut buffer = Buffer::with_capacity(capacity);
                        buffer.try_init(0, buffer.spare());
                        buffer.write_buf(&mut &input[..end]).unwrap();
                        buffer.advance(start);

                        buffer.reserve(additional);

                        assert_eq!(buffer.buffer(), &input[start..end]);
                        assert!(
                            buffer.spare() >= additional,
                            "capacity={capacity}, end={end}, start={start}, additional={additional}"
                        );
                        if additional != 0 {
                            let more = vec![0xff; additional];
                            buffer.try_init(0, additional);
                            assert_eq!(buffer.write_buf(&mut &more[..]).unwrap(), additional);
                            let expected = [&input[start..end], &more[..]].concat();
                            assert_eq!(buffer.buffer(), expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn split_off_uses_pending_input_offsets_and_excludes_the_initialized_tail() {
        let input = b"abcdef";
        for start in 0..=input.len() {
            for at in 0..=input.len() - start {
                let mut buffer = Buffer::with_capacity(16);
                buffer.try_init(0, buffer.spare());
                buffer.write_buf(&mut &input[..]).unwrap();
                buffer.advance(start);
                let address = buffer.buffer().as_ptr() as usize;
                let suffix = buffer.split_off(at);
                assert_eq!(buffer.buffer(), &input[start..start + at]);
                assert_eq!(buffer.len(), at);
                assert_eq!(&suffix[..], &input[start + at..]);
                if !suffix.is_empty() {
                    assert_eq!(suffix.as_ptr() as usize, address + at);
                }
            }
        }
        let mut empty = Buffer::with_capacity(1);
        assert!(empty.split_off(0).is_empty());
        assert!(empty.is_empty());
        empty.reserve(1);
        empty.try_init(0, 1);
        assert_eq!(empty.write_buf(&mut &b"x"[..]).unwrap(), 1);
        assert_eq!(empty.buffer(), b"x");
    }

    #[test]
    fn split_off_rejects_indices_past_input_before_changing_the_window() {
        let mut buffer = Buffer::with_capacity(16);
        buffer.try_init(0, buffer.spare());
        buffer.write_buf(&mut &b"abcdef"[..]).unwrap();
        buffer.advance(2);
        for at in [5, usize::MAX] {
            assert!(catch_unwind(AssertUnwindSafe(|| buffer.split_off(at))).is_err());
            assert_eq!(buffer.buffer(), b"cdef");
        }
        assert_eq!(&buffer.split_off(2)[..], b"ef");
        assert_eq!(buffer.buffer(), b"cd");
    }

    #[test]
    fn reserve_reclaims_released_storage_without_allocating() {
        let mut buffer = Buffer::with_capacity(16);
        buffer.extend_from_slice(b"0123456789abcdef");
        let address = buffer.buffer().as_ptr();
        drop(buffer.split_to(12));
        buffer.consume(1);

        let (_, count) = allocations(|| {
            buffer.reserve(13);
        });
        assert_eq!(count, 0);
        assert_eq!(buffer.buffer(), b"def");
        assert_eq!(buffer.buffer().as_ptr(), address);
        assert_eq!(buffer.capacity(), 16);
        assert_eq!(buffer.spare(), 13);
    }

    #[test]
    fn reclaim_does_not_allocate_on_success_or_failure() {
        for retain in [false, true] {
            for split in [0, 4, 8] {
                for consumed in [0, 2, 8] {
                    for additional in [0, 1, 8, usize::MAX] {
                        let mut buffer = Buffer::with_capacity(16);
                        buffer.extend_from_slice(b"0123456789abcdef");
                        let prefix = (split != 0).then(|| buffer.split_to(split));
                        let retained = if retain {
                            prefix
                        } else {
                            drop(prefix);
                            None
                        };
                        buffer.consume(consumed);
                        let expected = buffer.buffer().to_vec();

                        let (reclaimed, count) = allocations(|| buffer.try_reclaim(additional));
                        assert_eq!(count, 0);
                        assert_eq!(buffer.buffer(), expected);
                        if reclaimed {
                            assert!(buffer.spare() >= additional);
                        }
                        if let Some(retained) = retained {
                            assert_eq!(&retained[..], &b"0123456789abcdef"[..split]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn take_view_accepts_slice_ranges_without_copying_or_coupling_consumption() {
        fn check<I: std::ops::RangeBounds<usize> + Clone>(range: I, start: usize, end: usize) {
            for consume in [0, 2, 6] {
                let mut buffer = Buffer::with_capacity(32);
                buffer.extend_from_slice(b"old:abcdef");
                buffer.consume(4);
                let address = buffer.buffer().as_ptr();
                let view = buffer.take_view(range.clone(), consume);
                assert_eq!(view, &b"abcdef"[start..end]);
                assert_eq!(view.as_ptr(), address.wrapping_add(start));
                assert_eq!(buffer.buffer(), &b"abcdef"[consume..]);
            }
        }
        check(.., 0, 6);
        check(2.., 2, 6);
        check(..4, 0, 4);
        check(..=3, 0, 4);
        check(1..=3, 1, 4);
        check(1..4, 1, 4);
        check(6.., 6, 6);
        check(..0, 0, 0);
        check(
            (std::ops::Bound::Excluded(0), std::ops::Bound::Included(3)),
            1,
            4,
        );
    }

    #[test]
    fn take_view_rejects_invalid_slice_ranges_without_consuming_input() {
        fn check(range: impl std::ops::RangeBounds<usize>, consume: usize) {
            let mut buffer = Buffer::with_capacity(32);
            let spare = buffer.spare();
            buffer.try_init(0, spare);
            buffer.write_buf(&mut &b"old:abcdef"[..]).unwrap();
            buffer.consume(4);
            let address = buffer.buffer().as_ptr();
            let consumed = buffer.consumed();
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    buffer.take_view(range, consume);
                }))
                .is_err()
            );
            assert_eq!(buffer.consumed(), consumed);
            assert_eq!(buffer.buffer(), b"abcdef");
            assert_eq!(buffer.buffer().as_ptr(), address);
        }
        check(..=usize::MAX, 1);
        check(..=6, 1);
        check((Excluded(usize::MAX), Unbounded), 1);
    }

    struct Tail(usize);

    impl std::ops::Index<Tail> for [u8] {
        type Output = [u8];

        fn index(&self, index: Tail) -> &Self::Output {
            &self[index.0..]
        }
    }

    impl std::ops::RangeBounds<usize> for Tail {
        fn start_bound(&self) -> std::ops::Bound<&usize> {
            std::ops::Bound::Included(&self.0)
        }

        fn end_bound(&self) -> std::ops::Bound<&usize> {
            std::ops::Bound::Unbounded
        }
    }

    #[test]
    fn custom_indices_and_bounds_work_for_views_and_appending_across_reservation() {
        for capacity in [0, 8, 32] {
            let mut buffer = Buffer::with_capacity(capacity);
            buffer.extend_from_slice(b"old:abcdef");
            buffer.consume(4);
            let address = buffer.buffer().as_ptr();
            let view = buffer.take_view(Tail(2), 1);
            assert_eq!(view, b"cdef");
            assert_eq!(view.as_ptr(), address.wrapping_add(2));
            assert_eq!(buffer.buffer(), b"bcdef");
            buffer.extend_from_within(Tail(2));
            assert_eq!(buffer.buffer(), b"bcdefdef");
            buffer.extend_from_within(Tail(buffer.len()));
            assert_eq!(buffer.buffer(), b"bcdefdef");
        }
    }

    #[test]
    fn reclaimed_adopted_storage_can_be_filled_and_read_again() {
        for capacity in [256, 1024] {
            let mut source = BytesMut::with_capacity(capacity);
            source.resize(capacity, b'x');
            source.advance(capacity - 128);
            let mut buffer = Buffer::new();
            buffer.put(source);
            assert_eq!(buffer.buffer_size, 128);
            assert!(buffer.try_reclaim(1));
            assert_eq!(buffer.capacity(), capacity);
            assert_eq!(buffer.buffer_size, capacity);
            buffer.extend_from_slice(&vec![b'y'; buffer.spare()]);
            assert_eq!(buffer.buffered, capacity);
            assert_eq!(buffer.spare(), 0);

            let expected = buffer.buffer().to_vec();
            let mut input = &b"!"[..];
            assert_eq!(buffer.write_buf(&mut input).unwrap(), 1);
            assert!(input.is_empty());
            assert_eq!(&buffer.buffer()[..expected.len()], expected);
            assert_eq!(&buffer.buffer()[expected.len()..], b"!");
            assert_eq!(buffer.buf.len(), capacity + 1);
            assert_eq!(buffer.buffer_size, buffer.capacity());
        }
    }

    #[test]
    fn clearing_or_consuming_all_input_reuses_storage() {
        for clear in [false, true] {
            let mut buffer = Buffer::with_capacity(64);
            buffer.extend_from_slice(b"old");
            buffer.try_init(0xa5, buffer.spare());
            let address = buffer.buf.as_ptr();
            if clear {
                buffer.clear();
            } else {
                buffer.consume(3);
            }
            assert!(buffer.is_empty());
            assert_eq!(&buffer.buf[..3], b"old");
            assert!(buffer.buf[3..].iter().all(|&byte| byte == 0xa5));
            assert_eq!(buffer.write_buf(&mut &b"new"[..]).unwrap(), 3);
            assert_eq!(buffer.buffer(), b"new");
            assert_eq!(buffer.buf.as_ptr(), address);
            assert_eq!(buffer.capacity(), 64);
        }
    }

    #[test]
    fn make_room_compacts_consumed_space_or_grows_above_half_full() {
        for initialized in [false, true] {
            for (filled, consumed, grows) in [
                (0, 0, false),
                (1, 0, false),
                (31, 0, false),
                (32, 0, false),
                (33, 0, true),
                (48, 0, true),
                (63, 0, true),
                (64, 0, true),
                (40, 32, false),
                (63, 40, false),
                (64, 32, false),
                (64, 64, false),
            ] {
                let mut buffer = Buffer::with_capacity(64);
                buffer.extend_from_slice(&vec![0x5a; filled]);
                buffer.consume(consumed);
                if initialized {
                    buffer.try_init(0xa5, buffer.spare());
                }
                let address = buffer.buf.as_ptr();
                buffer.make_room();
                if !grows {
                    assert_eq!(buffer.capacity(), 64);
                    assert_eq!(buffer.buf.as_ptr(), address);
                } else {
                    assert!(buffer.capacity() >= 128);
                }
                assert_eq!(buffer.buffer_size, buffer.capacity());
                assert_eq!(buffer.buffer(), vec![0x5a; filled - consumed]);
                assert!(buffer.spare() >= 32);
            }
        }
    }

    #[test]
    fn exhausted_views_use_the_recorded_growth_size() {
        let mut buffer = Buffer::with_capacity(64);
        buffer.extend_from_slice(&[0x55; 64]);
        buffer.make_room();
        assert!(buffer.capacity() >= 128);
        let grown = buffer.buffer_size;
        assert_eq!(grown, buffer.capacity());
        let rest = buffer.spare();
        buffer.extend_from_slice(&vec![0x66; rest]);
        let retained = buffer.split_to(buffer.len());
        assert_eq!(buffer.capacity(), 0);
        buffer.make_room();
        assert!(buffer.capacity() >= grown.div_ceil(2));
        assert_eq!(buffer.buffer_size, grown);
        assert_eq!(&retained[..64], &[0x55; 64]);
        assert_eq!(&retained[64..], vec![0x66; rest]);
    }
}
