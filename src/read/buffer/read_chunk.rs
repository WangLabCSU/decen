#[cfg(feature = "nightly")]
use std::io::BorrowedBuf;
use std::io::{self, Read};
use std::ops::Range;
#[cfg(not(feature = "nightly"))]
use std::ptr;
use std::slice;
use std::slice::SliceIndex;

use bytes::{Buf, BytesMut};

// buf: [ consumed | buffered | initialized spare | spare capacity ]
// consumed + buffered <= buf.len() <= buf.capacity() <= buffer_size.
// buf.len() tracks initialization separately from committed input. Failed
// reads never add input, even when they initialize or write their destination.
#[derive(Debug)]
pub(super) struct ReadChunk {
    buf: BytesMut,
    consumed: usize,
    buffered: usize,
    // Keep the refill size after owned extraction shrinks the BytesMut view.
    buffer_size: usize,
}

impl ReadChunk {
    #[inline]
    pub(super) fn with_capacity(capacity: usize) -> Self {
        let buf = BytesMut::with_capacity(capacity);
        Self {
            buffer_size: buf.capacity(),
            buf,
            consumed: 0,
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
    pub(super) fn buffer(&self) -> &[u8] {
        // SAFETY: consumed + buffered <= buf.len() bounds this slice by
        // initialized input. The borrow prevents changes to its storage.
        unsafe { slice::from_raw_parts(self.buf.as_ptr().add(self.consumed), self.buffered) }
    }

    // Discard input without changing the known initialization of storage.
    #[inline]
    pub(super) fn clear(&mut self) {
        self.consumed = 0;
        self.buffered = 0;
    }

    #[inline]
    pub(super) fn clear_buf(&mut self) {
        self.buf.clear();
    }

    /// Consumes `amount` bytes without checking the input length.
    ///
    /// # Safety
    ///
    /// Requires `amount <= self.len()`.
    #[inline]
    pub(super) unsafe fn consume_unchecked(&mut self, amount: usize) {
        self.consumed += amount;
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
        // SAFETY: buffer() excludes consumed bytes and spare storage; the
        // caller guarantees that range is valid within that input slice.
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
        let start = self.consumed + range.start;
        let end = self.consumed + range.end;
        // SAFETY: the caller bounds consumption by the current input length.
        // Only counters change, so the saved selection stays in place.
        unsafe { self.consume_unchecked(consume) };
        // SAFETY: the saved range lies within initialized storage, including
        // any selected bytes that were just consumed.
        unsafe { self.buf.get_unchecked(start..end) }
    }

    /// Transfers `range`, consuming through its end plus `addition` bytes.
    /// May allocate, but does not copy bytes.
    ///
    /// # Safety
    ///
    /// Requires `range.start <= range.end <= self.len()` and
    /// `addition <= self.len() - range.end`.
    #[inline]
    pub(super) unsafe fn take_split_bytes_unchecked(
        &mut self,
        range: Range<usize>,
        addition: usize,
    ) -> BytesMut {
        let skip = self.consumed + range.start;
        let mut bytes = self.buf.split_to(self.consumed + range.end);
        self.buffered -= range.end + addition;
        self.consumed = addition;
        bytes.advance(skip);
        bytes
    }

    #[inline]
    pub(super) fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    #[inline]
    pub(super) fn buffer_size(&self) -> usize {
        self.buffer_size.max(1)
    }

    // Preserve known initialization while trying to restore the read size.
    // Recovered storage from released Items cannot be assumed initialized.
    #[inline]
    pub(super) fn try_reclaim(&mut self) -> bool {
        let reclaimed = self.buf.try_reclaim(self.buffer_size() - self.buf.len());
        self.buffer_size = self.buffer_size.max(self.buf.capacity());
        reclaimed
    }

    // Discard input and its initialization bound, then reserve a read chunk.
    // The caller retains pending input elsewhere before resetting this chunk.
    #[inline]
    pub(super) fn reserve(&mut self, capacity: usize) {
        self.buf.reserve(capacity);
        // BytesMut may provide more than requested; retain that read size.
        self.buffer_size = self.buffer_size.max(self.buf.capacity());
    }

    #[inline]
    pub(super) fn read_from<R: ?Sized + Read>(&mut self, reader: &mut R) -> io::Result<usize> {
        debug_assert_eq!(self.consumed, 0);
        debug_assert_eq!(self.buffered, 0);
        #[cfg(feature = "nightly")]
        {
            self.write_uninit_buf(reader)
        }
        #[cfg(not(feature = "nightly"))]
        {
            self.write_init_buf(reader)
        }
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
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::super::Buffer;

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
                buffer.chunk.buf.extend_from_slice(b"ab");
                buffer.chunk.buffered = 2;
                if initialized {
                    buffer.chunk.buf.resize(8, b'x');
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
        buffer.chunk.buf.extend_from_slice(b"abcdefgh");
        buffer.chunk.buffered = 8;
        buffer.consume(8);
        buffer.write_buf(&mut Inspect).unwrap();
        assert_eq!(buffer.buffer().to_vec(), b"xy");
        // Clear and poison an owned part before returning its storage. Reuse
        // must not expose it as input or trust its previous initialization.
        let mut owned = buffer.take_split_bytes(..2, 0);
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
            buffer.chunk.buf.extend_from_slice(b"ab");
            buffer.chunk.buffered = 2;
            assert_eq!(buffer.write_buf(&mut Source(success)).is_ok(), success);
            let expected: &[u8] = if success { b"abcd" } else { b"ab" };
            assert_eq!(buffer.buffer().to_vec(), expected);
            assert_eq!(buffer.chunk.buf.len(), 2);
            buffer.write_buf(&mut &b"!"[..]).unwrap();
            assert_eq!(buffer.buffer().to_vec(), [expected, b"!"].concat());
        }
    }
}
