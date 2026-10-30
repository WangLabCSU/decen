use std::ops::{Bound, Index, Range, RangeBounds};

use bytes::BytesMut;

/// Consecutive bytes stored in one or two parts.
///
/// In [`Split`](Self::Split), the first part precedes the second. Offsets
/// and lengths refer to their concatenation. Inspecting or borrowing the
/// parts does not copy bytes. A frame can cross the boundary between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Segments<T> {
    /// A contiguous sequence of bytes.
    Contiguous(T),
    /// Two consecutive parts of the same sequence.
    Split(T, T),
}

impl<T: AsRef<[u8]>> Segments<T> {
    /// Returns both parts as slices. The first is empty for `Contiguous`.
    #[inline]
    pub fn parts(&self) -> (&[u8], &[u8]) {
        match self {
            Self::Contiguous(data) => (&[], data.as_ref()),
            Self::Split(first, second) => (first.as_ref(), second.as_ref()),
        }
    }

    /// Returns the total number of bytes.
    #[inline]
    pub fn len(&self) -> usize {
        let (first, second) = self.parts();
        first.len() + second.len()
    }

    /// Returns `true` if both parts are empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrows the bytes without copying them.
    #[inline]
    pub fn as_borrowed(&self) -> Segments<&[u8]> {
        let (first, second) = self.parts();
        Segments::from_parts(first, second)
    }

    /// Returns an iterator over the bytes in order.
    #[inline]
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &u8> + Clone {
        let (first, second) = self.parts();
        first.iter().chain(second)
    }

    /// Allocates a vector and copies both parts into it in order.
    #[inline]
    pub fn to_vec(&self) -> Vec<u8> {
        match self {
            Self::Contiguous(data) => data.as_ref().to_vec(),
            Self::Split(first, second) => Self::copy_parts(first.as_ref(), second.as_ref()),
        }
    }

    // Keep the ordinary one-slice copy small enough to inline into decoders.
    #[inline(never)]
    fn copy_parts(first: &[u8], second: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(first.len() + second.len());
        bytes.extend_from_slice(first);
        bytes.extend_from_slice(second);
        bytes
    }
}

impl<'a> Segments<&'a [u8]> {
    #[inline]
    pub(super) fn from_parts(first: &'a [u8], second: &'a [u8]) -> Self {
        if first.is_empty() {
            Self::Contiguous(second)
        } else if second.is_empty() {
            Self::Contiguous(first)
        } else {
            Self::Split(first, second)
        }
    }

    /// Borrows `range`, which may cross the boundary between the parts.
    ///
    /// Does not allocate or copy bytes.
    ///
    /// # Panics
    ///
    /// Panics if `range` is out of bounds or starts after it ends.
    #[inline]
    pub fn slice(self, range: impl RangeBounds<usize>) -> Self {
        let range = bounds(range, self.len());
        match self {
            Self::Contiguous(data) => Self::Contiguous(&data[range]),
            Self::Split(first, second) => {
                if range.start >= first.len() {
                    Self::Contiguous(&second[range.start - first.len()..range.end - first.len()])
                } else if range.end <= first.len() {
                    Self::Contiguous(&first[range])
                } else {
                    Self::Split(&first[range.start..], &second[..range.end - first.len()])
                }
            }
        }
    }
}

impl Segments<BytesMut> {
    /// Returns one contiguous buffer containing these bytes.
    ///
    /// Returns a `Contiguous` buffer directly. For `Split`, appends `second` to
    /// `first`, copying `second` and possibly allocating or moving `first`.
    pub fn join(self) -> BytesMut {
        match self {
            Self::Contiguous(data) => data,
            Self::Split(mut first, second) => {
                first.extend_from_slice(&second);
                first
            }
        }
    }
}

impl<T: AsRef<[u8]>> Index<usize> for Segments<T> {
    type Output = u8;

    #[inline]
    fn index(&self, index: usize) -> &u8 {
        let (first, second) = self.parts();
        if index < first.len() {
            &first[index]
        } else {
            &second[index - first.len()]
        }
    }
}

// Resolve bounds once before consumption. Overflow is an invalid selection,
// even when wrapping would happen to produce an in-bounds index.
#[inline]
pub(super) fn bounds(range: impl RangeBounds<usize>, len: usize) -> Range<usize> {
    let start = match range.start_bound() {
        Bound::Included(&n) => n,
        Bound::Excluded(&n) => n.checked_add(1).expect("range start overflow"),
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&n) => n.checked_add(1).expect("range end overflow"),
        Bound::Excluded(&n) => n,
        Bound::Unbounded => len,
    };
    assert!(start <= end && end <= len, "range exceeds input");
    start..end
}

#[cfg(test)]
mod test {
    use std::ops::Bound::{Excluded, Included};

    use bytes::BytesMut;

    use super::Segments;

    #[test]
    fn parts_and_ranges_keep_byte_order_without_joining() {
        let bytes = b"abcdef";
        assert_eq!(
            Segments::Contiguous(&bytes[..]).parts(),
            (&b""[..], &bytes[..])
        );
        for boundary in 0..=bytes.len() {
            let input = Segments::Split(&bytes[..boundary], &bytes[boundary..]);
            assert_eq!(input.iter().copied().collect::<Vec<_>>(), bytes);
            for start in 0..=bytes.len() {
                for end in start..=bytes.len() {
                    assert_eq!(input.slice(start..end).to_vec(), &bytes[start..end]);
                }
            }
            assert_eq!(input.slice((Excluded(0), Included(3))).to_vec(), b"bcd");
            assert_eq!(
                input[boundary.min(bytes.len() - 1)],
                bytes[boundary.min(bytes.len() - 1)]
            );
            assert!(std::panic::catch_unwind(|| input.slice(..=usize::MAX)).is_err());
        }
        let contiguous = BytesMut::from(&bytes[..]);
        let ptr = contiguous.as_ptr();
        assert_eq!(Segments::Contiguous(contiguous).join().as_ptr(), ptr);
        let split = Segments::Split(BytesMut::from(&bytes[..2]), BytesMut::from(&bytes[2..]));
        assert_eq!(&split.join()[..], bytes);
    }
}
