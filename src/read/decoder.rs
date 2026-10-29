use std::io;
use std::ops::{Bound, Index, RangeBounds};

use bytes::BytesMut;

/// A framing decision returned by a [`Framer`].
///
/// Use [`ready_split`](Self::ready_split) or [`ready_copy`](Self::ready_copy)
/// to select framed bytes, [`skip`](Self::skip) to discard input, and
/// [`need_more`](Self::need_more) to consume bytes and request more input.
///
/// Use [`kind`](Self::kind) to match the action's kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame(pub(super) FrameInner);

/// The kind of action returned by a [`Framer`].
///
/// Returned by [`Frame::kind`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameKind {
    /// Selects bytes to decode into an Item.
    Ready,
    /// Skips input without producing an Item.
    Skip,
    /// Requests more input after any requested consumption.
    NeedMore,
}

// Skip and NeedMore can be dispatched without inspecting a ready selection.
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum FrameInner {
    Ready(FrameReady),
    Skip(usize),
    NeedMore(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum FrameReady {
    Split { range: FrameRange, addition: usize },
    Copy { range: FrameRange, consume: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct FrameRange {
    start: Bound<usize>,
    end: Bound<usize>,
}

impl<R> From<R> for FrameRange
where
    R: RangeBounds<usize>,
{
    fn from(range: R) -> Self {
        Self {
            start: range.start_bound().cloned(),
            end: range.end_bound().cloned(),
        }
    }
}

impl Index<FrameRange> for [u8] {
    type Output = [u8];

    #[inline]
    fn index(&self, range: FrameRange) -> &Self::Output {
        self.index((range.start_bound().cloned(), range.end_bound().cloned()))
    }
}

impl FrameRange {
    #[inline]
    pub fn bounds(&self) -> (Bound<&usize>, Bound<&usize>) {
        (self.start_bound(), self.end_bound())
    }

    #[inline]
    pub fn start_bound(&self) -> Bound<&usize> {
        self.start.as_ref()
    }

    #[inline]
    pub fn end_bound(&self) -> Bound<&usize> {
        self.end.as_ref()
    }
}

impl Frame {
    /// Returns the kind of this action.
    ///
    /// # Examples
    ///
    /// ```
    /// use decen::read::{Frame, FrameKind};
    ///
    /// let frame = Frame::ready_split(..3, 0);
    /// let action = match frame.kind() {
    ///     FrameKind::Ready => "decode an Item",
    ///     FrameKind::Skip => "skip input",
    ///     FrameKind::NeedMore => "read more input",
    /// };
    /// assert_eq!(action, "decode an Item");
    /// ```
    #[inline]
    pub fn kind(&self) -> FrameKind {
        match &self.0 {
            FrameInner::Ready(_) => FrameKind::Ready,
            FrameInner::Skip(_) => FrameKind::Skip,
            FrameInner::NeedMore(_) => FrameKind::NeedMore,
        }
    }

    /// Selects `range` for decoding and consumes through its end plus
    /// `addition` bytes.
    ///
    /// Bytes before the range and the additional bytes after it are discarded.
    /// Use [`ready_copy`](Self::ready_copy) to select bytes independently of
    /// how much input is consumed.
    ///
    /// # Contract
    ///
    /// `range` must index a valid slice of the input passed to the framer.
    /// The input must contain at least `addition` bytes after that slice.
    /// Unbounded endpoints refer to the beginning or end of the input.
    /// The range's end plus `addition` must not overflow.
    #[inline]
    pub fn ready_split(range: impl RangeBounds<usize>, addition: usize) -> Self {
        Self(FrameInner::Ready(FrameReady::Split {
            range: range.into(),
            addition,
        }))
    }

    /// Selects `range` for decoding and consumes the first `consume` bytes.
    ///
    /// Selection is independent of consumption, allowing adjacent frames to
    /// overlap. Empty ranges and zero consumption are allowed.
    ///
    /// If your framer uses this decision, consider implementing
    /// [`OwnedDecoder`] so callers can obtain owned Items by copying only the
    /// data they need, without first copying the entire frame into a
    /// `BytesMut`.
    ///
    /// For example, `Frame::ready_copy(..4, 2)` selects the first four bytes
    /// while leaving the input from offset two available for the next Item.
    ///
    /// # Contract
    ///
    /// `range` must index a valid slice of the input passed to the framer.
    /// `consume` must not exceed the input length. Unbounded endpoints refer
    /// to the beginning or end of the input. `consume` need not equal or cover
    /// the selection.
    #[inline]
    pub fn ready_copy(range: impl RangeBounds<usize>, consume: usize) -> Self {
        Self(FrameInner::Ready(FrameReady::Copy {
            range: range.into(),
            consume,
        }))
    }

    /// Consumes some bytes, then requests more input.
    ///
    /// Use `need_more(0)` to retain all current input. The reader attempts a
    /// read before calling the framer again, even if some input remains.
    /// Consumption is not rolled back if reading returns zero, fails, or panics.
    ///
    /// # Contract
    ///
    /// `consume` must not exceed the length of the input supplied to the
    /// current framer call.
    #[inline]
    pub fn need_more(consume: usize) -> Self {
        Self(FrameInner::NeedMore(consume))
    }

    /// Skips the first `amount` bytes without producing an Item.
    ///
    /// # Contract
    ///
    /// `amount` must not exceed the length of the input supplied to the
    /// current framer call.
    #[inline]
    pub fn skip(amount: usize) -> Self {
        Self(FrameInner::Skip(amount))
    }
}

/// Identifies complete frames in buffered input.
///
/// [`FramedRead`](super::FramedRead) uses this trait to decide which bytes to
/// pass to a decoder and when to read more input. To decode Items with
/// `FramedRead`, the same type must also implement at least one of
/// [`BytesDecoder`], [`ViewDecoder`], or [`OwnedDecoder`]. These traits define
/// how framed bytes become Items.
///
/// Implementations may keep parsing state in `self`, such as a length read
/// from a header or whether a quoted field is still open. Formats that need
/// no state can use a unit struct.
pub trait Framer {
    /// The error type shared by framing, decoding, and input I/O.
    ///
    /// [`FramedRead`](super::FramedRead) converts source errors through
    /// `From<io::Error>`. The default [`frame_eof`](Self::frame_eof) also uses
    /// this conversion to report incomplete input. Use [`io::Error`] directly
    /// if the format does not need its own error type.
    type Error: From<io::Error>;

    /// Examines `src` and returns the next framing decision.
    ///
    /// `src` contains input that has not been consumed, including any bytes
    /// retained from an earlier call. It may be empty.
    ///
    /// When a complete frame is available, return [`Frame::ready_split`] or
    /// [`Frame::ready_copy`] to select its bytes. The reader passes that
    /// selection to the decoder and consumes the amount specified by the
    /// decision. Return [`Frame::skip`] to discard input without decoding an
    /// Item and continue framing the remaining input.
    ///
    /// When more bytes are needed, return [`Frame::need_more`] to request more
    /// input. This decision can also consume bytes before reading:
    /// `Frame::need_more(consume)` consumes the first `consume` bytes and retains
    /// the rest of `src`. `Frame::need_more(0)` retains all of `src`.
    ///
    /// Each decision must satisfy its constructor's contract. Adjust saved
    /// scan offsets to account for consumed bytes, which are no longer part
    /// of the input on the next call.
    ///
    /// # Errors
    ///
    /// Return an error when the input cannot be framed. The reader returns it
    /// to the caller without consuming input for this decision. Earlier
    /// consumption and changes to format state are not rolled back.
    /// Implementations should document whether framing can continue after an
    /// error.
    fn frame(&mut self, src: &[u8]) -> Result<Frame, Self::Error>;

    /// Examines remaining input when the source has reached EOF.
    ///
    /// [`FramedRead`](super::FramedRead) calls this method after a read returns
    /// zero bytes. `src` may contain a final frame or incomplete input, or it
    /// may be empty. The contracts of [`frame`](Self::frame) also apply here.
    ///
    /// The default calls `frame` and reports [`io::ErrorKind::UnexpectedEof`]
    /// if the decision requests more input while retaining bytes from `src`.
    /// Override it when the format accepts a final frame without a terminator
    /// or needs to check unfinished parsing state.
    ///
    /// Return a ready decision for another frame, or [`Frame::skip`] to discard
    /// input and continue framing. Return [`Frame::need_more`] when no further
    /// frame is available. The reader then attempts another read and returns
    /// `None` if the source still supplies no bytes, even if buffered input
    /// remains. That input is retained for later calls.
    ///
    /// EOF need not be permanent. Later decoding calls can invoke this method
    /// again with the remaining input, even if the source has not resumed.
    /// If more bytes arrive, framing continues through `frame`. An override
    /// can return `Frame::need_more(0)` to retain an incomplete frame until
    /// more input becomes available.
    ///
    /// # Errors
    ///
    /// The default implementation returns errors from [`frame`](Self::frame)
    /// and reports incomplete trailing input as
    /// [`io::ErrorKind::UnexpectedEof`].
    fn frame_eof(&mut self, src: &[u8]) -> Result<Frame, Self::Error> {
        let frame = self.frame(src)?;
        match frame.0 {
            // Consuming a prefix does not complete the remaining input.
            FrameInner::NeedMore(consume) if consume < src.len() => {
                Err(io::Error::new(io::ErrorKind::UnexpectedEof, "expect more to read").into())
            }
            _ => Ok(frame),
        }
    }
}

/// Decodes framed bytes from an owned [`BytesMut`] into Items.
///
/// [`FramedRead::decode_bytes`](super::FramedRead::decode_bytes) uses this
/// trait after [`Framer`] has selected a complete frame. The decoder takes
/// ownership of those bytes, so an Item can retain them across later reads.
///
/// A framer returning [`Frame::ready_split`] enables zero-copy transfer;
/// [`Frame::ready_copy`] requires a copy before decoding. Implement
/// [`OwnedDecoder`] as well when callers could instead copy only the fields
/// they need, or [`ViewDecoder`] to support borrowed Items.
pub trait BytesDecoder: Framer {
    /// The decoded Item, which may retain the supplied [`BytesMut`].
    type Item;

    /// Decodes one complete Item from `frame`.
    ///
    /// The reader supplies exactly the bytes selected by [`Framer`], which may
    /// be empty. Interpret them and return the decoded Item. The result may
    /// retain `frame` or use another representation, but cannot borrow this
    /// call's decoder reference.
    ///
    /// # Errors
    ///
    /// Return an error if `frame` cannot be decoded. The reader has already
    /// applied the framing decision's consumption and does not undo it.
    fn decode_bytes(&mut self, frame: BytesMut) -> Result<Self::Item, Self::Error>;
}

/// Decodes framed bytes into Items that may borrow the input.
///
/// [`FramedRead::decode_view`](super::FramedRead::decode_view) uses this trait
/// after [`Framer`] has selected a complete frame. An Item can refer directly
/// to those bytes, making this suitable for processing an Item before reading
/// the next one.
///
/// Implement independently or alongside [`BytesDecoder`] and [`OwnedDecoder`].
/// An Item may own converted fields, such as unescaped text, as well as borrow
/// input.
///
/// Although an Item can be fully owned, using this trait for results that do
/// not borrow the input is discouraged. Implement [`OwnedDecoder`] instead.
pub trait ViewDecoder: Framer {
    /// The decoded result, which may borrow the frame for `'a`.
    type Item<'a>;

    /// Decodes one complete Item from `frame`, optionally borrowing its bytes.
    ///
    /// The reader supplies exactly the bytes selected by [`Framer`], which may
    /// be empty. The returned Item may borrow `frame` for `'a`, but cannot
    /// borrow this call's decoder reference. Producing owned fields may
    /// allocate or copy data.
    ///
    /// # Errors
    ///
    /// Return an error if `frame` cannot be decoded. The reader has already
    /// applied the framing decision's consumption and does not undo it.
    fn decode_view<'a>(&mut self, frame: &'a [u8]) -> Result<Self::Item<'a>, Self::Error>;
}

/// Decodes framed bytes into Items that do not borrow the input.
///
/// [`FramedRead::decode_owned`](super::FramedRead::decode_owned) uses this
/// trait after [`Framer`] has selected a complete frame. The decoder receives
/// a slice and returns an Item independent of that slice's lifetime. Explicitly
/// copy any input data the Item needs to retain; parsed values need no copy
/// of the input bytes.
///
/// Implement independently or alongside [`BytesDecoder`] and [`ViewDecoder`].
/// This is useful with [`Frame::ready_copy`], since it avoids copying the
/// entire frame into a [`BytesMut`] before decoding.
///
/// An existing [`ViewDecoder`] implementation can be reused when its Item
/// provides a conversion such as `into_owned()` to a result that no longer
/// borrows the input. For example, [`decode_owned`](Self::decode_owned) can
/// return `Ok(self.decode_view(frame)?.into_owned())`. That conversion may
/// allocate and copy borrowed fields; its cost is part of decoding.
pub trait OwnedDecoder: Framer {
    /// The decoded result, independent of the input slice's lifetime.
    type Item;

    /// Decodes one complete Item from `frame`.
    ///
    /// The reader supplies exactly the bytes selected by [`Framer`], which may
    /// be empty. Return an Item containing the parsed values and any copied
    /// input data it needs. The result cannot borrow `frame` or this call's
    /// decoder reference.
    ///
    /// # Errors
    ///
    /// Return an error if `frame` cannot be decoded. The reader has already
    /// applied the framing decision's consumption and does not undo it.
    fn decode_owned(&mut self, frame: &[u8]) -> Result<Self::Item, Self::Error>;
}

#[cfg(test)]
mod test {
    use std::io;

    use super::{Frame, Framer};

    #[test]
    fn default_eof_distinguishes_completion_from_truncation() {
        struct Pairs;
        impl Framer for Pairs {
            type Error = io::Error;

            fn frame(&mut self, src: &[u8]) -> io::Result<Frame> {
                Ok(if src.len() < 2 {
                    Frame::need_more(0)
                } else {
                    Frame::ready_split(..2, 0)
                })
            }
        }

        let mut codec = Pairs;
        assert_eq!(codec.frame(b"").unwrap(), Frame::need_more(0));
        assert_eq!(codec.frame_eof(b"").unwrap(), Frame::need_more(0));
        assert!(codec.frame_eof(b"a").is_err());
        assert_eq!(codec.frame_eof(b"ab").unwrap(), Frame::ready_split(..2, 0));
    }

    #[test]
    fn default_eof_preserves_explicit_decisions_and_format_errors() {
        struct Decision(io::Result<Frame>);
        impl Framer for Decision {
            type Error = io::Error;

            fn frame(&mut self, _: &[u8]) -> io::Result<Frame> {
                std::mem::replace(&mut self.0, Ok(Frame::need_more(0)))
            }
        }

        for frame in [
            Frame::skip(1),
            Frame::need_more(2),
            Frame::ready_split(..0, 0),
            Frame::ready_copy(..2, 1),
        ] {
            assert_eq!(Decision(Ok(frame.clone())).frame_eof(b"ab").unwrap(), frame);
        }
        for consume in [0, 1] {
            assert_eq!(
                Decision(Ok(Frame::need_more(consume)))
                    .frame_eof(b"ab")
                    .unwrap_err()
                    .kind(),
                io::ErrorKind::UnexpectedEof
            );
        }
        let error = io::Error::new(io::ErrorKind::InvalidData, "invalid header");
        assert_eq!(
            Decision(Err(error)).frame_eof(b"ab").unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn frame_bounds_preserve_the_original_bounds_and_index_the_selected_bytes() {
        use std::ops::Bound::{Excluded, Included, Unbounded};

        use super::{FrameInner, FrameRange, FrameReady};

        let input: &[u8] = b"abcdef";
        for (range, expected) in [
            (FrameRange::from(..4), &input[..4]),
            (FrameRange::from(0..4), &input[..4]),
            (FrameRange::from(1..=2), &input[1..3]),
            (FrameRange::from((Excluded(0), Included(2))), &input[1..3]),
            (FrameRange::from(..), input),
            (FrameRange::from(0..), input),
        ] {
            assert_eq!(&input[range], expected);
        }
        for decision in [
            Frame::ready_split((Excluded(0), Included(2)), 4),
            Frame::ready_copy((Excluded(0), Included(2)), 1),
        ] {
            let range = match decision.0 {
                FrameInner::Ready(FrameReady::Split { range, .. })
                | FrameInner::Ready(FrameReady::Copy { range, .. }) => range,
                _ => unreachable!(),
            };
            assert_eq!(range.start_bound().cloned(), Excluded(0));
            assert_eq!(range.end_bound().cloned(), Included(2));
        }
        let full = FrameRange::from(..);
        assert_eq!(
            (full.start_bound(), full.end_bound()),
            (Unbounded, Unbounded)
        );
        // Constructors preserve invalid bounds; using the selection validates
        // them.
        for bounds in [
            (Included(2), Excluded(1)),
            (Excluded(usize::MAX), Unbounded),
            (Unbounded, Included(usize::MAX)),
        ] {
            let _ = Frame::ready_split(bounds, usize::MAX);
            let _ = Frame::ready_copy(bounds, 0);
            assert!(std::panic::catch_unwind(|| &input[FrameRange::from(bounds)]).is_err());
        }
        let _ = Frame::skip(0);
    }
}
