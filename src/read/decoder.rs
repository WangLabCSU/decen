use std::io;
use std::ops::{Bound, RangeBounds};

use bytes::BytesMut;

/// Input consumption and the next action requested by a [`Framer`].
///
/// [`action`](Self::action) selects an Item, continues framing, or requests
/// more input.
/// [`consume`](Self::consume) reports an explicit consumption count, if set.
/// By default, a ready decision consumes through the selection's end; other
/// actions consume nothing.
///
/// Any selection refers to the input before consumption.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    consume: Option<usize>,
    action: FrameAction,
}

/// What the reader should do after applying a framing decision's consumption.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameAction {
    /// Decodes the selected bytes into one Item before framing again.
    ///
    /// The arguments are the start and end bounds, respectively.
    /// Unbounded endpoints refer to the beginning or end of the input supplied
    /// to the framer. Selection is independent of [`Frame::consume`], allowing
    /// adjacent frames to overlap. By default, the reader consumes through the
    /// selection's end. Empty selections and explicit zero consumption are
    /// allowed. Selecting `1..4` with consumption of 5 bytes selects `abc`
    /// from `<abc>tail` and retains `tail`.
    Ready(Bound<usize>, Bound<usize>),
    /// Immediately frames the remaining input without reading or decoding.
    Reframe,
    /// Attempts a read before framing again, even if some input remains.
    ///
    /// The reader consumes the requested prefix and retains the rest of the
    /// input. A read that returns zero, fails, or panics does not undo
    /// consumption.
    NeedMore,
}

impl FrameAction {
    /// Creates a [`Ready`](Self::Ready) action selecting `range`.
    #[inline]
    pub fn ready(range: impl RangeBounds<usize>) -> Self {
        Self::Ready(range.start_bound().cloned(), range.end_bound().cloned())
    }
}

impl Frame {
    /// Creates a decision with `action` and its default consumption.
    ///
    /// A ready decision consumes through the selection's end; other actions
    /// consume nothing. Use [`with_consume`](Self::with_consume) to override it.
    #[inline]
    pub fn new(action: FrameAction) -> Self {
        Self {
            consume: None,
            action,
        }
    }

    /// Returns this decision with an explicit input consumption count.
    #[inline]
    pub fn with_consume(mut self, consume: usize) -> Self {
        self.consume = Some(consume);
        self
    }

    /// Returns this decision with `action`, keeping any explicit consumption count.
    ///
    /// If consumption has not been set, the new action's default applies.
    ///
    /// ```
    /// use decen::read::{Frame, FrameAction};
    ///
    /// let frame = Frame::reframe()
    ///     .with_consume(4)
    ///     .with_action(FrameAction::NeedMore);
    /// assert_eq!(frame.consume(), Some(4));
    /// assert_eq!(frame.action(), &FrameAction::NeedMore);
    /// ```
    #[inline]
    pub fn with_action(mut self, action: FrameAction) -> Self {
        self.action = action;
        self
    }

    /// Returns the explicit consumption count, or `None` for the action's default.
    ///
    /// The default consumes through a ready selection's end, including all
    /// input when its end is unbounded. Other actions consume nothing.
    ///
    /// ```
    /// use decen::read::Frame;
    ///
    /// let frame = Frame::ready(..4);
    /// assert_eq!(frame.consume(), None);
    /// let frame = frame.with_consume(0);
    /// assert_eq!(frame.consume(), Some(0));
    /// ```
    #[inline]
    pub fn consume(&self) -> Option<usize> {
        self.consume
    }

    /// Returns the next action, including the selection for a ready Item.
    ///
    /// # Examples
    ///
    /// ```
    /// use decen::read::{Frame, FrameAction};
    ///
    /// let input: &[u8] = b"<abc>tail";
    /// let frame = Frame::new(FrameAction::ready(1..4)).with_consume(5);
    /// assert_eq!(frame.consume(), Some(5));
    /// match frame.action() {
    ///     FrameAction::Ready(start, end) => {
    ///         assert_eq!(&input[(*start, *end)], b"abc");
    ///     }
    ///     FrameAction::Reframe | FrameAction::NeedMore => unreachable!(),
    /// }
    /// ```
    #[inline]
    pub fn action(&self) -> &FrameAction {
        &self.action
    }

    /// Creates a [`FrameAction::Ready`] decision selecting `range` and requesting
    /// consumption through its end by default.
    ///
    /// For example, `ready(1..4)` consumes 4 bytes, and `ready(..)` consumes all
    /// input supplied to the framer. Use [`with_consume`](Self::with_consume) to
    /// override this, including with zero to retain all input.
    #[inline]
    pub fn ready(range: impl RangeBounds<usize>) -> Self {
        Self::new(FrameAction::ready(range))
    }

    /// Creates a [`FrameAction::NeedMore`] decision with zero consumption.
    ///
    /// Use [`with_consume`](Self::with_consume) to request prefix consumption.
    #[inline]
    pub fn need_more() -> Self {
        Self::new(FrameAction::NeedMore)
    }

    /// Creates a [`FrameAction::Reframe`] decision with zero consumption.
    ///
    /// Use [`with_consume`](Self::with_consume) to request prefix consumption.
    #[inline]
    pub fn reframe() -> Self {
        Self::new(FrameAction::Reframe)
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
/// Framing and decoding can be stateful. Implementations may keep parsing
/// state in `self` across calls.
///
/// [`FramedRead`](super::FramedRead) uses this call sequence on the same
/// format instance:
///
/// - Call [`frame`](Self::frame) with the buffered input, which may be empty.
/// - For a valid [`FrameAction::Ready`] decision, consume the requested prefix
///   and call exactly one decoder method with the selected bytes before framing
///   again. The selection's contents and the format's parsing state are unchanged
///   between framing and decoding. The decoding method may differ per Item.
/// - For [`FrameAction::NeedMore`], consume the requested prefix and attempt a
///   read before framing again. Retained bytes stay unchanged; new bytes are
///   appended at the end. A failed read does not undo consumption.
/// - For [`FrameAction::Reframe`], consume the requested prefix and frame again
///   without decoding or reading.
///
/// After a read returns zero, framing uses [`frame_eof`](Self::frame_eof)
/// until more input arrives. A framing error or rejected decision consumes
/// nothing for that call; a decoding error leaves consumption applied. The
/// same applies if either operation panics. Implementations must remain safe
/// to retry with the corresponding remaining input.
///
/// These rules allow framing to reuse scan offsets and decoding to reuse
/// information recorded while framing. Direct callers must maintain the same input
/// continuity and call sequence and satisfy each method's safety requirements.
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
    /// Use [`FrameAction::Ready`] to select bytes for decoding,
    /// [`FrameAction::Reframe`] to frame again without reading or decoding,
    /// or [`FrameAction::NeedMore`] to request more input.
    ///
    /// Ready decisions consume through the selection's end by default; other
    /// actions consume nothing. Use [`Frame::with_consume`] to override the
    /// consumed prefix. The reader retains the rest of `src`.
    ///
    /// Selections refer to `src` before consumption. [`FramedRead`](super::FramedRead)
    /// panics if a selection is outside `src` or the consumption count exceeds
    /// its length, before consuming input, reading, or decoding for that
    /// decision. Adjust saved scan offsets to account for consumed bytes, which
    /// are no longer part of the input on the next call.
    ///
    /// # Errors
    ///
    /// Return an error when the input cannot be framed. The reader returns it
    /// to the caller without consuming input for this decision. Earlier
    /// consumption and changes to format state are not rolled back.
    /// Implementations should document whether framing can continue after an
    /// error.
    ///
    /// # Safety
    ///
    /// Between calls to `frame`, the unconsumed contents of `src` must remain
    /// unchanged, with new bytes appended only at the end.
    unsafe fn frame(&mut self, src: &[u8]) -> Result<Frame, Self::Error>;

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
    /// Use [`FrameAction::Ready`] for another Item or [`FrameAction::Reframe`]
    /// to frame again. Use [`FrameAction::NeedMore`] when no further frame is
    /// available. The reader then attempts another read and returns `None` if
    /// the source still supplies no bytes, even if buffered input remains.
    /// That input is retained for later calls.
    ///
    /// EOF need not be permanent. Later decoding calls can invoke this method
    /// again with the remaining input, even if the source has not resumed.
    /// If more bytes arrive, framing continues through `frame`. An override
    /// can request [`FrameAction::NeedMore`] with zero consumption to retain an
    /// incomplete frame until more input becomes available.
    ///
    /// # Errors
    ///
    /// The default implementation returns errors from [`frame`](Self::frame)
    /// and reports incomplete trailing input as
    /// [`io::ErrorKind::UnexpectedEof`].
    ///
    /// # Safety
    ///
    /// The input and decoder state must satisfy [`frame`](Self::frame)'s
    /// safety requirements. This method may be called repeatedly at EOF;
    /// framing may resume through `frame` if further input arrives.
    unsafe fn frame_eof(&mut self, src: &[u8]) -> Result<Frame, Self::Error> {
        // SAFETY: frame_eof forwards the same input and framing state.
        let frame = unsafe { self.frame(src)? };
        match frame.action() {
            // Consuming a prefix does not complete the remaining input.
            FrameAction::NeedMore if frame.consume().unwrap_or(0) < src.len() => {
                Err(io::Error::new(io::ErrorKind::UnexpectedEof, "incomplete frame at EOF").into())
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
/// When consumption covers the prefix through the selection's end, its bytes
/// are transferred without copying. Otherwise, the reader copies the selection
/// before decoding. Implement [`OwnedDecoder`] as well when callers could
/// instead copy only the fields they need, or [`ViewDecoder`] to support borrowed
/// Items.
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
    ///
    /// # Safety
    ///
    /// The most recent framing call on this decoder must have returned a
    /// [`FrameAction::Ready`] decision that has not yet been decoded. `frame`
    /// must contain exactly its selected bytes, in order and unchanged, and
    /// the decoder's parsing state must not have been changed since that call.
    unsafe fn decode_bytes(&mut self, frame: BytesMut) -> Result<Self::Item, Self::Error>;
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
    ///
    /// # Safety
    ///
    /// The most recent framing call on this decoder must have returned a
    /// [`FrameAction::Ready`] decision that has not yet been decoded. `frame`
    /// must contain exactly its selected bytes, in order and unchanged, and
    /// the decoder's parsing state must not have been changed since that call.
    unsafe fn decode_view<'a>(&mut self, frame: &'a [u8]) -> Result<Self::Item<'a>, Self::Error>;
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
/// This avoids copying the entire frame into a [`BytesMut`] when the selection
/// overlaps input retained for the next Item.
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
    ///
    /// # Safety
    ///
    /// The most recent framing call on this decoder must have returned a
    /// [`FrameAction::Ready`] decision that has not yet been decoded. `frame`
    /// must contain exactly its selected bytes, in order and unchanged, and
    /// the decoder's parsing state must not have been changed since that call.
    unsafe fn decode_owned(&mut self, frame: &[u8]) -> Result<Self::Item, Self::Error>;
}

#[cfg(test)]
mod test {
    use std::io;

    use super::{Frame, FrameAction, Framer};

    #[test]
    fn default_eof_distinguishes_completion_from_truncation() {
        struct Pairs;
        impl Framer for Pairs {
            type Error = io::Error;

            unsafe fn frame(&mut self, src: &[u8]) -> io::Result<Frame> {
                Ok(if src.len() < 2 {
                    Frame::need_more()
                } else {
                    Frame::ready(..2).with_consume(2)
                })
            }
        }

        let mut codec = Pairs;
        // SAFETY: start with empty input, then retain it and append bytes.
        // No call consumes input or produces a ready frame until the last.
        unsafe {
            assert_eq!(codec.frame(b"").unwrap(), Frame::need_more());
            assert_eq!(codec.frame_eof(b"").unwrap(), Frame::need_more());
            assert!(codec.frame_eof(b"a").is_err());
            assert_eq!(
                codec.frame_eof(b"ab").unwrap(),
                Frame::ready(..2).with_consume(2)
            );
        }
    }

    #[test]
    fn default_eof_preserves_explicit_decisions_and_format_errors() {
        struct Decision(io::Result<Frame>);
        impl Framer for Decision {
            type Error = io::Error;

            unsafe fn frame(&mut self, src: &[u8]) -> io::Result<Frame> {
                if src.is_empty() {
                    return Ok(Frame::need_more());
                }
                std::mem::replace(&mut self.0, Ok(Frame::need_more()))
            }
        }

        fn at_eof(decision: io::Result<Frame>) -> io::Result<Frame> {
            let mut codec = Decision(decision);
            // SAFETY: start with empty input, then supply the appended bytes.
            // The empty call neither consumes input nor produces an Item.
            unsafe {
                codec.frame(b"")?;
                codec.frame_eof(b"ab")
            }
        }

        for frame in [
            Frame::reframe().with_consume(1),
            Frame::need_more().with_consume(2),
            Frame::ready(..0),
            Frame::ready(..2).with_consume(1),
        ] {
            assert_eq!(at_eof(Ok(frame.clone())).unwrap(), frame);
        }
        for consume in [0, 1] {
            assert_eq!(
                at_eof(Ok(Frame::need_more().with_consume(consume)))
                    .unwrap_err()
                    .kind(),
                io::ErrorKind::UnexpectedEof
            );
        }
        let error = io::Error::new(io::ErrorKind::InvalidData, "invalid header");
        assert_eq!(
            at_eof(Err(error)).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    fn frame_bounds_preserve_the_original_bounds_and_index_the_selected_bytes() {
        use std::ops::Bound::{Excluded, Included, Unbounded};
        use std::ops::RangeBounds;

        let input: &[u8] = b"abcdef";
        for (frame, expected) in [
            (Frame::ready(..4), &input[..4]),
            (Frame::ready(0..4), &input[..4]),
            (Frame::ready(1..=2), &input[1..3]),
            (Frame::ready((Excluded(0), Included(2))), &input[1..3]),
            (Frame::ready(..), input),
            (Frame::ready(0..), input),
        ] {
            let FrameAction::Ready(start, end) = frame.action() else {
                unreachable!();
            };
            assert_eq!(&input[(*start, *end)], expected);
        }
        for decision in [
            Frame::ready((Excluded(0), Included(2))).with_consume(4),
            Frame::ready((Excluded(0), Included(2))).with_consume(1),
        ] {
            let range = match decision.action() {
                FrameAction::Ready(start, end) => (*start, *end),
                _ => unreachable!(),
            };
            assert_eq!(range.start_bound().cloned(), Excluded(0));
            assert_eq!(range.end_bound().cloned(), Included(2));
        }
        assert_eq!(
            Frame::ready(..).action(),
            &FrameAction::Ready(Unbounded, Unbounded)
        );
        // Constructors preserve invalid bounds; using the selection validates
        // them.
        for bounds in [
            (Included(2), Excluded(1)),
            (Excluded(usize::MAX), Unbounded),
            (Unbounded, Included(usize::MAX)),
        ] {
            for consume in [0, usize::MAX] {
                let frame = Frame::ready(bounds).with_consume(consume);
                let FrameAction::Ready(start, end) = frame.action() else {
                    unreachable!();
                };
                assert_eq!((*start, *end), bounds);
                assert!(std::panic::catch_unwind(|| &input[(*start, *end)]).is_err());
            }
        }
        let _ = Frame::reframe();
    }
}
