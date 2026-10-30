//! Framing and decoding byte streams with [`FramedRead`].
//!
//! The `nightly` feature uses `Read::read_buf` and requires nightly Rust.

use std::io::Read;

mod buffer;
mod decoder;
mod segments;

pub use self::decoder::{BytesDecoder, Frame, FrameKind, Framer, OwnedDecoder, ViewDecoder};
pub use self::segments::Segments;

use self::buffer::Buffer;
use self::decoder::{FrameInner, FrameReady};
use crate::DEFAULT_BUF_SIZE;

/// A buffered reader that decodes a byte stream into Items.
///
/// Reads bytes from a [`Read`] source as needed, uses the supplied [`Framer`]
/// to select a complete frame, and passes that frame to the decoder. The format
/// implements [`Framer`] and one or more of [`BytesDecoder`], [`ViewDecoder`],
/// and [`OwnedDecoder`].
///
/// Framing and decoding use the same format instance throughout reading. It
/// can retain parsing state across calls, including while awaiting more input.
///
/// Choose a decoding method based on how the Item uses its input:
///
/// - [`decode_view`](Self::decode_view) requires [`ViewDecoder`] and
///   passes framed bytes as a slice. The returned Item can borrow from that
///   slice, making this suitable for processing an Item before reading the
///   next one. Frames within one input part are borrowed directly; frames
///   crossing parts are joined before decoding, which can allocate and copy.
/// - [`decode_owned`](Self::decode_owned) requires [`OwnedDecoder`] and
///   passes framed bytes as [`Segments<&[u8]>`](Segments), without joining
///   the parts. The returned Item cannot borrow the input. The decoder copies
///   any input data the Item needs to retain.
/// - [`decode_bytes`](Self::decode_bytes) requires [`BytesDecoder`] and
///   passes framed bytes as an owned [`bytes::BytesMut`]. The Item can retain
///   those bytes across reads without borrowing from the reader.
///
/// With `decode_bytes` and [`Frame::ready_split`], frames within one input part
/// are transferred without copying; frames crossing parts are joined, which
/// copies bytes. [`Frame::ready_copy`] copies the selected bytes. For
/// `ready_copy`, prefer
/// [`decode_owned`](Self::decode_owned) when an owned result is needed and the
/// decoder implements [`OwnedDecoder`], so it can copy only the data the Item
/// needs.
///
/// `ready_copy` allows adjacent frames to overlap by selecting bytes
/// independently of how many are consumed. If your framer uses `ready_copy`,
/// consider implementing [`OwnedDecoder`] so callers can obtain owned Items
/// without first copying the entire frame into a `BytesMut`.
///
/// Reading and buffering may still allocate or copy input bytes. Capacity is
/// not a memory limit: large Items can grow the buffer, and retained owned
/// Items can keep shared allocations alive after the reader is dropped.
///
/// # Examples
///
/// Reading one borrowed pair of bytes at a time:
///
/// ```
/// use std::io;
///
/// use decen::read::{Frame, FramedRead, Framer, Segments, ViewDecoder};
///
/// struct Pairs;
///
/// impl Framer for Pairs {
///     type Error = io::Error;
///
///     fn frame(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
///         Ok(if src.len() < 2 {
///             Frame::need_more(0)
///         } else {
///             Frame::ready_split(..2, 0)
///         })
///     }
/// }
///
/// impl ViewDecoder for Pairs {
///     type Item<'a> = &'a [u8];
///
///     fn decode_view<'a>(
///         &mut self,
///         frame: &'a [u8],
///     ) -> io::Result<Self::Item<'a>> {
///         Ok(frame)
///     }
/// }
///
/// let mut reader = FramedRead::with_capacity(&b"\xff\0ab"[..], Pairs, 1);
/// for expected in [b"\xff\0", b"ab"] {
///     let pair = reader.decode_view()?.unwrap();
///     assert_eq!(pair, expected);
/// }
/// assert!(reader.decode_view()?.is_none());
/// # Ok::<(), io::Error>(())
/// ```
#[derive(Debug)]
pub struct FramedRead<R: ?Sized + Read, D> {
    decoder: D,
    buffer: Buffer,
    eof: bool,
    inner: R,
}

impl<R: Read, D> FramedRead<R, D> {
    /// Creates a reader using `decoder` and the default buffer capacity.
    ///
    /// The current default is 8 KiB, or 512 bytes on ESP-IDF.
    pub fn new(inner: R, decoder: D) -> Self {
        Self::with_capacity(inner, decoder, DEFAULT_BUF_SIZE)
    }

    /// Creates a reader with an initial buffer capacity of at least `capacity`
    /// bytes.
    ///
    /// `capacity` does not limit Item size. Zero defers allocation until
    /// reading. Larger Items can grow the buffer; retained Items may keep
    /// memory in use beyond this initial capacity.
    pub fn with_capacity(inner: R, decoder: D, capacity: usize) -> Self {
        Self {
            decoder,
            buffer: Buffer::with_capacity(capacity),
            eof: false,
            inner,
        }
    }
}

impl<R: ?Sized + Read, D> FramedRead<R, D> {
    /// Returns a reference to the underlying reader.
    pub fn reader(&self) -> &R {
        &self.inner
    }

    /// Borrows the buffered input without joining its parts.
    pub fn buffer(&self) -> Segments<&[u8]> {
        self.buffer.buffer()
    }

    /// Returns a reference to the decoder.
    pub fn decoder(&self) -> &D {
        &self.decoder
    }
}

// Inline the decoding entry points into the caller's Item loop so framing
// and extraction can be optimized without passing an intermediate result.
impl<R: ?Sized + Read, D: Framer> FramedRead<R, D> {
    /// Decodes the next Item from owned framed bytes.
    ///
    /// Returns `None` when the input has ended and no further Item is
    /// available.
    ///
    /// # Errors
    ///
    /// Returns I/O, framing, or decoding errors. I/O errors, including
    /// [`std::io::ErrorKind::Interrupted`], are returned without retrying.
    ///
    /// # Panics
    ///
    /// Panics if the supplied [`Framer`] returns an invalid framing decision,
    /// or if the [`Framer`] or [`BytesDecoder`] implementation panics.
    #[inline(always)]
    pub fn decode_bytes(&mut self) -> Result<Option<D::Item>, D::Error>
    where
        D: BytesDecoder,
    {
        let Some(ready) = self.frame_ready()? else {
            return Ok(None);
        };
        let bytes = match ready {
            FrameReady::Copy { range, consume } => self.buffer.take_bytes(range.bounds(), consume),
            FrameReady::Split { range, addition } => {
                self.buffer.take_bytes_to(range.bounds(), addition)
            }
        };
        self.decoder.decode_bytes(bytes).map(Some)
    }

    /// Decodes the next Item from a slice of framed bytes.
    ///
    /// Returns `None` when the input has ended and no further Item is
    /// available.
    ///
    /// # Errors
    ///
    /// Returns I/O, framing, or decoding errors. I/O errors, including
    /// [`std::io::ErrorKind::Interrupted`], are returned without retrying.
    ///
    /// # Panics
    ///
    /// Panics if the supplied [`Framer`] returns an invalid framing decision,
    /// or if the [`Framer`] or [`ViewDecoder`] implementation panics.
    #[inline(always)]
    pub fn decode_view(&mut self) -> Result<Option<D::Item<'_>>, D::Error>
    where
        D: ViewDecoder,
    {
        let Some(ready) = self.frame_ready()? else {
            return Ok(None);
        };
        let slice = match ready {
            FrameReady::Copy { range, consume } => self.buffer.take_view(range.bounds(), consume),
            FrameReady::Split { range, addition } => {
                self.buffer.take_view_to(range.bounds(), addition)
            }
        };
        Ok(Some(self.decoder.decode_view(slice)?))
    }

    /// Decodes the next Item without borrowing framed bytes in the result.
    ///
    /// Returns `None` when the input has ended and no further Item is
    /// available.
    ///
    /// # Errors
    ///
    /// Returns I/O, framing, or decoding errors. I/O errors, including
    /// [`std::io::ErrorKind::Interrupted`], are returned without retrying.
    ///
    /// # Panics
    ///
    /// Panics if the supplied [`Framer`] returns an invalid framing decision,
    /// or if the [`Framer`] or [`OwnedDecoder`] implementation panics.
    #[inline(always)]
    pub fn decode_owned(&mut self) -> Result<Option<D::Item>, D::Error>
    where
        D: OwnedDecoder,
    {
        let Some(ready) = self.frame_ready()? else {
            return Ok(None);
        };
        let view = match ready {
            FrameReady::Copy { range, consume } => {
                self.buffer.take_view_segments(range.bounds(), consume)
            }
            FrameReady::Split { range, addition } => {
                self.buffer.take_view_segments_to(range.bounds(), addition)
            }
        };
        Ok(Some(self.decoder.decode_owned(view)?))
    }

    // Keep framing in the caller's Item loop.
    #[inline(always)]
    fn frame_ready(&mut self) -> Result<Option<FrameReady>, D::Error> {
        loop {
            let buf = self.buffer.buffer();
            let frame = if self.eof {
                std::hint::cold_path();
                self.decoder.frame_eof(buf)?
            } else {
                self.decoder.frame(buf)?
            };
            match frame.0 {
                FrameInner::Ready(ready) => return Ok(Some(ready)),

                FrameInner::Skip(amount) => {
                    self.buffer.consume(amount);
                }

                FrameInner::NeedMore(consume) => {
                    self.buffer.consume(consume);
                    let nread = self.buffer.write_buf(&mut self.inner)?;
                    if nread == 0 {
                        std::hint::cold_path();
                        if self.eof {
                            // A custom frame_eof may retain partial input.
                            // Wait for another call instead of polling EOF.
                            return Ok(None);
                        }
                        // Give frame_eof a chance to finish pending decoder
                        // state, even if no buffered bytes remain.
                        self.eof = true;
                        continue;
                    }

                    // Resume normal framing on new input.
                    self.eof = false;
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::{
        collections::VecDeque,
        io::{self, Cursor, ErrorKind, Read},
        panic::{AssertUnwindSafe, catch_unwind},
    };

    use bytes::{Bytes, BytesMut};

    use super::{BytesDecoder, Frame, FramedRead, Framer, OwnedDecoder, Segments, ViewDecoder};
    use crate::test::allocations;

    // Minimal framing fixture for I/O tests, not a public format
    // implementation.
    // Retain each LF in the selected bytes; accept a final nonempty EOF tail.
    #[derive(Default)]
    struct Lines {
        scanned: usize,
    }
    impl Framer for Lines {
        type Error = io::Error;
        fn frame(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
            if let Some(offset) = src.slice(self.scanned..).iter().position(|&b| b == b'\n') {
                let end = self.scanned + offset + 1;
                self.scanned = 0;
                Ok(Frame::ready_split(..end, 0))
            } else {
                self.scanned = src.len();
                Ok(Frame::need_more(0))
            }
        }
        fn frame_eof(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
            let frame = self.frame(src)?;
            if frame != Frame::need_more(0) {
                return Ok(frame);
            }
            self.scanned = 0;
            Ok(if src.is_empty() {
                Frame::need_more(0)
            } else {
                Frame::ready_split(.., 0)
            })
        }
    }
    impl BytesDecoder for Lines {
        type Item = Bytes;
        fn decode_bytes(&mut self, frame: BytesMut) -> io::Result<Bytes> {
            Ok(frame.freeze())
        }
    }
    impl ViewDecoder for Lines {
        type Item<'a> = &'a [u8];
        fn decode_view<'a>(&mut self, frame: &'a [u8]) -> io::Result<&'a [u8]> {
            Ok(frame)
        }
    }

    impl OwnedDecoder for Lines {
        type Item = Vec<u8>;
        fn decode_owned(&mut self, frame: Segments<&[u8]>) -> io::Result<Self::Item> {
            Ok(frame.to_vec())
        }
    }

    enum Step {
        Data(&'static [u8]),
        Error(ErrorKind),
        DirtyError(ErrorKind),
        Overreport,
        Panic,
        Eof,
    }
    struct Script {
        steps: VecDeque<Step>,
        reads: usize,
        destinations: Vec<usize>,
    }
    impl Script {
        fn new(steps: impl IntoIterator<Item = Step>) -> Self {
            Self {
                steps: steps.into_iter().collect(),
                reads: 0,
                destinations: Vec::new(),
            }
        }
    }
    impl Read for Script {
        fn read(&mut self, dst: &mut [u8]) -> io::Result<usize> {
            assert!(!dst.is_empty(), "empty destinations can produce false EOF");
            self.reads += 1;
            self.destinations.push(dst.as_ptr() as usize);
            match self.steps.pop_front().unwrap_or(Step::Eof) {
                Step::Data(bytes) => {
                    let n = bytes.len().min(dst.len());
                    dst[..n].copy_from_slice(&bytes[..n]);
                    if n < bytes.len() {
                        self.steps.push_front(Step::Data(&bytes[n..]));
                    }
                    Ok(n)
                }
                Step::Error(kind) => Err(kind.into()),
                Step::DirtyError(kind) => {
                    dst.fill(0xee);
                    Err(kind.into())
                }
                Step::Overreport => {
                    dst.fill(0xee);
                    Ok(dst.len() + 1)
                }
                Step::Panic => panic!("source panic"),
                Step::Eof => Ok(0),
            }
        }
    }
    struct ShortReads<R> {
        source: R,
        limit: usize,
    }
    impl<R: Read> Read for ShortReads<R> {
        fn read(&mut self, dst: &mut [u8]) -> io::Result<usize> {
            let len = dst.len().min(self.limit);
            self.source.read(&mut dst[..len])
        }
    }

    #[test]
    fn reader_borrows_a_complete_chunk_and_joins_only_crossing_frames() {
        let source = Script::new([Step::Data(b"a\nkey="), Step::Data(b"\xff\0\nb\n")]);
        let mut reader = FramedRead::with_capacity(source, Lines::default(), 32);
        let item = reader.decode_view().unwrap().unwrap();
        let address = item.as_ptr() as usize;
        assert_eq!(item, b"a\n");
        assert_eq!(reader.reader().destinations, [address]);
        assert_eq!(reader.decode_view().unwrap().unwrap(), b"key=\xff\0\n");
        // Both reads use the whole chunk. Only the crossing frame is joined;
        // the following frame still borrows its original read destination.
        assert_eq!(reader.reader().destinations, [address, address]);
        let item = reader.decode_view().unwrap().unwrap();
        assert_eq!(item, b"b\n");
        assert_eq!(item.as_ptr() as usize, address + 3);
    }

    #[test]
    fn unsized_source_supports_accessors_and_all_decode_modes() {
        let mut reader: Box<FramedRead<dyn Read, Lines>> =
            Box::new(FramedRead::new(Cursor::new(b"a\nb\nc\n"), Lines::default()));
        let _: &dyn Read = reader.reader();
        let _: &Lines = reader.decoder();
        assert!(reader.buffer().is_empty());
        assert_eq!(reader.decode_view().unwrap().unwrap(), b"a\n");
        assert_eq!(&reader.decode_bytes().unwrap().unwrap()[..], b"b\n");
        assert_eq!(reader.decode_owned().unwrap().unwrap(), b"c\n");
        assert!(reader.decode_view().unwrap().is_none());
    }

    #[test]
    fn multiple_items_drain_without_reads_and_eof_can_resume() {
        let source = Script::new([
            Step::Data(b"a=1\nb=2\n"),
            Step::Eof,
            Step::Eof,
            Step::Data(b"c=3\n"),
        ]);
        let mut reader = FramedRead::new(source, Lines::default());
        for expected in [b"a=1\n", b"b=2\n"] {
            assert_eq!(reader.decode_view().unwrap().unwrap(), expected);
            assert_eq!(reader.reader().reads, 1);
        }
        assert!(reader.decode_view().unwrap().is_none());
        assert_eq!(reader.reader().reads, 3);
        assert_eq!(reader.decode_view().unwrap().unwrap(), b"c=3\n");
        assert!(reader.decode_view().unwrap().is_none());
    }

    #[test]
    fn explicit_retries_after_interrupted_and_other_read_errors_preserve_state() {
        let source = Script::new([
            Step::Error(ErrorKind::Interrupted),
            Step::Data(b"a="),
            Step::Error(ErrorKind::Interrupted),
            Step::Error(ErrorKind::WouldBlock),
            Step::Data(b"b\n"),
        ]);
        let mut reader = FramedRead::new(source, Lines::default());
        for expected in [
            ErrorKind::Interrupted,
            ErrorKind::Interrupted,
            ErrorKind::WouldBlock,
        ] {
            assert_eq!(reader.decode_view().unwrap_err().kind(), expected);
        }
        assert_eq!(reader.reader().reads, 4);
        assert_eq!(reader.decode_view().unwrap().unwrap(), b"a=b\n");
    }

    #[test]
    fn failed_reads_preserve_input_across_decode_modes() {
        for capacity in [0, 8] {
            for prefix in [&b""[..], &b"key="[..]] {
                for overreport in [false, true] {
                    for mode in 0..3 {
                        for retry_mode in 0..3 {
                            let failure = if overreport {
                                Step::Overreport
                            } else {
                                Step::DirtyError(ErrorKind::Other)
                            };
                            let mut steps = Vec::new();
                            if !prefix.is_empty() {
                                steps.push(Step::Data(prefix));
                            }
                            steps.extend([failure, Step::Data(b"ok\n")]);
                            let mut reader = FramedRead::with_capacity(
                                Script::new(steps),
                                Lines::default(),
                                capacity,
                            );
                            let result = catch_unwind(AssertUnwindSafe(|| match mode {
                                0 => reader.decode_bytes().map(|_| ()),
                                1 => reader.decode_view().map(|_| ()),
                                _ => reader.decode_owned().map(|_| ()),
                            }));
                            if overreport {
                                assert!(result.is_err());
                            } else {
                                assert_eq!(result.unwrap().unwrap_err().kind(), ErrorKind::Other);
                            }
                            assert_eq!(reader.buffer().to_vec(), prefix);
                            let expected = [prefix, b"ok\n"].concat();
                            match retry_mode {
                                0 => assert_eq!(
                                    &reader.decode_bytes().unwrap().unwrap()[..],
                                    expected
                                ),
                                1 => assert_eq!(reader.decode_view().unwrap().unwrap(), expected),
                                _ => assert_eq!(reader.decode_owned().unwrap().unwrap(), expected),
                            }
                            assert!(reader.decode_view().unwrap().is_none());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn every_input_split_roundtrips() {
        let bytes = b"\xff=\0\n=\na=b=c\n";
        for split in 0..=bytes.len() {
            let mut steps = Vec::new();
            if split != 0 {
                steps.push(Step::Data(&bytes[..split]));
            }
            if split != bytes.len() {
                steps.push(Step::Data(&bytes[split..]));
            }
            let mut reader = FramedRead::with_capacity(Script::new(steps), Lines::default(), 1);
            let mut decoded = Vec::new();
            while let Some(item) = reader.decode_view().unwrap() {
                decoded.extend_from_slice(item);
            }
            assert_eq!(decoded, bytes);
        }
    }

    #[test]
    fn short_reads_and_large_items_roundtrip_in_both_modes() {
        for (payload_len, limit) in [(6, 1), (24, 3), (20_000, 8192), (20_000, 30_000)] {
            // Bytewise reads use small frames because pending input doubles the
            // read capacity. Large frames exercise refills without thousands of
            // deliberate short reads and exponential test allocations.
            let items = [
                vec![b'\n'],
                vec![0, 255, 13, b'\n'],
                (0..=255)
                    .filter(|&b| b != b'\n')
                    .cycle()
                    .take(payload_len)
                    .chain(*b"\n")
                    .collect(),
                vec![b'\n'],
            ];
            let wire = items.concat();
            for capacity in [1, 4, 7, 8192] {
                let source = || ShortReads {
                    source: Cursor::new(&wire),
                    limit,
                };
                let mut reader = FramedRead::with_capacity(source(), Lines::default(), capacity);
                for expected in &items {
                    assert_eq!(reader.decode_view().unwrap().unwrap(), expected.as_slice());
                }
                assert!(reader.decode_view().unwrap().is_none());
                let mut reader = FramedRead::with_capacity(source(), Lines::default(), capacity);
                let mut kept = Vec::new();
                for _ in &items {
                    kept.push(reader.decode_bytes().unwrap().unwrap());
                }
                drop(reader);
                for (actual, expected) in kept.iter().zip(&items) {
                    assert_eq!(actual.as_ref(), expected);
                }
            }
        }
    }

    #[test]
    fn dropping_a_reader_preserves_the_borrowed_sources_read_ahead_position() {
        let mut source = Cursor::new(b"a=1\nb=2\n");
        let mut reader = FramedRead::new(&mut source, Lines::default());
        assert_eq!(reader.decode_view().unwrap().unwrap(), b"a=1\n");
        // Dropping the reader discards buffered input without rewinding the
        // source.
        drop(reader);
        assert_eq!(source.position(), 8);
        let mut remaining = Vec::new();
        source.read_to_end(&mut remaining).unwrap();
        assert!(remaining.is_empty());
    }

    #[test]
    fn owned_views_survive_subsequent_reads_and_reader_drop() {
        struct OwnedView(Lines);

        impl Framer for OwnedView {
            type Error = io::Error;

            fn frame(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
                self.0.frame(src)
            }

            fn frame_eof(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
                self.0.frame_eof(src)
            }
        }

        impl ViewDecoder for OwnedView {
            type Item<'a> = Vec<u8>;

            fn decode_view(&mut self, frame: &[u8]) -> io::Result<Vec<u8>> {
                Ok(frame.to_vec())
            }
        }

        for capacity in [1, 8] {
            let mut reader = FramedRead::with_capacity(
                &b"\xff\0\nsecond\ntail"[..],
                OwnedView(Lines::default()),
                capacity,
            );
            let first = reader.decode_view().unwrap().unwrap();
            let second = reader.decode_view().unwrap().unwrap();
            let tail = reader.decode_view().unwrap().unwrap();
            assert!(reader.decode_view().unwrap().is_none());
            drop(reader);
            assert_eq!(first, b"\xff\0\n");
            assert_eq!(second, b"second\n");
            assert_eq!(tail, b"tail");
        }
    }

    #[test]
    fn borrowing_owned_and_copied_items_can_alternate() {
        let expected: Vec<Vec<u8>> = (0..25)
            .map(|i| {
                let mut item = vec![128 + i as u8; i % 5 * 101];
                item.push(b'\n');
                item
            })
            .collect();
        let wire = expected.concat();
        for capacity in [1, 7, 64] {
            let mut reader = FramedRead::with_capacity(&wire[..], Lines::default(), capacity);
            let mut kept_bytes = Vec::new();
            let mut kept_vecs = Vec::new();
            for (i, payload) in expected.iter().enumerate() {
                match i % 3 {
                    0 => {
                        assert_eq!(reader.decode_view().unwrap().unwrap(), payload.as_slice());
                    }
                    1 => kept_bytes.push((i, reader.decode_bytes().unwrap().unwrap())),
                    _ => kept_vecs.push((i, reader.decode_owned().unwrap().unwrap())),
                }
            }
            assert!(reader.decode_bytes().unwrap().is_none());
            drop(reader);
            for (i, item) in kept_bytes {
                assert_eq!(item.as_ref(), expected[i]);
            }
            for (i, item) in kept_vecs {
                assert_eq!(item, expected[i]);
            }
        }
    }

    #[test]
    fn eof_checks_internal_state_and_can_yield_multiple_zero_byte_items() {
        struct Closing {
            remaining: u8,
            eof_calls: usize,
        }
        impl Framer for Closing {
            type Error = io::Error;
            fn frame(&mut self, _: Segments<&[u8]>) -> Result<Frame, Self::Error> {
                Ok(Frame::need_more(0))
            }
            fn frame_eof(&mut self, _: Segments<&[u8]>) -> Result<Frame, Self::Error> {
                self.eof_calls += 1;
                Ok(if self.remaining > 0 {
                    Frame::ready_split(..0, 0)
                } else {
                    Frame::need_more(0)
                })
            }
        }
        impl ViewDecoder for Closing {
            type Item<'a> = u8;
            fn decode_view<'a>(&mut self, _: &'a [u8]) -> Result<Self::Item<'a>, Self::Error> {
                self.remaining -= 1;
                Ok(self.remaining)
            }
        }
        let mut reader = FramedRead::new(
            io::empty(),
            Closing {
                remaining: 2,
                eof_calls: 0,
            },
        );
        assert_eq!(reader.decode_view().unwrap(), Some(1));
        assert_eq!(reader.decode_view().unwrap(), Some(0));
        assert!(reader.decode_view().unwrap().is_none());
        assert!(reader.decode_view().unwrap().is_none());
        assert_eq!(reader.decoder().eof_calls, 4);
        struct Pending;
        impl Framer for Pending {
            type Error = io::Error;
            fn frame(&mut self, _: Segments<&[u8]>) -> Result<Frame, Self::Error> {
                Ok(Frame::need_more(0))
            }
            fn frame_eof(&mut self, _: Segments<&[u8]>) -> Result<Frame, Self::Error> {
                Err(io::Error::new(ErrorKind::UnexpectedEof, "unfinished state"))
            }
        }
        impl ViewDecoder for Pending {
            type Item<'a> = ();
            fn decode_view<'a>(&mut self, _: &'a [u8]) -> Result<Self::Item<'a>, Self::Error> {
                unreachable!()
            }
        }
        let mut reader = FramedRead::new(io::empty(), Pending);
        assert!(matches!(
            reader.decode_view(),
            Err(error) if error.kind() == ErrorKind::UnexpectedEof
        ));
    }

    // A format can retain quote state while framing. Transformations cannot be
    // represented solely by offsets into raw input: this test Item owns the
    // result.
    #[derive(Default)]
    struct Quoted {
        scanned: usize,
        quoted: bool,
        escaped: bool,
    }
    struct QuotedItem<B> {
        raw: B,
        end: usize,
        unescaped: Box<[u8]>,
    }
    impl Framer for Quoted {
        type Error = io::Error;
        fn frame(&mut self, src: Segments<&[u8]>) -> Result<Frame, Self::Error> {
            for (i, &b) in src.iter().enumerate().skip(self.scanned) {
                if self.escaped {
                    self.escaped = false;
                } else {
                    match b {
                        b'\\' => self.escaped = true,
                        b'"' => self.quoted = !self.quoted,
                        b'\n' if !self.quoted => return Ok(Frame::ready_split(..i + 1, 0)),
                        _ => {}
                    }
                }
            }
            self.scanned = src.len();
            Ok(Frame::need_more(0))
        }
        fn frame_eof(&mut self, src: Segments<&[u8]>) -> Result<Frame, Self::Error> {
            let frame = self.frame(src)?;
            if frame != Frame::need_more(0) {
                return Ok(frame);
            }
            if self.quoted || self.escaped {
                return Err(io::Error::new(
                    ErrorKind::UnexpectedEof,
                    "truncated quote/escape",
                ));
            }
            Ok(if src.is_empty() {
                Frame::need_more(0)
            } else {
                Frame::ready_split(..src.len(), 0)
            })
        }
    }
    impl BytesDecoder for Quoted {
        type Item = QuotedItem<Bytes>;
        fn decode_bytes(&mut self, src: BytesMut) -> Result<Self::Item, Self::Error> {
            self.parse(src.freeze())
        }
    }
    impl ViewDecoder for Quoted {
        type Item<'a> = QuotedItem<&'a [u8]>;
        fn decode_view<'a>(&mut self, src: &'a [u8]) -> Result<Self::Item<'a>, Self::Error> {
            self.parse(src)
        }
    }
    impl Quoted {
        fn parse<B: AsRef<[u8]>>(&mut self, src: B) -> io::Result<QuotedItem<B>> {
            let bytes = src.as_ref();
            let end = bytes.len() - usize::from(bytes.last() == Some(&b'\n'));
            let mut escape = false;
            let unescaped = bytes[..end]
                .iter()
                .copied()
                .filter(|&b| {
                    if escape {
                        escape = false;
                        true
                    } else {
                        match b {
                            b'\\' => {
                                escape = true;
                                false
                            }
                            b'"' => false,
                            _ => true,
                        }
                    }
                })
                .collect();
            self.scanned = 0;
            self.quoted = false;
            self.escaped = false;
            Ok(QuotedItem {
                raw: src,
                end,
                unescaped,
            })
        }
    }
    #[test]
    fn quote_state_controls_boundaries_and_conversion_storage_has_item_lifetime() {
        let bytes = b"\"a\nb\"\n\"c\\\"d\"\n";
        let mut reader = FramedRead::with_capacity(
            ShortReads {
                source: &bytes[..],
                limit: 1,
            },
            Quoted::default(),
            1,
        );
        let first = reader.decode_view().unwrap().unwrap();
        assert_eq!(&first.raw[..first.end], b"\"a\nb\"");
        assert_eq!(&*first.unescaped, b"a\nb");
        assert_ne!(first.raw.as_ptr(), first.unescaped.as_ptr());
        assert_eq!(&*reader.decode_view().unwrap().unwrap().unescaped, b"c\"d");
        assert!(reader.decode_view().unwrap().is_none());
        let mut reader = FramedRead::new(&bytes[..], Quoted::default());
        let first = reader.decode_bytes().unwrap().unwrap();
        let second = reader.decode_bytes().unwrap().unwrap();
        drop(reader);
        assert_eq!(&*first.unescaped, b"a\nb");
        assert_eq!(&*second.unescaped, b"c\"d");
        for input in [&b"\"open"[..], b"escape\\"] {
            let mut reader = FramedRead::new(input, Quoted::default());
            assert!(matches!(
                reader.decode_view(),
                Err(error) if error.kind() == ErrorKind::UnexpectedEof
            ));
        }
    }

    // Script framing decisions independently of the reader's implementation.
    // Record the slices seen by framing and decoding to check both coordinate
    // systems.
    #[derive(Default)]
    struct FrameScript {
        frames: VecDeque<io::Result<Frame>>,
        eof_frames: VecDeque<io::Result<Frame>>,
        inputs: Vec<(bool, Vec<u8>)>,
        decoded: Vec<Vec<u8>>,
        reject_item: bool,
    }
    impl FrameScript {
        fn new(frames: impl IntoIterator<Item = Frame>) -> Self {
            Self {
                frames: frames.into_iter().map(Ok).collect(),
                ..Self::default()
            }
        }
        fn inspect(&mut self, bytes: &[u8]) -> io::Result<()> {
            self.decoded.push(bytes.to_vec());
            if self.reject_item {
                Err(io::Error::new(ErrorKind::InvalidData, "rejected Item"))
            } else {
                Ok(())
            }
        }
    }
    impl Framer for FrameScript {
        type Error = io::Error;
        fn frame(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
            self.inputs.push((false, src.to_vec()));
            self.frames.pop_front().unwrap_or(Ok(Frame::need_more(0)))
        }
        fn frame_eof(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
            self.inputs.push((true, src.to_vec()));
            self.eof_frames
                .pop_front()
                .unwrap_or(Ok(Frame::need_more(0)))
        }
    }
    impl BytesDecoder for FrameScript {
        type Item = Bytes;
        fn decode_bytes(&mut self, frame: BytesMut) -> io::Result<Bytes> {
            self.inspect(&frame)?;
            Ok(frame.freeze())
        }
    }
    impl ViewDecoder for FrameScript {
        type Item<'a> = &'a [u8];
        fn decode_view<'a>(&mut self, frame: &'a [u8]) -> io::Result<&'a [u8]> {
            self.inspect(frame)?;
            Ok(frame)
        }
    }

    impl OwnedDecoder for FrameScript {
        type Item = Vec<u8>;

        fn decode_owned(&mut self, frame: Segments<&[u8]>) -> io::Result<Vec<u8>> {
            self.inspect(&frame.to_vec())?;
            Ok(frame.to_vec())
        }
    }

    #[test]
    fn bound_selections_and_consumption_agree_in_all_decode_modes() {
        use std::ops::Bound::{Excluded, Included, Unbounded};

        let input = b"abcdef";
        for bounds in [
            (Unbounded, Included(0)),
            (Included(1), Excluded(3)),
            (Excluded(0), Included(2)),
            (Included(4), Unbounded),
            (Unbounded, Unbounded),
            (Included(2), Excluded(2)),
        ] {
            let expected = &input[bounds];
            let end = expected.as_ptr().addr() - input.as_ptr().addr() + expected.len();
            for copy in [false, true] {
                let consume = if copy { 1 } else { end };
                let decision = if copy {
                    Frame::ready_copy(bounds, consume)
                } else {
                    Frame::ready_split(bounds, 0)
                };
                for mode in 0..3 {
                    let codec =
                        FrameScript::new([Frame::need_more(0), Frame::skip(1), decision.clone()]);
                    let mut reader =
                        FramedRead::with_capacity(&b"#abcdef"[..], codec, input.len() + 1);
                    let actual = match mode {
                        0 => reader.decode_bytes().unwrap().unwrap().to_vec(),
                        1 => reader.decode_view().unwrap().unwrap().to_vec(),
                        _ => reader.decode_owned().unwrap().unwrap(),
                    };
                    assert_eq!(actual, expected);
                    assert_eq!(reader.buffer().to_vec(), &input[consume..]);
                }
            }
        }
        for bounds in [
            (Excluded(usize::MAX), Unbounded),
            (Unbounded, Included(usize::MAX)),
        ] {
            for mode in 0..3 {
                let codec = FrameScript::new([Frame::need_more(0), Frame::ready_split(bounds, 0)]);
                let mut reader = FramedRead::with_capacity(&input[..], codec, input.len());
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match mode {
                        0 => reader.decode_bytes().map(|_| ()),
                        1 => reader.decode_view().map(|_| ()),
                        _ => reader.decode_owned().map(|_| ()),
                    }))
                    .is_err()
                );
            }
        }
    }

    #[test]
    fn default_eof_can_resume_reading_after_none_and_source_errors() {
        struct Pairs;
        impl Framer for Pairs {
            type Error = io::Error;

            fn frame(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
                Ok(if src.len() < 2 {
                    Frame::need_more(0)
                } else {
                    Frame::ready_split(..2, 0)
                })
            }
        }
        impl BytesDecoder for Pairs {
            type Item = BytesMut;

            fn decode_bytes(&mut self, frame: BytesMut) -> io::Result<BytesMut> {
                Ok(frame)
            }
        }
        impl ViewDecoder for Pairs {
            type Item<'a> = &'a [u8];

            fn decode_view<'a>(&mut self, frame: &'a [u8]) -> io::Result<&'a [u8]> {
                Ok(frame)
            }
        }

        impl OwnedDecoder for Pairs {
            type Item = Vec<u8>;

            fn decode_owned(&mut self, frame: Segments<&[u8]>) -> io::Result<Vec<u8>> {
                Ok(frame.to_vec())
            }
        }

        for mode in 0..3 {
            let source = Script::new([
                Step::Eof,
                Step::Eof,
                Step::Error(ErrorKind::WouldBlock),
                Step::Data(b"ab"),
                Step::Eof,
                Step::Eof,
                Step::Data(b"cd"),
                Step::Eof,
                Step::Eof,
                Step::Data(b"e"),
                Step::Eof,
            ]);
            let mut reader = FramedRead::new(source, Pairs);
            for (expected, reads) in [
                (Ok(None), 2),
                (Err(ErrorKind::WouldBlock), 3),
                (Ok(Some(&b"ab"[..])), 4),
                (Ok(None), 6),
                (Ok(Some(&b"cd"[..])), 7),
                (Ok(None), 9),
            ] {
                let result = match mode {
                    0 => reader
                        .decode_bytes()
                        .map(|item| item.map(|bytes| bytes.to_vec())),
                    1 => reader.decode_view().map(|item| item.map(<[u8]>::to_vec)),
                    _ => reader.decode_owned(),
                };
                assert_eq!(
                    result
                        .as_ref()
                        .map(|item| item.as_deref())
                        .map_err(|e| e.kind()),
                    expected
                );
                assert_eq!(reader.reader().reads, reads);
            }
            // The final single byte is a truncated pair, not successful
            // completion.
            let result = match mode {
                0 => reader.decode_bytes().map(|_| ()),
                1 => reader.decode_view().map(|_| ()),
                _ => reader.decode_owned().map(|_| ()),
            };
            assert_eq!(result.unwrap_err().kind(), ErrorKind::UnexpectedEof);
            assert_eq!(reader.reader().reads, 11);
        }
    }

    #[test]
    fn eof_preparation_provides_space_without_growing_existing_capacity() {
        // Cover no allocation, uninitialized capacity, and a consumed full
        // buffer.
        for (capacity, consumed) in [(0, 0), (4, 0), (4, 4)] {
            let codec = FrameScript::new([Frame::ready_split(..1, 0)]);
            let mut reader = FramedRead::with_capacity(&b"x"[..], codec, capacity);
            if consumed != 0 {
                reader.buffer.write_buf(&mut &b"abcd"[..]).unwrap();
                reader.buffer.consume(consumed);
            }
            reader.eof = true;

            assert_eq!(reader.decode_view().unwrap().unwrap(), b"x");
            assert!(!reader.eof);
            assert!(reader.reader().is_empty());
        }
        let mut reader = FramedRead::with_capacity(&b"tail"[..], Lines::default(), 8);
        let retained = reader.decode_bytes().unwrap().unwrap();
        let (_, count) = allocations(|| {
            for _ in 0..3 {
                assert!(reader.decode_view().unwrap().is_none());
                assert!(reader.decode_bytes().unwrap().is_none());
                assert!(reader.decode_owned().unwrap().is_none());
            }
        });
        assert_eq!(count, 0);
        assert_eq!(&retained[..], b"tail");
    }

    #[test]
    fn need_more_at_eof_keeps_partial_input_until_the_source_resumes() {
        for owned in [false, true] {
            let mut codec = FrameScript::new([
                Frame::need_more(0),
                Frame::need_more(0),
                Frame::ready_split(..2, 0),
            ]);
            codec.eof_frames = std::iter::repeat_n(Frame::need_more(0), 2)
                .map(Ok)
                .collect();
            let source = Script::new([Step::Data(b"a"), Step::Eof, Step::Eof, Step::Data(b"b")]);
            let mut reader = FramedRead::new(source, codec);
            if owned {
                assert!(reader.decode_bytes().unwrap().is_none());
                assert_eq!(reader.reader().reads, 3);
                assert_eq!(&reader.decode_bytes().unwrap().unwrap()[..], b"ab");
            } else {
                assert!(reader.decode_view().unwrap().is_none());
                assert_eq!(reader.reader().reads, 3);
                assert_eq!(reader.decode_view().unwrap().unwrap(), b"ab");
            }
            assert_eq!(reader.reader().reads, 4);
            assert_eq!(reader.decoder().decoded, [b"ab".to_vec()]);
        }
    }

    #[test]
    fn need_more_consumes_before_reading_in_all_decode_modes() {
        for (consume, expected) in [(0, &b"#abc"[..]), (1, &b"abc"[..]), (2, &b"bc"[..])] {
            for mode in 0..3 {
                let codec = FrameScript::new([
                    Frame::need_more(0),
                    Frame::need_more(consume),
                    Frame::ready_split(.., 0),
                ]);
                let source = Script::new([Step::Data(b"#a"), Step::Data(b"bc")]);
                let mut reader = FramedRead::with_capacity(source, codec, 16);
                let item = match mode {
                    0 => reader.decode_bytes().unwrap().unwrap().to_vec(),
                    1 => reader.decode_view().unwrap().unwrap().to_vec(),
                    _ => reader.decode_owned().unwrap().unwrap(),
                };
                assert_eq!(item, expected);
                assert_eq!(reader.reader().reads, 2);
                assert_eq!(
                    reader.decoder().inputs,
                    [
                        (false, Vec::new()),
                        (false, b"#a".to_vec()),
                        (false, expected.to_vec()),
                    ]
                );
                assert_eq!(reader.decoder().decoded, [expected.to_vec()]);
            }
        }
    }

    #[test]
    fn need_more_consumption_survives_read_errors_and_panics() {
        for failure in 0..5 {
            for mode in 0..3 {
                let step = match failure {
                    0 => Step::Error(ErrorKind::Interrupted),
                    1 => Step::Error(ErrorKind::WouldBlock),
                    2 => Step::DirtyError(ErrorKind::Other),
                    3 => Step::Overreport,
                    _ => Step::Panic,
                };
                let codec = FrameScript::new([
                    Frame::need_more(0),
                    Frame::need_more(1),
                    Frame::ready_split(.., 0),
                    Frame::need_more(0),
                    Frame::ready_split(.., 0),
                ]);
                let source = Script::new([Step::Data(b"#a"), step, Step::Data(b"b")]);
                let mut reader = FramedRead::with_capacity(source, codec, 16);
                let result = catch_unwind(AssertUnwindSafe(|| match mode {
                    0 => reader.decode_bytes().map(|_| ()),
                    1 => reader.decode_view().map(|_| ()),
                    _ => reader.decode_owned().map(|_| ()),
                }));
                if failure < 3 {
                    assert_eq!(
                        result.unwrap().unwrap_err().kind(),
                        [
                            ErrorKind::Interrupted,
                            ErrorKind::WouldBlock,
                            ErrorKind::Other,
                        ][failure]
                    );
                } else {
                    assert!(result.is_err());
                }
                assert_eq!(reader.buffer().to_vec(), b"a");
                assert_eq!(reader.reader().reads, 2);
                assert!(reader.decoder().decoded.is_empty());
                assert_eq!(reader.decode_view().unwrap().unwrap(), b"a");
                assert_eq!(reader.reader().reads, 2);
                assert_eq!(reader.decode_owned().unwrap().unwrap(), b"b");
                assert_eq!(reader.reader().reads, 3);
            }
        }
    }

    #[test]
    fn need_more_consumes_at_eof_and_preserves_the_remainder_for_resumption() {
        for (consume, remaining, expected) in [
            (0, &b"#a"[..], &b"#ab"[..]),
            (1, &b"a"[..], &b"ab"[..]),
            (2, &b""[..], &b"b"[..]),
        ] {
            for mode in 0..3 {
                let mut codec = FrameScript::new([
                    Frame::need_more(0),
                    Frame::need_more(0),
                    Frame::ready_split(.., 0),
                ]);
                codec.eof_frames = [Frame::need_more(consume), Frame::need_more(0)]
                    .into_iter()
                    .map(Ok)
                    .collect();
                let source =
                    Script::new([Step::Data(b"#a"), Step::Eof, Step::Eof, Step::Data(b"b")]);
                let mut reader = FramedRead::with_capacity(source, codec, 16);
                let no_item = match mode {
                    0 => reader.decode_bytes().unwrap().is_none(),
                    1 => reader.decode_view().unwrap().is_none(),
                    _ => reader.decode_owned().unwrap().is_none(),
                };
                assert!(no_item);
                assert!(reader.eof);
                assert_eq!(reader.buffer().to_vec(), remaining);
                assert_eq!(reader.reader().reads, 3);
                let item = match mode {
                    0 => reader.decode_bytes().unwrap().unwrap().to_vec(),
                    1 => reader.decode_view().unwrap().unwrap().to_vec(),
                    _ => reader.decode_owned().unwrap().unwrap(),
                };
                assert_eq!(item, expected);
                assert!(!reader.eof);
                assert_eq!(reader.reader().reads, 4);
                assert_eq!(reader.decoder().inputs[3], (true, remaining.to_vec()));
            }
        }
    }

    #[test]
    fn excessive_skip_and_need_more_panic_before_consuming_or_reading() {
        for eof in [false, true] {
            for input in [&b""[..], &b"abc"[..]] {
                for amount in [input.len() + 1, usize::MAX] {
                    for action in [Frame::skip(amount), Frame::need_more(amount)] {
                        for mode in 0..3 {
                            let mut codec = FrameScript::default();
                            let decisions = [action.clone(), Frame::ready_split(.., 0)]
                                .into_iter()
                                .map(Ok);
                            if eof {
                                codec.eof_frames.extend(decisions);
                            } else {
                                codec.frames.extend(decisions);
                            }
                            let mut reader = FramedRead::with_capacity(
                                Script::new([Step::Data(b"next")]),
                                codec,
                                16,
                            );
                            reader
                                .buffer
                                .write_buf(&mut io::repeat(0).take(16))
                                .unwrap();
                            reader.buffer.consume(16);
                            let initial = [b"old:", input].concat();
                            reader.buffer.write_buf(&mut &initial[..]).unwrap();
                            reader.buffer.consume(4);
                            reader.eof = eof;

                            let result = catch_unwind(AssertUnwindSafe(|| match mode {
                                0 => reader.decode_bytes().map(|_| ()),
                                1 => reader.decode_view().map(|_| ()),
                                _ => reader.decode_owned().map(|_| ()),
                            }));
                            assert!(result.is_err());
                            assert_eq!(reader.buffer().to_vec(), input);
                            assert_eq!(reader.eof, eof);
                            assert_eq!(reader.reader().reads, 0);
                            assert!(reader.decoder().decoded.is_empty());
                            assert_eq!(reader.decode_view().unwrap().unwrap(), input);
                            assert_eq!(reader.reader().reads, 0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn frame_ranges_and_consumption_preserve_suffix_and_borrow_original_addresses() {
        for owned in [false, true] {
            let mut codec = FrameScript::new([
                Frame::need_more(0),
                Frame::skip(2),
                Frame::ready_split(1..4, 2),
                Frame::ready_split(1..3, 2),
                Frame::need_more(0),
            ]);
            codec.eof_frames.push_back(Ok(Frame::skip(4)));
            let source = Script::new([Step::Data(b"##<abc>!<de>!tail")]);
            let mut reader = FramedRead::with_capacity(source, codec, 32);
            let mut kept = Vec::new();
            for (expected, offset) in [(&b"abc"[..], 3), (b"de", 9)] {
                let address;
                if owned {
                    let item = reader.decode_bytes().unwrap().unwrap();
                    assert_eq!(&item[..], expected);
                    address = item.as_ptr() as usize;
                    kept.push(item);
                } else {
                    let item = reader.decode_view().unwrap().unwrap();
                    assert_eq!(item, expected);
                    address = item.as_ptr() as usize;
                }
                assert_eq!(address, reader.reader().destinations[0] + offset);
                assert_eq!(reader.reader().reads, 1);
            }
            assert!(reader.decode_view().unwrap().is_none());
            assert!(reader.decode_bytes().unwrap().is_none());
            assert_eq!(reader.decoder().decoded, [b"abc".to_vec(), b"de".to_vec()]);
            let eof_inputs: Vec<_> = reader
                .decoder()
                .inputs
                .iter()
                .filter(|(eof, _)| *eof)
                .map(|(_, bytes)| &bytes[..])
                .collect();
            assert_eq!(eof_inputs, [&b"tail"[..], b"", b""]);
            drop(reader);
            if owned {
                assert_eq!(
                    kept,
                    [Bytes::from_static(b"abc"), Bytes::from_static(b"de")]
                );
            }
        }
    }

    #[test]
    fn need_more_after_skipping_preserves_scan_phase_across_read_error_and_compact() {
        let codec = FrameScript::new([
            Frame::need_more(0),
            Frame::skip(2),
            Frame::need_more(0),
            Frame::need_more(0), // Reframing after Interrupted.
            Frame::need_more(0), // Reframing after WouldBlock.
            Frame::ready_split(1..3, 1),
        ]);
        let source = Script::new([
            Step::Data(b"##<a"),
            Step::Error(ErrorKind::Interrupted),
            Step::Error(ErrorKind::WouldBlock),
            Step::Data(b"b>"),
        ]);
        let mut reader = FramedRead::with_capacity(source, codec, 4);
        assert_eq!(
            reader.decode_view().unwrap_err().kind(),
            ErrorKind::Interrupted
        );
        assert_eq!(
            reader.decode_view().unwrap_err().kind(),
            ErrorKind::WouldBlock
        );
        assert_eq!(reader.decoder().inputs.len(), 4);
        let item = reader.decode_bytes().unwrap().unwrap();
        assert_eq!(&item[..], b"ab");
        let inputs: Vec<_> = reader
            .decoder()
            .inputs
            .iter()
            .map(|(_, bytes)| &bytes[..])
            .collect();
        assert_eq!(inputs, [&b""[..], b"##<a", b"<a", b"<a", b"<a", b"<ab>"]);
        assert_eq!(reader.decoder().decoded, [b"ab".to_vec()]);
    }

    #[test]
    fn eof_skipping_can_precede_ready_empty_items_and_source_resumption() {
        let mut codec = FrameScript::new([
            Frame::need_more(0),
            Frame::need_more(0),
            Frame::ready_split(..1, 0),
        ]);
        codec.eof_frames = [
            Frame::skip(0),
            Frame::skip(1),
            Frame::ready_split(1..2, 1),
            Frame::ready_split(0..0, 0),
            Frame::need_more(0),
        ]
        .into_iter()
        .map(Ok)
        .collect();
        let source = Script::new([Step::Data(b"#<x>"), Step::Eof, Step::Eof, Step::Data(b"y")]);
        let mut reader = FramedRead::new(source, codec);
        let x = reader.decode_bytes().unwrap().unwrap();
        assert_eq!(&x[..], b"x");
        assert!(reader.decode_bytes().unwrap().unwrap().is_empty());
        assert!(reader.decode_view().unwrap().is_none());
        assert_eq!(reader.decode_view().unwrap().unwrap(), b"y");
        assert_eq!(
            reader
                .decoder()
                .inputs
                .iter()
                .filter(|(eof, _)| *eof)
                .count(),
            6
        );
        drop(reader);
        assert_eq!(&x[..], b"x");
    }

    #[test]
    fn invalid_selection_ranges_panic_before_decoding_including_at_eof() {
        for eof in [false, true] {
            for owned in [false, true] {
                for action in [
                    Frame::ready_split(..2, 0),
                    Frame::ready_split(std::ops::Range { start: 1, end: 0 }, 0),
                    Frame::ready_copy(..2, 0),
                    Frame::ready_copy(std::ops::Range { start: 1, end: 0 }, 0),
                ] {
                    let mut codec = FrameScript::new([Frame::need_more(0)]);
                    if eof {
                        codec.frames.push_back(Ok(Frame::need_more(0)));
                        codec.eof_frames.push_back(Ok(action));
                    } else {
                        codec.frames.push_back(Ok(action));
                    }
                    let mut reader = FramedRead::with_capacity(&b"x"[..], codec, 8);
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        if owned {
                            reader.decode_bytes().map(|_| ())
                        } else {
                            reader.decode_view().map(|_| ())
                        }
                    }));
                    assert!(result.is_err());
                    assert!(reader.decoder().decoded.is_empty());
                }
            }
        }
    }

    #[test]
    fn overflowing_split_consumption_panics_without_decoding_or_consuming() {
        fn check(source: impl Read, end: usize, addition: usize) {
            let codec = FrameScript::new([
                Frame::need_more(0),
                Frame::ready_split(..end, addition),
                Frame::ready_split(.., 0),
            ]);
            let mut reader = FramedRead::new(source, codec);
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = reader.decode_view().unwrap();
            }));
            assert!(outcome.is_err(), "overflowing consumption must be rejected");
            assert!(reader.decoder().decoded.is_empty());
            assert_eq!(reader.decode_view().unwrap().unwrap(), b"abc");
        }
        for (end, addition) in [(1, usize::MAX), (2, usize::MAX), (3, usize::MAX - 1)] {
            check(Script::new([Step::Data(b"abc")]), end, addition);
            check(Cursor::new(&b"abc"[..]), end, addition);
            check(Cursor::new(b"abc".to_vec()), end, addition);
            check(
                Cursor::new(b"abc".to_vec().into_boxed_slice()),
                end,
                addition,
            );
        }
    }

    #[test]
    fn zero_byte_skipping_and_borrowed_lookahead_keep_unconsumed_overlap() {
        let codec = FrameScript::new([
            Frame::skip(0),
            Frame::need_more(0),
            Frame::ready_copy(1..4, 2),
            Frame::skip(0),
            Frame::ready_split(0..2, 1),
        ]);
        let source = Script::new([Step::Data(b"<abc>")]);
        let mut reader = FramedRead::new(source, codec);
        let first = reader.decode_view().unwrap().unwrap();
        assert_eq!(first, b"abc");
        let address = first.as_ptr() as usize;
        let second = reader.decode_view().unwrap().unwrap();
        assert_eq!(second, b"bc");
        assert_eq!(second.as_ptr() as usize, address + 1);
        assert_eq!(reader.reader().reads, 1);
        assert_eq!(reader.decoder().decoded, [b"abc".to_vec(), b"bc".to_vec()]);
    }

    #[test]
    fn ready_ranges_and_additions_transfer_without_copying() {
        let input = b"a\0\xffbcdef";
        for start in 0..=input.len() {
            for end in start..=input.len() {
                for addition in 0..=input.len() - end {
                    let consume = end + addition;
                    let codec = FrameScript::new([
                        Frame::need_more(0),
                        Frame::ready_split(start..end, addition),
                        Frame::ready_split(..input.len() - consume, 0),
                    ]);
                    let source = Script::new([Step::Data(input)]);
                    let mut reader = FramedRead::with_capacity(source, codec, 16);
                    let first = reader.decode_bytes().unwrap().unwrap();
                    assert_eq!(&first[..], &input[start..end]);
                    if !first.is_empty() {
                        assert_eq!(
                            first.as_ptr() as usize,
                            reader.reader().destinations[0] + start
                        );
                    }
                    let suffix = reader.decode_bytes().unwrap().unwrap();
                    assert_eq!(reader.reader().reads, 1);
                    drop(reader);
                    assert_eq!(&first[..], &input[start..end]);
                    assert_eq!(&suffix[..], &input[consume..]);
                }
            }
        }
    }

    #[test]
    fn ready_copy_copies_only_owned_items_for_every_valid_range_and_consumption() {
        let input = b"a\0\xffbcdef";
        for owned in [false, true] {
            for start in 0..=input.len() {
                for end in start..=input.len() {
                    for consume in 0..=input.len() {
                        let codec = FrameScript::new([
                            Frame::need_more(0),
                            Frame::ready_copy(start..end, consume),
                            Frame::ready_split(..input.len() - consume, 0),
                        ]);
                        let source = Script::new([Step::Data(input)]);
                        let mut reader = FramedRead::with_capacity(source, codec, 16);
                        let mut held = None;
                        let address = if owned {
                            let item = reader.decode_bytes().unwrap().unwrap();
                            assert_eq!(&item[..], &input[start..end]);
                            let address = item.as_ptr() as usize;
                            held = Some(item);
                            address
                        } else {
                            let item = reader.decode_view().unwrap().unwrap();
                            assert_eq!(item, &input[start..end]);
                            item.as_ptr() as usize
                        };
                        let original = reader.reader().destinations[0];
                        if start != end {
                            assert_eq!(address == original + start, !owned);
                        }
                        let suffix = reader.decode_bytes().unwrap().unwrap();
                        if !suffix.is_empty() {
                            assert_eq!(suffix.as_ptr() as usize, original + consume);
                        }
                        assert_eq!(reader.reader().reads, 1);
                        drop(reader);
                        if let Some(first) = held {
                            assert_eq!(&first[..], &input[start..end]);
                        }
                        assert_eq!(&suffix[..], &input[consume..]);
                    }
                }
            }
        }
    }

    #[test]
    fn copied_owned_items_and_borrowed_overlaps_keep_the_suffix_in_original_storage() {
        let codec = FrameScript::new([
            Frame::need_more(0),
            Frame::ready_copy(1..5, 1),
            Frame::ready_copy(1..3, 2),
            Frame::ready_copy(0..3, 1),
            Frame::ready_split(.., 0),
        ]);
        let mut reader = FramedRead::new(Script::new([Step::Data(b"abcdefgh")]), codec);
        let first = reader.decode_bytes().unwrap().unwrap();
        let original = reader.reader().destinations[0];
        let view = reader.decode_view().unwrap().unwrap();
        assert_eq!(view, b"cd");
        assert_eq!(view.as_ptr() as usize, original + 2);
        assert_ne!(view.as_ptr(), first[1..].as_ptr());
        let third = reader.decode_bytes().unwrap().unwrap();
        assert_eq!(&third[..], b"def");
        assert_ne!(third.as_ptr() as usize, original + 3);
        let suffix = reader.decode_view().unwrap().unwrap();
        assert_eq!(suffix.as_ptr() as usize, original + 4);
        assert_eq!(suffix, b"efgh");
        assert_eq!(reader.reader().reads, 1);
        drop(reader);
        assert_eq!(&first[..], b"bcde");
        assert_eq!(&third[..], b"def");
    }

    #[test]
    fn copied_items_survive_refill_errors_short_reads_and_compact() {
        for capacity in [3, 8] {
            let codec = FrameScript::new([
                Frame::need_more(0),
                Frame::ready_copy(0..3, 1),
                Frame::need_more(0),
                Frame::need_more(0),
                Frame::need_more(0), // Reframing after each returned source error.
                Frame::need_more(0),
                Frame::ready_split(..4, 0),
            ]);
            let source = Script::new([
                Step::Data(b"abc"),
                Step::DirtyError(ErrorKind::Other),
                Step::Error(ErrorKind::Interrupted),
                Step::Data(b"d"),
                Step::Data(b"e"),
            ]);
            let mut reader = FramedRead::with_capacity(source, codec, capacity);
            let first = reader.decode_bytes().unwrap().unwrap();
            assert!(reader.decode_view().is_err());
            assert_eq!(reader.decoder().inputs.len(), 3);
            assert_eq!(reader.decoder().inputs.last().unwrap().1, b"bc");
            assert_eq!(
                reader.decode_bytes().unwrap_err().kind(),
                ErrorKind::Interrupted
            );
            let second = reader.decode_bytes().unwrap().unwrap();
            assert_eq!(&second[..], b"bcde");
            assert_eq!(
                reader.decoder().decoded,
                [b"abc".to_vec(), b"bcde".to_vec()]
            );
            drop(reader);
            assert_eq!(&first[..], b"abc");
            assert_eq!(&second[..], b"bcde");
        }
    }

    #[test]
    fn mutating_an_owned_copy_does_not_change_the_unconsumed_overlap() {
        let codec = FrameScript::new([
            Frame::need_more(0),
            Frame::ready_copy(..4, 1),
            Frame::ready_split(.., 0),
        ]);
        let mut reader = FramedRead::with_capacity(&b"abcd"[..], codec, 8);
        let first = reader.decode_bytes().unwrap().unwrap();
        let mut writable = first.try_into_mut().unwrap();
        writable[1] = b'B';
        assert_eq!(reader.decode_view().unwrap().unwrap(), b"bcd");
        assert_eq!(&writable[..], b"aBcd");
    }

    #[test]
    fn decode_errors_consume_extracted_frames_including_at_eof() {
        for eof in [false, true] {
            for mode in 0..3 {
                for (action, selected, suffix) in [
                    (Frame::ready_split(1..2, 2), &b"x"[..], &b"tail"[..]),
                    (Frame::ready_copy(1..4, 2), &b"x>!"[..], &b">!tail"[..]),
                ] {
                    let mut codec = FrameScript::new([
                        Frame::need_more(0),
                        Frame::ready_split(..1, 0),
                        Frame::skip(1),
                    ]);
                    let frames = [
                        action,
                        Frame::ready_split(..1, 0),
                        Frame::ready_split(.., 0),
                    ]
                    .into_iter()
                    .map(Ok);
                    if eof {
                        codec.frames.push_back(Ok(Frame::need_more(0)));
                        codec.eof_frames.extend(frames);
                    } else {
                        codec.frames.extend(frames);
                    }
                    let source = Script::new([Step::Data(b"p#<x>!tail")]);
                    let mut reader = FramedRead::new(source, codec);
                    assert_eq!(reader.decode_view().unwrap().unwrap(), b"p");

                    reader.decoder.reject_item = true;
                    for expected in [selected, &suffix[..1]] {
                        let error = match mode {
                            0 => reader.decode_bytes().unwrap_err(),
                            1 => reader.decode_view().unwrap_err(),
                            _ => reader.decode_owned().unwrap_err(),
                        };
                        assert_eq!(error.kind(), ErrorKind::InvalidData);
                        assert_eq!(reader.decoder().decoded.last().unwrap(), expected);
                    }
                    let reads = reader.reader().reads;
                    reader.decoder.reject_item = false;
                    assert_eq!(reader.decode_view().unwrap().unwrap(), &suffix[1..]);
                    assert_eq!(reader.reader().reads, reads);
                    assert!(reader.decode_bytes().unwrap().is_none());
                }
            }
        }
    }

    #[test]
    fn copied_overlap_at_eof_can_be_consumed_before_source_resumes() {
        let mut codec = FrameScript::new([
            Frame::need_more(0),
            Frame::need_more(0),
            Frame::ready_split(..1, 0),
        ]);
        codec.eof_frames = [
            Frame::ready_copy(0..3, 0),
            Frame::skip(0),
            Frame::ready_copy(1..3, 1),
            Frame::skip(2),
            Frame::need_more(0),
        ]
        .into_iter()
        .map(Ok)
        .collect();
        let source = Script::new([Step::Data(b"abc"), Step::Eof, Step::Eof, Step::Data(b"x")]);
        let mut reader = FramedRead::with_capacity(source, codec, 3);
        let first = reader.decode_bytes().unwrap().unwrap();
        let second = reader.decode_bytes().unwrap().unwrap();
        assert_ne!(second.as_ptr(), first[1..].as_ptr());
        assert!(reader.decode_view().unwrap().is_none());
        assert_eq!(reader.decode_view().unwrap().unwrap(), b"x");
        assert_eq!(
            reader
                .decoder()
                .inputs
                .iter()
                .filter(|(eof, _)| *eof)
                .count(),
            6
        );
        drop(reader);
        assert_eq!(&first[..], b"abc");
        assert_eq!(&second[..], b"bc");
    }

    #[test]
    fn eof_is_rechecked_after_source_errors_and_none_before_reading_resumes() {
        let mut codec = FrameScript::new([
            Frame::need_more(0),
            Frame::ready_split(..1, 0),
            Frame::need_more(0),
        ]);
        codec.eof_frames.push_back(Ok(Frame::ready_split(..0, 0)));
        let source = Script::new([
            Step::Eof,
            Step::Error(ErrorKind::WouldBlock),
            Step::Error(ErrorKind::Interrupted),
            Step::Eof,
            Step::Data(b"x"),
            Step::Eof,
        ]);
        let mut reader = FramedRead::new(source, codec);
        assert!(reader.decode_bytes().unwrap().unwrap().is_empty());
        assert!(matches!(reader.decode_view(), Err(e) if e.kind() == ErrorKind::WouldBlock));
        assert!(matches!(reader.decode_view(), Err(e) if e.kind() == ErrorKind::Interrupted));
        assert!(reader.decode_view().unwrap().is_none());
        assert_eq!(
            reader
                .decoder()
                .inputs
                .iter()
                .filter(|(eof, _)| *eof)
                .count(),
            4
        );
        assert_eq!(reader.decode_view().unwrap().unwrap(), b"x");
        assert!(reader.decode_bytes().unwrap().is_none());
        assert_eq!(
            reader
                .decoder()
                .inputs
                .iter()
                .filter(|(eof, _)| *eof)
                .count(),
            6
        );
    }

    // Framing only selects bytes. Owned decoding can transform its own storage;
    // a borrowed Item owns any transformed bytes separately from its raw input.
    struct Editing {
        copy: bool,
    }
    impl Framer for Editing {
        type Error = io::Error;
        fn frame(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
            Ok(if src.is_empty() {
                Frame::need_more(0)
            } else if src[0] == b'#' {
                Frame::skip(1)
            } else if src.len() < 8 {
                Frame::need_more(0)
            } else if self.copy {
                Frame::ready_copy(1..3, 2)
            } else {
                Frame::ready_split(1..3, 1)
            })
        }
        fn frame_eof(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
            Ok(if src.is_empty() {
                Frame::need_more(0)
            } else {
                Frame::ready_split(.., 0)
            })
        }
    }
    impl BytesDecoder for Editing {
        type Item = Bytes;
        fn decode_bytes(&mut self, mut src: BytesMut) -> io::Result<Bytes> {
            src[0] = 0xff;
            Ok(src.freeze())
        }
    }
    impl ViewDecoder for Editing {
        type Item<'a> = (&'a [u8], Vec<u8>);
        fn decode_view<'a>(&mut self, src: &'a [u8]) -> io::Result<Self::Item<'a>> {
            let mut transformed = src.to_vec();
            transformed[0] = 0xff;
            Ok((src, transformed))
        }
    }

    #[test]
    fn decoder_transformations_leave_unconsumed_input_unchanged() {
        for copy in [false, true] {
            for owned in [false, true] {
                let source = Script::new([Step::Data(b"#<ab>tail")]);
                let mut reader = FramedRead::new(source, Editing { copy });
                if owned {
                    let item = reader.decode_bytes().unwrap().unwrap();
                    assert_eq!(&item[..], b"\xffb");
                    assert_eq!(
                        item.as_ptr() as usize == reader.reader().destinations[0] + 2,
                        !copy
                    );
                } else {
                    let (raw, transformed) = reader.decode_view().unwrap().unwrap();
                    assert_eq!(raw, b"ab");
                    assert_eq!(transformed, b"\xffb");
                    let address = raw.as_ptr() as usize;
                    assert_eq!(address, reader.reader().destinations[0] + 2);
                }
                let (suffix, _) = reader.decode_view().unwrap().unwrap();
                assert_eq!(suffix, if copy { &b"b>tail"[..] } else { b"tail" });
            }
        }
    }

    #[test]
    fn framing_errors_keep_state_changes_and_leave_input_for_retry() {
        struct RejectOnce(bool);
        impl Framer for RejectOnce {
            type Error = io::Error;
            fn frame(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
                if src.is_empty() {
                    return Ok(Frame::need_more(0));
                }
                if !self.0 {
                    self.0 = true;
                    return Err(io::Error::new(ErrorKind::InvalidData, "header rejected"));
                }
                assert_eq!(src.to_vec(), b"#abc");
                Ok(Frame::ready_split(1.., 0))
            }
            fn frame_eof(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
                self.frame(src)
            }
        }
        impl ViewDecoder for RejectOnce {
            type Item<'a> = &'a [u8];
            fn decode_view<'a>(&mut self, src: &'a [u8]) -> io::Result<&'a [u8]> {
                Ok(src)
            }
        }
        let mut reader = FramedRead::new(&b"#abc"[..], RejectOnce(false));
        assert_eq!(
            reader.decode_view().unwrap_err().kind(),
            ErrorKind::InvalidData
        );
        assert!(reader.decoder().0);
        assert_eq!(reader.decode_view().unwrap().unwrap(), b"abc");
    }

    #[test]
    fn need_more_keeps_the_prefix_after_skipping_across_a_dirty_read_error() {
        #[derive(Default)]
        struct RemoveHeader {
            removed: bool,
        }
        impl Framer for RemoveHeader {
            type Error = io::Error;
            fn frame(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
                if !self.removed && !src.is_empty() {
                    assert_eq!(src[0], b'#');
                    self.removed = true;
                    return Ok(Frame::skip(1));
                }
                Ok(if src.len() < 2 {
                    Frame::need_more(0)
                } else {
                    Frame::ready_split(..2, 0)
                })
            }
            fn frame_eof(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
                self.frame(src)
            }
        }
        impl ViewDecoder for RemoveHeader {
            type Item<'a> = &'a [u8];
            fn decode_view<'a>(&mut self, src: &'a [u8]) -> io::Result<&'a [u8]> {
                Ok(src)
            }
        }
        let source = Script::new([
            Step::Data(b"#a"),
            Step::DirtyError(ErrorKind::Other),
            Step::Data(b"b"),
        ]);
        let mut reader = FramedRead::with_capacity(source, RemoveHeader::default(), 2);
        assert_eq!(reader.decode_view().unwrap_err().kind(), ErrorKind::Other);
        assert!(reader.decoder().removed);
        assert_eq!(reader.decode_view().unwrap().unwrap(), b"ab");
    }

    struct DelimitedLines;

    impl Framer for DelimitedLines {
        type Error = io::Error;

        fn frame(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
            Ok(match src.iter().position(|&b| b == b'\n') {
                Some(end) if src[0] == b'#' => Frame::skip(end + 1),
                Some(end) => Frame::ready_split(..end, 1),
                None => Frame::need_more(0),
            })
        }

        fn frame_eof(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
            if src.is_empty() {
                Ok(Frame::need_more(0))
            } else {
                Ok(Frame::ready_split(.., 0))
            }
        }
    }

    impl BytesDecoder for DelimitedLines {
        type Item = BytesMut;

        fn decode_bytes(&mut self, frame: BytesMut) -> io::Result<BytesMut> {
            Ok(frame)
        }
    }

    impl ViewDecoder for DelimitedLines {
        type Item<'a> = &'a [u8];

        fn decode_view<'a>(&mut self, frame: &'a [u8]) -> io::Result<&'a [u8]> {
            Ok(frame)
        }
    }

    struct Triples {
        copy: bool,
    }

    impl Framer for Triples {
        type Error = io::Error;

        fn frame(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
            Ok(if src.len() < 3 {
                Frame::need_more(0)
            } else if self.copy {
                Frame::ready_copy(..3, 3)
            } else {
                Frame::ready_split(..3, 0)
            })
        }

        fn frame_eof(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
            if src.is_empty() {
                Ok(Frame::need_more(0))
            } else if src.len() < 3 {
                Err(io::ErrorKind::UnexpectedEof.into())
            } else {
                self.frame(src)
            }
        }
    }

    impl BytesDecoder for Triples {
        type Item = BytesMut;

        fn decode_bytes(&mut self, frame: BytesMut) -> io::Result<BytesMut> {
            Ok(frame)
        }
    }

    impl ViewDecoder for Triples {
        type Item<'a> = &'a [u8];

        fn decode_view<'a>(&mut self, frame: &'a [u8]) -> io::Result<&'a [u8]> {
            Ok(frame)
        }
    }

    #[derive(Default)]
    struct Repeating {
        bytes: usize,
        first_request: usize,
        max_request: usize,
    }

    impl Read for Repeating {
        fn read(&mut self, dst: &mut [u8]) -> io::Result<usize> {
            assert!(!dst.is_empty());
            if self.bytes == 0 {
                self.first_request = dst.len();
            }
            self.max_request = self.max_request.max(dst.len());
            for (i, byte) in dst.iter_mut().enumerate() {
                *byte = b"ab\n"[(self.bytes + i) % 3];
            }
            self.bytes += dst.len();
            Ok(dst.len())
        }
    }

    #[test]
    fn bounded_items_reuse_consumed_space_instead_of_growing_with_the_stream() {
        // Three-byte Items straddle an eight-byte buffer. A pending tail
        // prevents the empty-buffer reset from masking a failure to reclaim
        // the prefix.
        for (view, copy) in [(true, false), (false, true), (false, false)] {
            let mut reader = FramedRead::with_capacity(Repeating::default(), Triples { copy }, 8);
            for _ in 0..100_000 {
                if view {
                    assert_eq!(reader.decode_view().unwrap().unwrap(), b"ab\n");
                } else {
                    assert_eq!(&reader.decode_bytes().unwrap().unwrap()[..], b"ab\n");
                }
            }
            let source = reader.reader();
            assert_eq!(source.first_request, 8);
            assert!(source.max_request <= source.first_request);
        }
    }

    #[test]
    fn owned_reads_grow_as_needed_while_items_are_retained() {
        struct Fixed(usize);

        impl Framer for Fixed {
            type Error = io::Error;

            fn frame(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
                Ok(if src.len() < self.0 {
                    Frame::need_more(0)
                } else {
                    Frame::ready_split(..self.0, 0)
                })
            }
        }

        impl BytesDecoder for Fixed {
            type Item = BytesMut;

            fn decode_bytes(&mut self, frame: BytesMut) -> io::Result<BytesMut> {
                Ok(frame)
            }
        }

        #[derive(Default)]
        struct Source {
            requests: Vec<usize>,
        }

        impl Read for Source {
            fn read(&mut self, dst: &mut [u8]) -> io::Result<usize> {
                assert!(!dst.is_empty());
                self.requests.push(dst.len());
                dst.fill(0xa5);
                Ok(dst.len())
            }
        }

        for capacity in [8, 128 * 1024] {
            let width = capacity / 2;
            let mut reader = FramedRead::with_capacity(Source::default(), Fixed(width), capacity);
            let items: Vec<_> = (0..8)
                .map(|_| reader.decode_bytes().unwrap().unwrap())
                .collect();
            assert_eq!(reader.reader().requests[0], capacity);
            assert!(reader.reader().requests.len() > 1);
            drop(reader);
            for item in items {
                assert_eq!(item.len(), width);
                assert!(item.iter().all(|&byte| byte == 0xa5));
            }
        }
    }

    #[test]
    fn minimum_capacities_read_input_and_handle_empty_sources() {
        for capacity in [0, 1] {
            for (view, copy) in [(true, false), (false, true), (false, false)] {
                let (mut reader, count) = allocations(|| {
                    FramedRead::with_capacity(Cursor::new(b"ab\ncd\n"), Triples { copy }, capacity)
                });
                if capacity == 0 {
                    assert_eq!(count, 0);
                }
                for expected in [b"ab\n", b"cd\n"] {
                    if view {
                        assert_eq!(reader.decode_view().unwrap().unwrap(), expected);
                    } else {
                        assert_eq!(&reader.decode_bytes().unwrap().unwrap()[..], expected);
                    }
                }
                assert!(reader.decode_view().unwrap().is_none());
                assert_eq!(reader.reader().position(), 6);
            }
            let mut empty = FramedRead::with_capacity(io::empty(), DelimitedLines, capacity);
            assert!(empty.decode_view().unwrap().is_none());
        }
    }

    #[test]
    fn consumption_and_mixed_extraction_preserve_the_remaining_input() {
        let mut reader =
            FramedRead::with_capacity(&b"#header\na\nb\n\ntail"[..], DelimitedLines, 32);
        assert_eq!(reader.decode_view().unwrap().unwrap(), b"a");
        let owned = reader.decode_bytes().unwrap().unwrap();
        assert!(reader.decode_view().unwrap().unwrap().is_empty());
        assert_eq!(reader.decode_view().unwrap().unwrap(), b"tail");
        assert!(reader.decode_bytes().unwrap().is_none());
        assert!(reader.decode_view().unwrap().is_none());
        drop(reader);
        assert_eq!(&owned[..], b"b");
    }

    #[test]
    fn short_reads_keep_empty_items_arbitrary_bytes_and_the_eof_tail() {
        for capacity in [1, 3, 16] {
            let source = ShortReads {
                source: Cursor::new(b"#h\n\0\xff\n\nlast"),
                limit: 2,
            };
            let mut reader = FramedRead::with_capacity(source, DelimitedLines, capacity);
            let first = reader.decode_bytes().unwrap().unwrap();
            assert!(reader.decode_view().unwrap().unwrap().is_empty());
            let last = reader.decode_bytes().unwrap().unwrap();
            assert!(reader.decode_view().unwrap().is_none());
            assert_eq!(&first[..], b"\0\xff");
            assert_eq!(&last[..], b"last");
        }
    }

    #[test]
    fn a_read_error_after_consumption_preserves_the_unfinished_item() {
        let source = Script::new([
            Step::Data(b"a\nkey="),
            Step::DirtyError(ErrorKind::Interrupted),
            Step::Data(b"ok\n"),
        ]);
        let mut reader = FramedRead::with_capacity(source, DelimitedLines, 6);
        assert_eq!(reader.decode_view().unwrap().unwrap(), b"a");
        assert_eq!(
            reader.decode_view().unwrap_err().kind(),
            io::ErrorKind::Interrupted
        );
        let item = reader.decode_bytes().unwrap().unwrap();
        assert_eq!(&item[..], b"key=ok");
        assert!(reader.decode_view().unwrap().is_none());
    }

    struct InvalidFrame {
        split: bool,
    }

    impl Framer for InvalidFrame {
        type Error = io::Error;

        fn frame(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
            Ok(if src.is_empty() {
                Frame::need_more(0)
            } else if self.split {
                Frame::ready_split(..3, 0)
            } else {
                Frame::ready_copy(..1, 3)
            })
        }

        fn frame_eof(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
            self.frame(src)
        }
    }

    impl BytesDecoder for InvalidFrame {
        type Item = BytesMut;

        fn decode_bytes(&mut self, frame: BytesMut) -> io::Result<BytesMut> {
            Ok(frame)
        }
    }

    #[test]
    fn consumption_is_bounded_by_input_even_when_initialized_storage_is_larger() {
        let mut reader = FramedRead::with_capacity(&b"ab"[..], InvalidFrame { split: false }, 8);
        assert!(catch_unwind(AssertUnwindSafe(|| reader.decode_bytes())).is_err());
    }

    #[test]
    fn owned_splitting_cannot_expose_initialized_tail_as_input() {
        let mut reader = FramedRead::with_capacity(&b"ab"[..], InvalidFrame { split: true }, 8);
        assert!(catch_unwind(AssertUnwindSafe(|| reader.decode_bytes())).is_err());
    }
}
