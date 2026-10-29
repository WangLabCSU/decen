//! Encoding Items into byte streams with [`FramedWrite`].

use std::io::{self, BufWriter, Write};

mod encoder;

pub use self::encoder::Encoder;

use crate::DEFAULT_BUF_SIZE;

/// A buffered writer that encodes Items into a byte stream.
///
/// Each [`write`](Self::write) call passes an Item to the supplied [`Encoder`]
/// and buffers its encoded bytes for a [`Write`] destination. The encoder
/// defines the output format, including any frame headers or separators.
///
/// It is important to call [`flush`](Self::flush) and handle its result before
/// dropping this writer. Dropping attempts to write buffered bytes, but any
/// errors are ignored, and the underlying destination is not flushed.
/// A successful `flush` empties the buffer and flushes the destination. If no
/// more bytes are written afterward, there is no buffered output to write on
/// drop.
///
/// # Examples
///
/// ```
/// use std::io::{self, Write};
///
/// use decen::write::{Encoder, FramedWrite};
///
/// struct Pairs;
///
/// impl Encoder<[u8; 2]> for Pairs {
///     type Error = io::Error;
///
///     fn encode<W: Write + ?Sized>(&mut self, item: [u8; 2], writer: &mut W) -> io::Result<()> {
///         writer.write_all(&item)
///     }
/// }
///
/// let mut writer = FramedWrite::new(Vec::new(), Pairs);
/// writer.write([0xff, 0])?;
/// writer.flush()?;
/// assert_eq!(writer.writer(), &[0xff, 0]);
/// # Ok::<(), io::Error>(())
/// ```
#[derive(Debug)]
pub struct FramedWrite<W: ?Sized + Write, E> {
    encoder: E,
    inner: BufWriter<W>,
}

impl<W: Write, E> FramedWrite<W, E> {
    /// Creates a writer using `encoder` and the default buffer capacity.
    ///
    /// The current default is 8 KiB, or 512 bytes on ESP-IDF.
    pub fn new(inner: W, encoder: E) -> Self {
        Self::with_capacity(inner, encoder, DEFAULT_BUF_SIZE)
    }

    /// Creates a writer with a buffer capacity of at least `capacity` bytes.
    ///
    /// Capacity does not limit Item size. Large writes can go directly to the
    /// destination. Zero disables buffering.
    pub fn with_capacity(inner: W, encoder: E, capacity: usize) -> Self {
        Self {
            encoder,
            inner: BufWriter::with_capacity(capacity, inner),
        }
    }
}

impl<W: ?Sized + Write, E> FramedWrite<W, E> {
    /// Returns a reference to the underlying writer.
    pub fn writer(&self) -> &W {
        self.inner.get_ref()
    }

    /// Returns a reference to the encoder.
    pub fn encoder(&self) -> &E {
        &self.encoder
    }

    /// Returns a reference to the buffered bytes waiting to be written.
    pub fn buffer(&self) -> &[u8] {
        self.inner.buffer()
    }

    /// Returns the buffer's capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Encodes an Item and writes its bytes to the buffered destination.
    ///
    /// Bytes may be written to the destination during encoding. Success means
    /// all encoded bytes were accepted, but some may remain buffered. Call
    /// [`flush`](Self::flush) to write pending output and flush the
    /// destination.
    ///
    /// # Errors
    ///
    /// Returns encoding or I/O errors as `E::Error`. An error may leave part of
    /// the Item written or buffered and change encoder state. Bytes not
    /// accepted during encoding are not retained for a later flush. Retrying
    /// the entire Item may duplicate bytes; whether encoding can continue
    /// depends on the encoder.
    #[inline]
    pub fn write<Item>(&mut self, item: Item) -> Result<(), E::Error>
    where
        E: Encoder<Item>,
    {
        self.encoder.encode(item, &mut self.inner)
    }

    /// Writes buffered bytes, then flushes the underlying writer.
    ///
    /// # Errors
    ///
    /// Returns an error if the output cannot be fully written or flushed. Bytes
    /// remaining in the buffer can be written by retrying the flush.
    /// Interrupted writes of buffered bytes are retried automatically. Errors
    /// from the underlying writer's `flush` are returned without retrying.
    pub fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod test {
    use std::{
        collections::VecDeque,
        io::{self, ErrorKind, Write},
        panic::{AssertUnwindSafe, catch_unwind},
    };

    use super::{Encoder, FramedWrite};

    struct Raw;

    impl<T: AsRef<[u8]>> Encoder<T> for Raw {
        type Error = io::Error;

        fn encode<W: io::Write + ?Sized>(&mut self, item: T, writer: &mut W) -> io::Result<()> {
            writer.write_all(item.as_ref())?;
            Ok(())
        }
    }

    enum Step {
        Accept(usize),
        Error(ErrorKind),
        Panic,
        AcceptThenPanic(usize),
    }

    #[derive(Default)]
    struct Script {
        steps: VecDeque<Step>,
        flush_errors: VecDeque<ErrorKind>,
        output: Vec<u8>,
        writes: usize,
        flushes: usize,
    }

    impl Script {
        fn new(steps: impl IntoIterator<Item = Step>) -> Self {
            Self {
                steps: steps.into_iter().collect(),
                ..Self::default()
            }
        }
    }

    impl Write for Script {
        fn write(&mut self, src: &[u8]) -> io::Result<usize> {
            assert!(!src.is_empty());
            self.writes += 1;
            match self.steps.pop_front().unwrap_or(Step::Accept(usize::MAX)) {
                Step::Accept(limit) => {
                    let n = limit.min(src.len());
                    self.output.extend_from_slice(&src[..n]);
                    Ok(n)
                }
                Step::Error(kind) => Err(kind.into()),
                Step::Panic => panic!("writer panicked"),
                Step::AcceptThenPanic(limit) => {
                    self.output.extend_from_slice(&src[..limit.min(src.len())]);
                    panic!("writer panicked after accepting bytes");
                }
            }
        }

        fn flush(&mut self) -> io::Result<()> {
            self.flushes += 1;
            self.flush_errors
                .pop_front()
                .map_or(Ok(()), |kind| Err(kind.into()))
        }
    }

    #[derive(Debug)]
    enum EncodeError {
        Io(io::Error),
        InvalidItem,
    }

    impl From<io::Error> for EncodeError {
        fn from(error: io::Error) -> Self {
            Self::Io(error)
        }
    }

    #[derive(Default)]
    struct Fallible {
        calls: usize,
    }

    impl Encoder<bool> for Fallible {
        type Error = EncodeError;

        fn encode<W: io::Write + ?Sized>(
            &mut self,
            fail: bool,
            writer: &mut W,
        ) -> Result<(), EncodeError> {
            self.calls += 1;
            writer.write_all(if fail { b"partial" } else { b"ok" })?;
            if fail {
                Err(EncodeError::InvalidItem)
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn multi_part_encoding_matches_bufwriter_for_variable_items() {
        struct Prefixed;
        impl Encoder<&[u8]> for Prefixed {
            type Error = io::Error;

            fn encode<W: Write + ?Sized>(&mut self, item: &[u8], writer: &mut W) -> io::Result<()> {
                writer.write_all(&(item.len() as u32).to_le_bytes())?;
                writer.write_all(item)
            }
        }

        for capacity in [0, 1, 8, 8192] {
            let sink = || Script::new([Step::Accept(1), Step::Accept(2), Step::Accept(1)]);
            let mut framed = FramedWrite::with_capacity(sink(), Prefixed, capacity);
            let mut standard = io::BufWriter::with_capacity(capacity, sink());
            for len in [0, 1, 7, 31, 255, 1023, 4097, 9001, 65537, 3] {
                let item = vec![len as u8; len];
                framed.write(item.as_slice()).unwrap();
                standard.write_all(&(len as u32).to_le_bytes()).unwrap();
                standard.write_all(&item).unwrap();
                assert_eq!(framed.buffer(), standard.buffer());
                assert_eq!(framed.writer().output, standard.get_ref().output);
                assert_eq!(framed.writer().writes, standard.get_ref().writes);
                assert_eq!(framed.capacity(), capacity);
                assert_eq!(framed.writer().flushes, 0);
            }
            framed.flush().unwrap();
            standard.flush().unwrap();
            assert_eq!(framed.writer().output, standard.get_ref().output);
            assert_eq!(framed.writer().flushes, standard.get_ref().flushes);
        }
    }

    #[test]
    fn direct_write_errors_do_not_buffer_the_unaccepted_item_suffix() {
        let sink = Script::new([Step::Accept(2), Step::Error(ErrorKind::WouldBlock)]);
        let mut writer = FramedWrite::with_capacity(sink, Raw, 4);
        assert_eq!(
            writer.write(b"abcdef").unwrap_err().kind(),
            ErrorKind::WouldBlock
        );
        assert_eq!(writer.writer().output, b"ab");
        assert!(writer.buffer().is_empty());
        writer.flush().unwrap();
        assert_eq!(writer.writer().output, b"ab");
        assert_eq!(writer.writer().writes, 2);
    }

    #[test]
    fn items_are_batched_until_space_is_needed_or_explicit_flush() {
        let mut writer = FramedWrite::with_capacity(Script::default(), Raw, 3);
        writer.write(b"ab").unwrap();
        assert!(writer.writer().output.is_empty());
        writer.write(b"cd").unwrap();
        assert_eq!(writer.writer().output, b"ab");
        assert_eq!(writer.writer().flushes, 0);
        assert_eq!(writer.buffer(), b"cd");
        writer.flush().unwrap();
        assert_eq!(writer.writer().output, b"abcd");
        assert_eq!(writer.writer().flushes, 1);
        assert!(writer.buffer().is_empty());
    }

    #[test]
    fn automatic_drain_preserves_underlying_buffering_until_explicit_flush() {
        let sink = io::BufWriter::with_capacity(64, Script::default());
        let mut writer = FramedWrite::with_capacity(sink, Raw, 3);
        writer.write(b"ab").unwrap();
        writer.write(b"cd").unwrap();
        assert_eq!(writer.writer().buffer(), b"ab");
        assert!(writer.writer().get_ref().output.is_empty());
        assert_eq!(writer.writer().get_ref().flushes, 0);

        writer.flush().unwrap();
        assert_eq!(writer.writer().get_ref().output, b"abcd");
        assert_eq!(writer.writer().get_ref().writes, 1);
        assert_eq!(writer.writer().get_ref().flushes, 1);
    }

    #[test]
    fn panic_after_partial_write_preserves_only_the_unwritten_suffix() {
        let sink = Script::new([Step::Accept(2), Step::Panic]);
        let mut writer = FramedWrite::new(sink, Raw);
        writer.write(b"abcde").unwrap();
        assert!(catch_unwind(AssertUnwindSafe(|| writer.flush())).is_err());
        assert_eq!(writer.writer().output, b"ab");
        assert_eq!(writer.buffer(), b"cde");
        assert_eq!(writer.writer().flushes, 0);

        writer.write(b"fg").unwrap();
        writer.flush().unwrap();
        assert_eq!(writer.writer().output, b"abcdefg");
    }

    #[test]
    fn interrupted_writes_are_retried_and_destination_flush_errors_are_returned() {
        let mut sink = Script::new([
            Step::Accept(1),
            Step::Error(ErrorKind::Interrupted),
            Step::Accept(2),
        ]);
        sink.flush_errors.push_back(ErrorKind::Interrupted);
        let mut writer = FramedWrite::new(sink, Raw);
        writer.write(b"abc").unwrap();
        assert_eq!(writer.flush().unwrap_err().kind(), ErrorKind::Interrupted);
        assert_eq!(writer.writer().output, b"abc");
        assert!(writer.buffer().is_empty());
        assert_eq!(writer.writer().writes, 3);
        assert_eq!(writer.writer().flushes, 1);

        writer.flush().unwrap();
        assert_eq!(writer.writer().writes, 3);
        assert_eq!(writer.writer().flushes, 2);
    }

    #[test]
    fn new_items_can_follow_the_unwritten_suffix_after_an_io_error() {
        let sink = Script::new([Step::Accept(1), Step::Error(ErrorKind::WouldBlock)]);
        let mut writer = FramedWrite::new(sink, Raw);
        writer.write(b"abc").unwrap();
        assert_eq!(writer.flush().unwrap_err().kind(), ErrorKind::WouldBlock);
        assert_eq!(writer.buffer(), b"bc");
        writer.write(b"d").unwrap();
        writer.flush().unwrap();
        assert_eq!(writer.writer().output, b"abcd");
    }

    #[test]
    fn write_zero_and_flush_errors_preserve_the_correct_retry_position() {
        let mut sink = Script::new([Step::Accept(2), Step::Accept(0)]);
        sink.flush_errors.push_back(ErrorKind::BrokenPipe);
        let mut writer = FramedWrite::new(sink, Raw);
        writer.write(b"abc").unwrap();
        assert_eq!(writer.flush().unwrap_err().kind(), ErrorKind::WriteZero);
        assert_eq!(writer.buffer(), b"c");
        assert_eq!(writer.flush().unwrap_err().kind(), ErrorKind::BrokenPipe);
        assert!(writer.buffer().is_empty());
        let writes = writer.writer().writes;
        writer.flush().unwrap();
        assert_eq!(writer.writer().writes, writes);
        assert_eq!(writer.writer().output, b"abc");
    }

    #[test]
    fn automatic_drain_errors_are_converted_during_encoding() {
        let sink = Script::new([Step::Accept(1), Step::Error(ErrorKind::WouldBlock)]);
        let mut writer = FramedWrite::with_capacity(sink, Fallible::default(), 3);
        writer.write(false).unwrap();
        assert!(matches!(
            writer.write(true),
            Err(EncodeError::Io(error)) if error.kind() == ErrorKind::WouldBlock
        ));
        assert_eq!(writer.encoder().calls, 2);
        assert_eq!(writer.writer().output, b"o");
        assert_eq!(writer.buffer(), b"k");
        writer.write(false).unwrap();
        writer.flush().unwrap();
        assert_eq!(writer.writer().output, b"okok");
    }

    #[test]
    fn encoding_errors_preserve_appended_bytes_and_do_not_disable_the_writer() {
        let mut writer = FramedWrite::new(Vec::new(), Fallible::default());
        writer.write(false).unwrap();
        assert!(matches!(writer.write(true), Err(EncodeError::InvalidItem)));
        assert_eq!(writer.encoder().calls, 2);
        assert_eq!(writer.buffer(), b"okpartial");
        writer.write(false).unwrap();
        assert_eq!(writer.encoder().calls, 3);
        writer.flush().unwrap();
        assert_eq!(writer.writer(), b"okpartialok");
    }

    #[test]
    fn drop_writes_pending_output_without_flushing_the_destination() {
        for input in [&b""[..], b"abcdef"] {
            let mut sink = Script::new([Step::Accept(1), Step::Accept(2)]);
            {
                let mut writer = FramedWrite::new(&mut sink, Raw);
                writer.write(input).unwrap();
            }
            assert_eq!(sink.output, input);
            assert_eq!(sink.writes, if input.is_empty() { 0 } else { 3 });
            assert_eq!(sink.flushes, 0);
        }
    }

    #[test]
    fn drop_ignores_errors_after_partial_output() {
        for failure in [Step::Error(ErrorKind::BrokenPipe), Step::Accept(0)] {
            let mut sink = Script::new([Step::Accept(2), failure]);
            {
                let mut writer = FramedWrite::new(&mut sink, Raw);
                writer.write(b"abcde").unwrap();
            }
            assert_eq!(sink.output, b"ab");
            assert_eq!(sink.writes, 2);
            assert_eq!(sink.flushes, 0);
        }
    }

    #[test]
    fn drop_retries_only_the_pending_suffix_after_an_io_error() {
        let mut sink = Script::new([Step::Accept(2), Step::Error(ErrorKind::WouldBlock)]);
        {
            let mut writer = FramedWrite::new(&mut sink, Raw);
            writer.write(b"abcde").unwrap();
            assert_eq!(writer.flush().unwrap_err().kind(), ErrorKind::WouldBlock);
        }
        assert_eq!(sink.output, b"abcde");
        assert_eq!(sink.writes, 3);
        assert_eq!(sink.flushes, 0);
    }

    #[test]
    fn drop_does_not_retry_a_panicking_writer_during_unwinding() {
        let mut sink = Script::new([Step::AcceptThenPanic(2)]);
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let mut writer = FramedWrite::new(&mut sink, Raw);
                writer.write(b"abcde").unwrap();
                writer.flush().unwrap();
            }))
            .is_err()
        );
        assert_eq!(sink.output, b"ab");
        assert_eq!(sink.writes, 1);
        assert_eq!(sink.flushes, 0);
    }

    #[test]
    fn drop_does_not_retry_after_a_caught_write_panic() {
        let mut sink = Script::new([Step::AcceptThenPanic(2)]);
        {
            let mut writer = FramedWrite::new(&mut sink, Raw);
            writer.write(b"abcde").unwrap();
            assert!(catch_unwind(AssertUnwindSafe(|| writer.flush())).is_err());
            drop(writer);
        }
        assert_eq!(sink.output, b"ab");
        assert_eq!(sink.writes, 1);
        assert_eq!(sink.flushes, 0);
    }

    #[test]
    fn an_unrelated_panic_does_not_discard_pending_output() {
        let mut sink = Script::default();
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let mut writer = FramedWrite::new(&mut sink, Raw);
                writer.write(b"abc").unwrap();
                panic!("caller panicked");
            }))
            .is_err()
        );
        assert_eq!(sink.output, b"abc");
        assert_eq!(sink.writes, 1);
        assert_eq!(sink.flushes, 0);
    }

    #[test]
    fn drop_supports_an_unsized_destination() {
        let mut sink = Script::default();
        {
            let mut writer: Box<FramedWrite<dyn Write + '_, Raw>> =
                Box::new(FramedWrite::new(&mut sink, Raw));
            writer.write(b"abc").unwrap();
        }
        assert_eq!(sink.output, b"abc");
        assert_eq!(sink.flushes, 0);
    }
}
