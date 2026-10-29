use std::io::{self, Write};

/// Encodes Items to a [`Write`] destination.
///
/// `Item` may contain borrowed fields. An encoder may implement this trait for
/// multiple input types and keep format state across calls.
///
/// # Examples
///
/// ```
/// use std::io::{self, BufWriter, Write};
///
/// use decen::write::Encoder;
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
/// let mut output = BufWriter::new(Vec::new());
/// Pairs.encode([1, 2], &mut output)?;
/// output.flush()?;
/// assert_eq!(output.get_ref(), &[1, 2]);
/// # Ok::<(), io::Error>(())
/// ```
pub trait Encoder<Item> {
    /// The error type for encoding and output I/O.
    type Error: From<io::Error>;

    /// Writes one complete encoded Item to `writer`.
    ///
    /// Success means all encoded bytes have been accepted by `writer`, which
    /// may buffer them. Leave flushing to the caller so Items can be batched.
    ///
    /// # Errors
    ///
    /// Encoding or I/O errors may leave a partial Item written and change
    /// encoder state. Retrying the entire Item may duplicate bytes.
    /// Implementations should document whether encoding can continue after an
    /// error.
    fn encode<W: Write + ?Sized>(&mut self, item: Item, writer: &mut W) -> Result<(), Self::Error>;
}

#[cfg(test)]
mod test {
    use std::io::{self, Write};

    use super::Encoder;

    struct Raw;
    impl<T: AsRef<[u8]>> Encoder<T> for Raw {
        type Error = io::Error;
        fn encode<W: Write + ?Sized>(&mut self, item: T, writer: &mut W) -> io::Result<()> {
            writer.write_all(item.as_ref())
        }
    }

    #[test]
    fn encoder_can_write_directly_to_bufwriter_and_unsized_destinations() {
        let mut output = io::BufWriter::with_capacity(4, Vec::new());
        Raw.encode(b"ab", &mut output).unwrap();
        assert_eq!(output.buffer(), b"ab");
        assert!(output.get_ref().is_empty());
        Raw.encode(b"cdefgh", &mut output).unwrap();
        assert_eq!(output.get_ref(), b"abcdefgh");
        assert!(output.buffer().is_empty());

        let destination: &mut dyn Write = &mut output;
        Raw.encode(b"ij", destination).unwrap();
        destination.flush().unwrap();
        assert_eq!(output.get_ref(), b"abcdefghij");
    }
}
