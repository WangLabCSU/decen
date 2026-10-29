# decen

**decen** takes its name from **decoder** and **encoder**. It provides traits
for defining frame boundaries, decoding frames, and encoding values, together
with buffered readers and writers that use those implementations.

The design is inspired by
[`tokio-util::codec`](https://docs.rs/tokio-util/latest/tokio_util/codec/index.html),
which adapts asynchronous I/O into `Stream` and `Sink` transports. decen focuses
on efficient frame processing through `std::io::Read` and `std::io::Write`,
emphasizing memory reuse and control over the ownership of decoded data.

Borrow framed bytes for immediate processing, retain them without copying, or
build an owned Item containing only the data you need. A single framing
implementation can support all three modes.

You implement the format's framing, decoding, or encoding traits as needed. A
decoded result or a value supplied to an encoder is called an **Item**. Input
need not be UTF-8, and reading and writing can be used independently.

## Example

This format treats each byte as an Item. The same decoder supports borrowing
framed bytes, retaining them without copying, and copying them into an owned
value. The example writes each result through an encoder.

```rust
use std::io::{self, Cursor};

use bytes::BytesMut;
use decen::read::{
    BytesDecoder, Frame, FramedRead, Framer, OwnedDecoder, ViewDecoder,
};
use decen::write::{Encoder, FramedWrite};

struct ByteCodec;

impl Framer for ByteCodec {
    type Error = io::Error;

    fn frame(&mut self, src: &[u8]) -> io::Result<Frame> {
        Ok(if src.is_empty() {
            Frame::need_more(0)
        } else {
            Frame::ready_split(..1, 0)
        })
    }
}

impl BytesDecoder for ByteCodec {
    type Item = BytesMut;

    fn decode_bytes(&mut self, frame: BytesMut) -> io::Result<Self::Item> {
        Ok(frame)
    }
}

impl ViewDecoder for ByteCodec {
    type Item<'a> = &'a [u8];

    fn decode_view<'a>(&mut self, frame: &'a [u8]) -> io::Result<Self::Item<'a>> {
        Ok(frame)
    }
}

impl OwnedDecoder for ByteCodec {
    type Item = Vec<u8>;

    fn decode_owned(&mut self, frame: &[u8]) -> io::Result<Self::Item> {
        Ok(frame.to_vec())
    }
}

impl<T: AsRef<[u8]>> Encoder<T> for ByteCodec {
    type Error = io::Error;

    fn encode<W: io::Write + ?Sized>(&mut self, item: T, writer: &mut W) -> io::Result<()> {
        let bytes = item.as_ref();
        if bytes.len() != 1 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "expected one byte"));
        }
        writer.write_all(bytes)?;
        Ok(())
    }
}

fn main() -> io::Result<()> {
    let mut reader = FramedRead::new(Cursor::new(b"\xff\0\x01"), ByteCodec);
    let mut writer = FramedWrite::new(Vec::new(), ByteCodec);

    let view = reader.decode_view()?.unwrap();
    assert_eq!(view, b"\xff");
    writer.write(view)?;

    // The view is no longer used, so reading can continue.
    let owned = reader.decode_bytes()?.unwrap();
    writer.write(&owned[..])?;

    let copied = reader.decode_owned()?.unwrap();
    writer.write(&copied)?;
    assert!(reader.decode_view()?.is_none());
    assert_eq!(&owned[..], b"\0"); // Owned Items can be retained across reads.
    assert_eq!(copied, b"\x01");

    writer.flush()?;
    assert_eq!(writer.writer(), b"\xff\0\x01");
    Ok(())
}
```

Sources and destinations can also be files, standard input and output, or custom
`Read` and `Write` implementations. Finish using a borrowed Item before reading
again. Owned results can be retained across reads and can outlive the reader.

## Decoding

A format implements `Framer` and one or more decoder traits. `FramedRead` reads
bytes as needed, asks the framer to select a complete frame, and passes the
framed bytes to the decoder. An Item can be a parsed record rather than a byte
container.

| Trait          | Decoder input | Example Item       | Reader method    |
| -------------- | ------------- | ------------------ | ---------------- |
| `ViewDecoder`  | `&'a [u8]`    | `Record<&'a [u8]>` | `decode_view()`  |
| `OwnedDecoder` | `&[u8]`       | `Record<Vec<u8>>`  | `decode_owned()` |
| `BytesDecoder` | `BytesMut`    | `Record<Bytes>`    | `decode_bytes()` |

- `decode_view()` lets the Item borrow framed bytes without copying, making it
  suitable for processing an Item before reading the next one.
- `decode_owned()` passes framed bytes as a slice, but the Item cannot borrow
  from it. The decoder explicitly copies any input data the Item needs to
  retain. Parsed values need no copy of the input bytes.
- `decode_bytes()` passes framed bytes directly as an owned `BytesMut`. The
  Item can retain those bytes across reads, allowing zero-copy decoding
  without borrowing from the reader.

Although a `ViewDecoder` Item can be fully owned, using this trait for results
that do not borrow the input is discouraged. Implement `OwnedDecoder` instead.

If a view Item provides a conversion such as `into_owned()`, an `OwnedDecoder`
implementation can reuse parsing with
`Ok(self.decode_view(frame)?.into_owned())`. The converted result must no longer
borrow the supplied frame. Such a conversion may allocate and copy borrowed
fields.

Each successful decode call returns one Item or `None` when no further Item is
available at EOF. The framer determines how many input bytes are consumed; this
can differ from the number of bytes selected for decoding.

With `decode_bytes()`, `Frame::ready_split` transfers the selected bytes without
copying. `Frame::ready_copy` copies the selected range into a `BytesMut` before
calling the decoder. It does not copy the rest of the buffer.

`ready_copy` allows adjacent frames to overlap by selecting bytes independently
of how many are consumed. For an owned result, prefer `decode_owned()` when the
decoder implements `OwnedDecoder`. It receives the selected bytes as a slice
and can construct the Item directly, such as a `Vec<u8>` or a parsed record,
without first copying the selection into a `BytesMut`. The decoder can copy
only the fields it needs to retain. Consider implementing `OwnedDecoder` when
your framer uses `ready_copy`.

Both `decode_view()` and `decode_owned()` receive framed bytes without an
additional copy for either decision.

Zero-copy here refers to passing framed bytes to the decoder. Reading and
buffering may still allocate or copy input bytes; the decoder determines any
further allocation or copying. Retained owned Items can keep shared allocations
alive after the reader is dropped.

To compare copying with sharing input storage, measure with your application's
Item sizes and retention patterns.

`FramedRead::with_capacity(source, decoder, capacity)` sets the initial buffer
capacity. It does not limit Item size or total memory use. Large Items and
retained results can increase memory use. Zero defers buffer allocation until
reading.

## Implementing a format

To decode with `FramedRead`, implement `Framer` and at least one of
`BytesDecoder`, `ViewDecoder`, or `OwnedDecoder` on the same type. Frame
boundaries can depend on format state, such as quoting or length prefixes.
Implement `Encoder<Item>` to write an Item's encoded bytes.

`Framer::frame` returns a `Frame` describing the next action.

| Constructor                    | Selection and consumption                                                                                                                           |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ready_split(range, addition)` | Selects `range` and consumes through its end plus `addition` bytes; `decode_bytes()` transfers the selection without copying                        |
| `ready_copy(range, consume)`   | Selects `range` independently of consuming the first `consume` bytes, allowing adjacent frames to overlap; `decode_bytes()` copies the framed bytes |
| `skip(amount)`                 | Skips the first `amount` bytes of the current input without producing an Item                                                                       |
| `need_more(consume)`           | Consumes the first `consume` bytes of the current input before requesting more data                                                                 |

Both ready actions can be used with any of the three decoder traits. Use
`frame.kind()` to match `FrameKind::{Ready, Skip, NeedMore}`.

All selections and consumption counts are relative to the input passed to
`frame` and must fit within it, including counts passed to `skip` and
`need_more`. An unbounded range end selects through the end of that input.
Invalid decisions can panic; earlier consumption and changes to format state
are not rolled back.

Empty Items and zero consumption are allowed.

`skip(amount)` immediately resumes framing. `need_more(consume)` attempts a read
before framing again. Bytes consumed before that read are not restored if it
returns zero, returns an error, or panics. The next framing call receives the
remaining input followed by any newly read bytes. A format can remember a scan
offset, but must adjust it after consumption. It cannot retain references to
input across framing calls or rely on its address remaining unchanged.

The decoder receives framed bytes. A view Item can borrow fields directly;
converted fields, such as unescaped text, can use storage owned by the Item.
`BytesDecoder` can return `BytesMut` directly or freeze it into `Bytes`. A
record can also hold the frame and expose its fields as slices.

## EOF and errors

`Framer::frame_eof` handles trailing input. Its default implementation calls
`frame` and returns `UnexpectedEof` if a `need_more` decision would retain input.
Override it to accept unterminated trailing Items or check unfinished format
state.

After `frame_eof` returns `need_more(consume)`, the reader consumes those bytes
and tries to read again. If the source still supplies no data, decoding returns
`None`, even if buffered input remains. That input is retained for later calls,
which can invoke `frame_eof` again without new data. If more bytes arrive,
framing resumes through `frame`.

`Framer::Error` is shared by framing and decoding; `Encoder::Error` is defined
separately. Both accept I/O errors through `From<io::Error>`.
`FramedWrite::flush` returns `io::Result<()>`.

| Failure                                               | Effect                                                                                                                                                                                         |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Read::read` returns an error                         | Keeps input remaining after prior consumption. Bytes written by the failed call are not appended. Returns the error, including `Interrupted`, without retrying                                 |
| The source reports more bytes than the supplied space | Returns `InvalidData` with the reported and available lengths; does not append data from that read                                                                                             |
| `frame` or `frame_eof` returns an error               | That call consumes no input; earlier consumption and changes to format state are not rolled back                                                                                               |
| A decoder returns an error                            | The ready decision's consumption has already taken effect, even when it consumes only part of the selected bytes or none of them; consumption and changes to decoder state are not rolled back |

Whether decoding can continue depends on the source and format. A zero-byte read
appends no input and enters EOF handling.

## Encoding and flushing

`Encoder::encode` accepts any `Write` destination, including `BufWriter`,
`Vec<u8>`, and files. An encoder can use `write_all` to write complete encoded
bytes and leave flushing to the caller. An Item may contain borrowed fields,
and the same encoder can implement `Encoder<Item>` for multiple input types.

`FramedWrite` buffers encoded output. A successful `write` means the bytes have
been accepted; some may already have reached the destination. `with_capacity`
sets buffer capacity, and zero disables buffering. Large Items are not limited
by that capacity. `capacity()` returns the capacity; `buffer()` returns bytes
waiting to be written.

`flush` writes pending bytes and flushes the destination. Interrupted writes of
buffered bytes are retried; errors from the destination's `flush` are returned
without retrying. After an error, another call attempts to write any remaining
buffered bytes and flush the destination again. An encoding error may leave part
of an Item written or buffered; bytes not accepted by the writer are not saved
automatically. Retrying the whole Item may duplicate output, so recovery depends
on the encoder.

Call `flush` and handle its result before dropping the writer. Dropping attempts
to write pending output and ignores errors, but does not flush the destination.
A successful `flush` empties the buffer and flushes the destination. If no more
bytes are written afterward, there is no buffered output to write on drop.
If the destination panics while writing, dropping the writer during unwinding
does not retry that write.

`FramedRead::reader()` and `FramedWrite::writer()` return shared references to
the source and destination. `decoder()` and `encoder()` return shared
references to the format implementations. `FramedRead::buffer()` returns the
buffered input; `FramedWrite::buffer()` returns output waiting to be written.

## Validation

```text
cargo fmt --all -- --check
cargo test --offline
cargo clippy --offline --all-targets --all-features -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --offline --no-deps
```

## License

[MIT](LICENSE).
