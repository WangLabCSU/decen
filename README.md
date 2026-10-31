# decen

[<img alt="github" src="https://img.shields.io/badge/github-WangLabCSU/decen-8da0cb?style=for-the-badge&labelColor=555555&logo=github" height="20">](https://github.com/WangLabCSU/decen)
[<img alt="crates.io" src="https://img.shields.io/crates/v/decen.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/decen)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-decen-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs" height="20">](https://docs.rs/decen)
[<img alt="build status" src="https://img.shields.io/github/actions/workflow/status/WangLabCSU/decen/ci.yml?branch=main&style=for-the-badge" height="20">](https://github.com/WangLabCSU/decen/actions?query=branch%3Amain)

**decen** takes its name from **decoder** and **encoder**. It provides traits
for defining frame boundaries, decoding frames, and encoding values, together
with buffered readers and writers that use those implementations.

The default configuration supports stable Rust. The `nightly` feature switches
`FramedRead` to `Read::read_buf` and requires nightly Rust. Sources with their
own `read_buf` implementation can avoid initializing the destination before
reading.

To enable the nightly read path, use
`cargo +nightly build --features nightly`.

The design is inspired by
[`tokio-util::codec`][codec],
which adapts asynchronous I/O into `Stream` and `Sink` transports. decen focuses
on efficient frame processing through `std::io::Read` and `std::io::Write`,
emphasizing memory reuse and control over the ownership of decoded data.

Borrow framed bytes for immediate processing, retain them without copying, or
build an owned Item containing only the data you need. A single framing
implementation can support all three modes.

You implement the format's framing, decoding, or encoding traits as needed. A
decoded result or a value supplied to an encoder is called an **Item**. Input
need not be UTF-8, and reading and writing can be used independently.

Framers and decoders can be stateful. `FramedRead` uses the same format
instance across calls, allowing framing and decoding to share parsing state
and continue processing partial input as more bytes arrive.

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

Each successful decode call returns one Item or `None` when no further Item is
available at EOF. A later call can resume if the source provides more input.
The framer determines how many input bytes are consumed; this can differ from
the number of bytes selected for decoding.

## Encoding

`FramedWrite` passes each Item to an `Encoder` and buffers the encoded bytes
for a `Write` destination. Call `flush()` and handle its result before dropping
the writer, since errors during drop are ignored.

`Encoder::encode` also accepts any `Write` destination directly, including
`Vec<u8>` and files. Use `write_all` to write complete encoded bytes and leave
flushing to the caller. An Item may contain borrowed fields, and the same
encoder can implement `Encoder<Item>` for multiple input types.

## Implementing a format

To decode with `FramedRead`, implement `Framer` and at least one of
`BytesDecoder`, `ViewDecoder`, or `OwnedDecoder` on the same type. Frame
boundaries can depend on format state, such as quoting or length prefixes.
Implement `Encoder<Item>` to write an Item's encoded bytes.

`Framer::frame` returns a `Frame` with a consumption setting and a `FrameAction`:
`Ready(start, end)` decodes one Item, `Reframe` immediately frames the remaining
input, and `NeedMore` attempts a read before framing again. Inspect the complete
decision through `consume()` and `action()`.

- `ready(range)`: Requests selection of the range and consumption through its
  end by default.
- `reframe()`: Creates a `Reframe` decision with zero consumption.
- `need_more()`: Creates a `NeedMore` decision with zero consumption.
- `with_consume(consume)`: Returns the decision with an explicit consumption count,
  independently of the selection, allowing adjacent frames to overlap.
- `with_action(action)`: Returns the decision with another action, keeping any
  explicit consumption count.

Use `FrameAction::ready(range)` to construct a ready action, including when
replacing an action with `with_action`.

For example, `ready(1..4)` consumes 4 bytes, and `ready(..)` consumes all input
supplied to the framer when the reader applies the decision. `consume()` returns
`None` for the action's default, or
`Some(n)` after setting an explicit count. Set it separately when needed:

```rust
use decen::read::Frame;

let frame = Frame::need_more().with_consume(4);

let frame = Frame::ready(..4).with_consume(0); // Retain all input.
```

`Ready` works with all three decoder traits. `decode_bytes()` transfers the
selection without copying when consumption covers the prefix through its end;
otherwise, it copies the selection into a new `BytesMut`. `decode_view()` and
`decode_owned()` always pass a slice without this copy.

The reader interprets selections relative to the input before consumption and
rejects ranges or consumption counts outside that input. Empty Items and zero
consumption are allowed. Reading or decoding errors do not undo consumption.
For example, selecting `1..4` and requesting consumption of 5 bytes selects
`abc` from `<abc>tail` and leaves `tail`.

The decoder receives framed bytes. A view Item can borrow fields directly;
converted fields, such as unescaped text, can use storage owned by the Item.
`BytesDecoder` can return `BytesMut` directly or freeze it into `Bytes`. A
record can also hold the frame and expose its fields as slices.

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

    unsafe fn frame(&mut self, src: &[u8]) -> io::Result<Frame> {
        Ok(if src.is_empty() {
            Frame::need_more()
        } else {
            Frame::ready(..1)
        })
    }
}

impl BytesDecoder for ByteCodec {
    type Item = BytesMut;

    unsafe fn decode_bytes(&mut self, frame: BytesMut) -> io::Result<Self::Item> {
        Ok(frame)
    }
}

impl ViewDecoder for ByteCodec {
    type Item<'a> = &'a [u8];

    unsafe fn decode_view<'a>(
        &mut self,
        frame: &'a [u8],
    ) -> io::Result<Self::Item<'a>> {
        Ok(frame)
    }
}

impl OwnedDecoder for ByteCodec {
    type Item = Vec<u8>;

    unsafe fn decode_owned(&mut self, frame: &[u8]) -> io::Result<Self::Item> {
        Ok(frame.to_vec())
    }
}

impl<T: AsRef<[u8]>> Encoder<T> for ByteCodec {
    type Error = io::Error;

    fn encode<W: io::Write + ?Sized>(
        &mut self,
        item: T,
        writer: &mut W,
    ) -> io::Result<()> {
        let bytes = item.as_ref();
        if bytes.len() != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "expected one byte",
            ));
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

[codec]: https://docs.rs/tokio-util/latest/tokio_util/codec/index.html
