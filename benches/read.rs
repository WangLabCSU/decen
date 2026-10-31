//! Compare complete LF-delimited Items using identical input and initial buffer
//! capacities. Small, medium, and large groups use 20-byte steps up to
//! 500 B, 1000 B, and 1500 B respectively, including LF.
//! A fourth read-only workload uses Items larger than the initial buffer.
//! Run with `cargo bench --bench read`; add `--features nightly` on nightly
//! Rust to use Read::read_buf for FramedRead and direct BytesMut buffering.
//! BufReader::fill_buf already uses read_buf internally on the tested std.

#![cfg_attr(feature = "nightly", feature(core_io_borrowed_buf, read_buf))]

use std::fs::OpenOptions;
use std::hint::black_box;
#[cfg(feature = "nightly")]
use std::io::BorrowedBuf;
use std::io::{self, BufRead, BufReader, Read, Seek, Write};
use std::mem;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use bytes::{Buf, BytesMut};
use decen::read::{
    BytesDecoder, Frame, FrameKind, FramedRead, Framer, OwnedDecoder, Segments, ViewDecoder,
};
use memchr::memchr;

#[path = "src/table.rs"]
mod table;

const CAPACITY: usize = 8 << 10;
// One complete rotation gives every method each timing position once.
const SAMPLES: usize = METHODS.len();
const ROUNDS: usize = 4;
const METHODS: [&str; 13] = [
    "FramedRead::decode_bytes()",
    "Frame::ready_split(..end, 0); FramedRead::decode_view()",
    "Frame::ready_copy(..end, end); FramedRead::decode_view()",
    "FramedRead<_, Lines<false>>::decode_owned()",
    "BufReader::fill_buf() + consume(n)",
    "BufReader::fill_buf() + finish_owned::<true>() + consume(n)",
    "BufReader::read_until(b'\\n', &mut Vec::new())",
    "&BytesMut[..end] + advance(end)",
    "BytesMut[..end].to_vec() + advance(end)",
    "BytesMut::split_to(end)",
    "FramedRead<_, AccumulatingLines<false>>::decode_owned()",
    "FramedRead<_, AccumulatingLines<true>>::decode_owned()",
    "BufReader::fill_buf() + finish_owned::<false>() + consume(n)",
];

// Use the same LF search and include LF in every returned frame.
#[inline(always)]
fn find_line_end(src: &[u8]) -> Option<usize> {
    memchr(b'\n', src).map(|at| at + 1)
}

#[inline(always)]
fn copy_frame(frame: &[u8]) -> Vec<u8> {
    frame.to_vec()
}

// Shared by AccumulatingLines and BufReader. Complete buffered lines require
// one copy. Partial lines either transfer the accumulated Vec or copy it and
// retain scratch capacity, according to the same REUSE choice on both paths.
#[inline(always)]
fn finish_owned<const REUSE: bool>(partial: &mut Vec<u8>, frame: &[u8]) -> Vec<u8> {
    if partial.is_empty() {
        copy_frame(frame)
    } else {
        partial.extend_from_slice(frame);
        if REUSE {
            let item = copy_frame(partial);
            partial.clear();
            item
        } else {
            mem::take(partial)
        }
    }
}

// Remember how much of the pending frame has already been searched for LF.
#[derive(Default)]
struct Lines<const COPY: bool>(usize);
impl<const COPY: bool> Framer for Lines<COPY> {
    type Error = io::Error;
    fn frame(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
        Ok(match self.line_end_input(src) {
            Some(end) => Self::ready(end),
            None => Frame::need_more(0),
        })
    }
}
impl<const COPY: bool> Lines<COPY> {
    #[inline(always)]
    fn line_end_input(&mut self, src: Segments<&[u8]>) -> Option<usize> {
        if let Segments::Contiguous(data) = src {
            return self.line_end(data);
        }
        let suffix = src.slice(self.0..);
        let (prefix, data) = suffix.parts();
        let end =
            find_line_end(prefix).or_else(|| find_line_end(data).map(|end| prefix.len() + end));
        if let Some(end) = end {
            let end = self.0 + end;
            self.0 = 0;
            Some(end)
        } else {
            self.0 = src.len();
            None
        }
    }
    // Share this scanner with the direct BytesMut baseline so only buffering
    // and frame delivery differ, not the LF search or partial-line bookkeeping.
    #[inline(always)]
    fn line_end(&mut self, src: &[u8]) -> Option<usize> {
        // A new frame starts at zero. Keep offset handling on the partial-frame
        // path so complete frames do not repeatedly reset an already-zero scan.
        if self.0 != 0 {
            return self.line_end_partial(src);
        }
        if let Some(end) = find_line_end(src) {
            Some(end)
        } else {
            self.0 = src.len();
            None
        }
    }
    #[inline(always)]
    fn ready(end: usize) -> Frame {
        // Fix the decision for the whole run. Both choices select and consume
        // the same bytes; decode_view borrows them in either case.
        if COPY {
            Frame::ready_copy(..end, end)
        } else {
            Frame::ready_split(..end, 0)
        }
    }

    // Inline so the framing loop can simplify the returned decision's bounds.
    #[inline(always)]
    fn line_end_partial(&mut self, src: &[u8]) -> Option<usize> {
        // Previously searched bytes remain in src across reads. Search only
        // the new suffix, including when a frame spans several reads.
        if let Some(end) = find_line_end(&src[self.0..]) {
            let end = self.0 + end;
            self.0 = 0;
            Some(end)
        } else {
            self.0 = src.len();
            None
        }
    }
}
impl<const COPY: bool> BytesDecoder for Lines<COPY> {
    type Item = BytesMut;
    fn decode_bytes(&mut self, frame: BytesMut) -> io::Result<Self::Item> {
        Ok(frame)
    }
}
impl<const COPY: bool> ViewDecoder for Lines<COPY> {
    type Item<'a> = &'a [u8];
    fn decode_view<'a>(&mut self, frame: &'a [u8]) -> io::Result<Self::Item<'a>> {
        Ok(frame)
    }
}
impl<const COPY: bool> OwnedDecoder for Lines<COPY> {
    type Item = Vec<u8>;
    fn decode_owned(&mut self, frame: Segments<&[u8]>) -> io::Result<Self::Item> {
        Ok(frame.to_vec())
    }
}

// Consume partial lines after storing them in the decoder. This lets the
// reader reuse the entire buffer on the next read. Complete buffered lines
// still go directly to to_vec(); only lines crossing reads use partial.
#[derive(Default)]
struct AccumulatingLines<const REUSE: bool> {
    partial: Vec<u8>,
}

impl<const REUSE: bool> Framer for AccumulatingLines<REUSE> {
    type Error = io::Error;

    fn frame(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
        let end = match src {
            Segments::Contiguous(data) => find_line_end(data),
            Segments::Split(first, second) => {
                find_line_end(first).or_else(|| find_line_end(second).map(|end| first.len() + end))
            }
        };
        if let Some(end) = end {
            Ok(Frame::ready_split(..end, 0))
        } else {
            let (first, second) = src.parts();
            self.partial.extend_from_slice(first);
            self.partial.extend_from_slice(second);
            Ok(Frame::need_more(src.len()))
        }
    }

    fn frame_eof(&mut self, src: Segments<&[u8]>) -> io::Result<Frame> {
        let frame = self.frame(src)?;
        // The default EOF check cannot see bytes already consumed into partial.
        if frame.kind() == FrameKind::NeedMore && !self.partial.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "incomplete line",
            ));
        }
        Ok(frame)
    }
}

impl<const REUSE: bool> OwnedDecoder for AccumulatingLines<REUSE> {
    type Item = Vec<u8>;

    fn decode_owned(&mut self, frame: Segments<&[u8]>) -> io::Result<Self::Item> {
        match frame {
            Segments::Contiguous(data) => Ok(finish_owned::<REUSE>(&mut self.partial, data)),
            Segments::Split(first, second) => {
                self.partial.extend_from_slice(first);
                Ok(finish_owned::<REUSE>(&mut self.partial, second))
            }
        }
    }
}

// Every path presents the same full frame, including LF, to the consumer.
fn observe<const VERIFY: bool>(frame: &[u8], totals: &mut (usize, usize)) {
    if VERIFY {
        assert_eq!(frame.last(), Some(&b'\n'));
        for (offset, &byte) in frame[..frame.len() - 1].iter().enumerate() {
            assert_eq!(byte, ((totals.1 + offset) % 245 + 11) as u8);
        }
    }
    black_box(frame);
    totals.0 += 1;
    totals.1 += frame.len();
}

#[inline(never)]
fn framed<R: Read, const MODE: usize, const COPY: bool, const VERIFY: bool>(
    source: R,
) -> (usize, usize) {
    let mut reader = FramedRead::with_capacity(source, Lines::<COPY>::default(), CAPACITY);
    let mut totals = (0, 0);
    match MODE {
        0 => {
            while let Some(item) = reader.decode_bytes().unwrap() {
                observe::<VERIFY>(&item, &mut totals);
            }
        }
        1 => {
            while let Some(item) = reader.decode_view().unwrap() {
                observe::<VERIFY>(item, &mut totals);
            }
        }
        2 => {
            while let Some(item) = reader.decode_owned().unwrap() {
                observe::<VERIFY>(&item, &mut totals);
            }
        }
        _ => unreachable!(),
    }
    totals
}

#[inline(never)]
fn accumulating<R: Read, const REUSE: bool, const VERIFY: bool>(source: R) -> (usize, usize) {
    let mut reader =
        FramedRead::with_capacity(source, AccumulatingLines::<REUSE>::default(), CAPACITY);
    let mut totals = (0, 0);
    while let Some(item) = reader.decode_owned().unwrap() {
        observe::<VERIFY>(&item, &mut totals);
    }
    totals
}

// MODE 0 borrows complete frames; MODE 1 copies them with to_vec(); MODE 2
// uses read_until with a fresh Vec. REUSE controls the same partial-line
// completion strategy as AccumulatingLines for MODE 1.
#[inline(never)]
fn standard<R: Read, const MODE: usize, const REUSE: bool, const VERIFY: bool>(
    source: R,
) -> (usize, usize) {
    let mut reader = BufReader::with_capacity(CAPACITY, source);
    let mut totals = (0, 0);
    if MODE == 2 {
        loop {
            let mut item = Vec::new();
            if reader.read_until(b'\n', &mut item).unwrap() == 0 {
                break;
            }
            observe::<VERIFY>(&item, &mut totals);
        }
    } else {
        let mut scratch = Vec::new();
        loop {
            let buf = reader.fill_buf().unwrap();
            if buf.is_empty() {
                assert!(scratch.is_empty());
                break;
            }
            if let Some(end) = find_line_end(buf) {
                if MODE == 1 {
                    let item = finish_owned::<REUSE>(&mut scratch, &buf[..end]);
                    observe::<VERIFY>(&item, &mut totals);
                } else if scratch.is_empty() {
                    observe::<VERIFY>(&buf[..end], &mut totals);
                } else {
                    scratch.extend_from_slice(&buf[..end]);
                    observe::<VERIFY>(&scratch, &mut totals);
                    scratch.clear();
                }
                reader.consume(end);
            } else {
                scratch.extend_from_slice(buf);
                let len = buf.len();
                reader.consume(len);
            }
        }
    }
    totals
}

// A direct BytesMut buffer: advance after borrowed/copied frames, or split off
// owned frames. Remember the scanned prefix so partial lines are searched once.
#[inline(never)]
fn bytes_mut<R: Read, const MODE: usize, const VERIFY: bool>(mut source: R) -> (usize, usize) {
    let mut buffer = BytesMut::with_capacity(CAPACITY);
    let mut lines = Lines::<false>::default();
    let mut totals = (0, 0);
    loop {
        if let Some(end) = lines.line_end(&buffer) {
            match MODE {
                0 => {
                    observe::<VERIFY>(&buffer[..end], &mut totals);
                    buffer.advance(end);
                }
                1 => {
                    let item = copy_frame(&buffer[..end]);
                    observe::<VERIFY>(&item, &mut totals);
                    buffer.advance(end);
                }
                2 => observe::<VERIFY>(&buffer.split_to(end), &mut totals),
                _ => unreachable!(),
            }
        } else {
            // Match the initial half-capacity refill target. BytesMut handles
            // reclaiming the prefix removed by advance/split_to itself.
            buffer.reserve(CAPACITY / 2);
            let filled = buffer.len();
            #[cfg(feature = "nightly")]
            let nread = {
                // Use the same read_buf interface as FramedRead.
                let mut dst = BorrowedBuf::from(buffer.spare_capacity_mut());
                source.read_buf(dst.unfilled()).unwrap();
                let nread = dst.len();
                // SAFETY: BorrowedBuf's filled prefix contains nread
                // initialized bytes following the existing input.
                unsafe { buffer.set_len(filled + nread) };
                nread
            };
            #[cfg(not(feature = "nightly"))]
            let nread = {
                // Read::read requires initialized bytes. This direct BytesMut
                // keeps only input in len(), so resize initializes the writable
                // tail on each refill; it has no separate initialization bound.
                buffer.resize(buffer.capacity(), 0);
                let dst = &mut buffer[filled..];
                let nread = source.read(dst).unwrap();
                assert!(
                    nread <= dst.len(),
                    "reader returned more bytes than requested"
                );
                buffer.truncate(filled + nread);
                nread
            };
            if nread == 0 {
                assert!(buffer.is_empty(), "incomplete line");
                break;
            }
        }
    }
    totals
}

fn run<R: Read, const VERIFY: bool>(source: R, mode: usize) -> (usize, usize) {
    match mode {
        0 => framed::<_, 0, false, VERIFY>(source),
        1 => framed::<_, 1, false, VERIFY>(source),
        2 => framed::<_, 1, true, VERIFY>(source),
        3 => framed::<_, 2, false, VERIFY>(source),
        4 => standard::<_, 0, true, VERIFY>(source),
        5 => standard::<_, 1, true, VERIFY>(source),
        6 => standard::<_, 2, true, VERIFY>(source),
        7 => bytes_mut::<_, 0, VERIFY>(source),
        8 => bytes_mut::<_, 1, VERIFY>(source),
        9 => bytes_mut::<_, 2, VERIFY>(source),
        10 => accumulating::<_, false, VERIFY>(source),
        11 => accumulating::<_, true, VERIFY>(source),
        12 => standard::<_, 1, false, VERIFY>(source),
        _ => unreachable!(),
    }
}

fn compare<R: Read + Seek>(
    source: &mut R,
    kind: &str,
    expected: (usize, usize),
    verify_only: bool,
) -> Vec<table::Row<7>> {
    // Verify every byte before timing; timed runs only present each frame to
    // black_box.
    for mode in 0..METHODS.len() {
        source.rewind().unwrap();
        assert_eq!(run::<_, true>(&mut *source, mode), expected);
    }
    if verify_only {
        return Vec::new();
    }
    let mut times: [Vec<f64>; METHODS.len()] = Default::default();
    // Two warm-ups; rotate methods so no method always runs first.
    for sample in 0..SAMPLES + 2 {
        for order in 0..METHODS.len() {
            let mode = (sample + order) % METHODS.len();
            let mut elapsed = 0.0;
            for _ in 0..ROUNDS {
                source.rewind().unwrap();
                let start = Instant::now();
                // Include reader construction, reading, decoding and
                // destruction.
                // Each owned result is dropped before decoding the next Item.
                let totals = run::<_, false>(&mut *source, mode);
                elapsed += start.elapsed().as_secs_f64();
                assert_eq!(totals, expected);
            }
            let elapsed = elapsed * 1e9 / (expected.0 * ROUNDS) as f64;
            if sample >= 2 {
                times[mode].push(elapsed);
            }
        }
    }
    for values in &mut times {
        values.sort_by(f64::total_cmp);
    }
    // Group by ownership: borrowed slices, copied Vecs, or owned BytesMut
    // storage. Both owned-byte paths return one contiguous BytesMut.
    let borrowed_best = [1, 2, 4, 7]
        .into_iter()
        .map(|mode| times[mode][SAMPLES / 2])
        .fold(f64::INFINITY, f64::min);
    let vec_best = [3, 5, 6, 8, 10, 11, 12]
        .into_iter()
        .map(|mode| times[mode][SAMPLES / 2])
        .fold(f64::INFINITY, f64::min);
    let bytes_best = times[0][SAMPLES / 2].min(times[9][SAMPLES / 2]);
    let mut rows = Vec::new();
    // Keep the three Lines variants adjacent without changing timing order.
    for mode in [1, 2, 4, 7, 3, 10, 11, 5, 12, 8, 6, 0, 9] {
        let method = METHODS[mode];
        let values = &times[mode];
        let (item, best) = match mode {
            1 | 2 | 4 | 7 => ("borrowed", borrowed_best),
            0 | 9 => ("owned bytes", bytes_best),
            _ => ("Vec<u8>", vec_best),
        };
        rows.push(table::Row {
            fastest: values[SAMPLES / 2] == best,
            cells: [
                kind.to_owned(),
                item.to_owned(),
                format!("`{method}`"),
                format!("{:.2}", values[SAMPLES / 2]),
                format!("{:.2}", values[0]),
                format!("{:.2}", values[SAMPLES - 1]),
                format!(
                    "{:.2}",
                    expected.1 as f64 / expected.0 as f64 * 1e9
                        / values[SAMPLES / 2]
                        / (1024.0 * 1024.0)
                ),
            ],
        });
    }
    rows
}

fn main() -> io::Result<()> {
    println!(
        "Read API for FramedRead and direct BytesMut: {}",
        if cfg!(feature = "nightly") {
            "Read::read_buf (nightly feature enabled)"
        } else {
            "Read::read (initialized destinations)"
        }
    );
    let verify_only = std::env::args().skip(1).any(|arg| arg == "--test");
    // Exercise repeated accumulation and truncation separately from timing;
    // normal workloads also verify every complete Item before each benchmark.
    let mut check = FramedRead::with_capacity(
        io::Cursor::new(b"abc\nxy\nz"),
        AccumulatingLines::<false>::default(),
        2,
    );
    assert_eq!(check.decode_owned()?.as_deref(), Some(&b"abc\n"[..]));
    assert_eq!(check.decode_owned()?.as_deref(), Some(&b"xy\n"[..]));
    assert_eq!(
        check.decode_owned().unwrap_err().kind(),
        io::ErrorKind::UnexpectedEof
    );

    drop(check);
    // Include a timestamp so an interrupted run's file cannot collide with a
    // later process that receives the same PID.
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path =
        std::env::temp_dir().join(format!("decen-read-bench-{}-{stamp}", std::process::id()));
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)?;
    let workloads: [(&str, Vec<usize>); 4] = [
        ("small-variable", (20..=500).step_by(20).collect()),
        ("medium-variable", (520..=1000).step_by(20).collect()),
        ("large-variable", (1020..=1500).step_by(20).collect()),
        // Avoid exact multiples of the read buffer so boundaries move across
        // reads. These Items exercise repeated refills and buffer growth.
        (
            "oversized-variable",
            vec![
                (16 << 10) + 3,
                (64 << 10) + 5,
                (100 << 10) + 7,
                (256 << 10) + 11,
                (1 << 20) + 13,
            ],
        ),
    ];
    for (label, sizes) in workloads {
        // All lengths include LF. Payload bytes exclude LF and vary by
        // position so verification can detect incorrect frame contents.
        let mut data = Vec::new();
        let mut frames = 0;
        while data.len() < 4 * 1024 * 1024 || frames < 64 {
            let size = sizes[frames % sizes.len()];
            let start = data.len();
            data.extend((start..start + size).map(|i| (i % 245 + 11) as u8));
            *data.last_mut().unwrap() = b'\n';
            frames += 1;
        }
        let expected = (frames, data.len());
        let mut rows = compare(&mut io::Cursor::new(&data), "memory", expected, verify_only);
        file.rewind()?;
        file.set_len(0)?;
        file.write_all(&data)?;
        // Recently written files measure cached I/O, not storage-device
        // latency.
        rows.extend(compare(&mut file, "file-cached", expected, verify_only));
        if verify_only {
            println!(
                "{label}: verified {} methods on memory and file-cached \
                 sources ({frames} Items, {} bytes each pass)",
                METHODS.len(),
                data.len(),
            );
            continue;
        }
        // Keep each highlighted comparison together without changing timing
        // order.
        rows.sort_by_key(|row| (row.cells[0] != "memory", row.cells[1] != "borrowed"));
        table::print(
            label,
            [
                "source",
                "Item",
                "call",
                "median ns/Item",
                "min",
                "max",
                "median MiB/s",
            ],
            &rows,
            3,
            2,
        );
        let mean = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
        println!(
            "\nItem sizes repeat {sizes:?} B, including the final LF. \
             Mean per cycle: {mean:.1} B ({:.2} KiB).",
            mean / 1024.0,
        );
        println!("Initial buffer capacity: {CAPACITY} B.");
        println!(
            "Each sample reads the full dataset {ROUNDS} times. \
             {SAMPLES} recorded samples follow two warm-ups; method order rotates. \
             Rewinding and content verification are outside timing."
        );
        println!(
            "BufReader and FramedRead refill whole chunks; partial frames use scratch. \
             Read request sizes can differ, and no artificial short-read limit is applied."
        );
        println!(
            "Framing: decode_view compares ready_split(..end, 0) with \
             ready_copy(..end, end); both join only frames crossing reads. \
             decode_bytes and decode_owned use ready_split(..end, 0)."
        );
        println!(
            "All explicit framing paths share find_line_end(). Contiguous \
             Vec results use to_vec(); split input copies both parts directly \
             into the final Vec. The borrowed fill_buf path reuses \
             scratch for frames crossing reads. The two owned fill_buf paths \
             share finish_owned::<REUSE>() with AccumulatingLines. read_until \
             uses its own standard-library framing and a fresh Vec per Item."
        );
        println!(
            "BytesMut starts at {CAPACITY} B and calls reserve({}) before each \
             read, then reads into the available tail. Borrowed and Vec paths \
             call advance(end); the BytesMut output path calls split_to(end).",
            CAPACITY / 2,
        );
        println!(
            "AccumulatingLines copies partial lines into its own Vec and \
             consumes that input before reading again. With REUSE=false it \
             moves the completed Vec to the caller; with REUSE=true it calls \
             to_vec() then clear(), retaining scratch capacity. Complete \
             buffered lines use to_vec() in both cases."
        );
    }
    drop(file);
    std::fs::remove_file(path)
}
