//! Compare FramedWrite with BufWriter using the same items and encoded bytes.
//! Run with `cargo bench --bench write`.

use std::fs::OpenOptions;
use std::hint::black_box;
use std::io::{self, BufWriter, Read, Seek, Write};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use decen::write::{Encoder, FramedWrite};

#[path = "src/table.rs"]
mod table;

const CAPACITY: usize = 8 << 10;
// An even count gives each method the first timing position equally often.
const SAMPLES: usize = 10;
const ROUNDS: usize = 4;

// Both paths write each Item's bytes with a single write_all call.
struct Codec;
impl Encoder<&[u8]> for Codec {
    type Error = io::Error;

    fn encode<W: Write + ?Sized>(&mut self, item: &[u8], writer: &mut W) -> io::Result<()> {
        writer.write_all(item)
    }
}

#[inline(never)]
fn run<W: Write>(out: &mut W, items: &[Vec<u8>], framed: bool) {
    if framed {
        let mut writer = FramedWrite::with_capacity(out, Codec, CAPACITY);
        for item in items {
            writer.write(black_box(item.as_slice())).unwrap();
        }
        writer.flush().unwrap();
    } else {
        let mut writer = BufWriter::with_capacity(CAPACITY, out);
        for item in items {
            let item = black_box(item.as_slice());
            writer.write_all(item).unwrap();
        }
        writer.flush().unwrap();
    }
}

fn compare<W: Write>(
    out: &mut W,
    mut reset: impl FnMut(&mut W),
    sink: &str,
    items: &[Vec<u8>],
) -> Vec<table::Row<5>> {
    let mut times = [Vec::new(), Vec::new()];
    // Alternate order on every sample, including two warm-ups. Time construction,
    // encoding, flushing and destruction. Reset the destination outside timing.
    for sample in 0..SAMPLES + 2 {
        for order in 0..2 {
            let mode = (sample + order) % 2;
            let mut elapsed = 0.0;
            for _ in 0..ROUNDS {
                reset(out);
                let start = Instant::now();
                run(out, items, mode == 0);
                // Observe the result before stopping the timer on both paths.
                black_box(&*out);
                elapsed += start.elapsed().as_secs_f64();
            }
            if sample >= 2 {
                times[mode].push(elapsed * 1e9 / (items.len() * ROUNDS) as f64);
            }
        }
    }
    let calls = ["FramedWrite::write(item)", "BufWriter::write_all(item)"];
    for values in &mut times {
        values.sort_by(f64::total_cmp);
    }
    let medians = times
        .each_ref()
        .map(|values| (values[SAMPLES / 2 - 1] + values[SAMPLES / 2]) / 2.0);
    let best = medians[0].min(medians[1]);
    let mut rows = Vec::new();
    for ((call, values), median) in calls.iter().zip(&times).zip(medians) {
        rows.push(table::Row {
            fastest: median == best,
            cells: [
                sink.to_owned(),
                format!("`{call}`"),
                format!("{median:.2}"),
                format!("{:.2}", values[0]),
                format!("{:.2}", values[SAMPLES - 1]),
            ],
        });
    }
    rows
}

fn workload(file: &mut std::fs::File, items: &[Vec<u8>]) -> Vec<table::Row<5>> {
    // Check exact output for both destinations before timing either path.
    let expected = items.concat();
    // Reuse the destination allocation so Vec growth is not part of the
    // comparison.
    let mut memory = Vec::with_capacity(expected.len());
    let mut file_output = Vec::with_capacity(expected.len());
    for framed in [false, true] {
        memory.clear();
        run(&mut memory, items, framed);
        assert_eq!(memory, expected);

        file.set_len(0).unwrap();
        file.rewind().unwrap();
        run(file, items, framed);
        file.rewind().unwrap();
        file_output.clear();
        file.read_to_end(&mut file_output).unwrap();
        assert_eq!(file_output, expected);
    }
    let mut rows = compare(&mut memory, Vec::clear, "memory", items);
    rows.extend(compare(
        file,
        |file| {
            file.set_len(0).unwrap();
            file.rewind().unwrap();
        },
        "file",
        items,
    ));
    // flush() hands data to the OS; this benchmark does not call sync_all().
    rows
}

fn main() -> io::Result<()> {
    // Avoid files left by interrupted processes that used the same PID.
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path =
        std::env::temp_dir().join(format!("decen-write-bench-{}-{stamp}", std::process::id()));
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)?;
    let workloads: [(&str, Vec<usize>); 3] = [
        ("small-variable", (20..=500).step_by(20).collect()),
        ("medium-variable", (520..=1000).step_by(20).collect()),
        ("large-variable", (1020..=1500).step_by(20).collect()),
    ];
    for (label, sizes) in workloads {
        let mut items = Vec::new();
        let mut bytes = 0;
        while bytes < 1024 * 1024 {
            let size = sizes[items.len() % sizes.len()];
            items.push((0..size).map(|i| ((bytes + i) % 251) as u8).collect());
            bytes += size;
        }
        let rows = workload(&mut file, &items);
        table::print(
            label,
            ["sink", "call", "median ns/Item", "min", "max"],
            &rows,
            2,
            1,
        );
        let mean = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
        println!(
            "\nItem sizes repeat {sizes:?} B. \
             Mean per cycle: {mean:.1} B ({:.2} KiB).",
            mean / 1024.0,
        );
        println!("Initial buffer capacity: {CAPACITY} B.");
    }
    drop(file);
    std::fs::remove_file(path)
}
