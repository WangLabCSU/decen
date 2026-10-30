# Benchmarks

Run from the repository root:

```sh
cargo bench --bench read
cargo bench --bench write
```

To benchmark the nightly read path:

```sh
cargo +nightly bench --bench read --features nightly
```

By default, `FramedRead` and direct `BytesMut` use `Read::read` with initialized
destinations. `FramedRead` tracks initialized spare space; direct `BytesMut`
initializes its tail on each refill. With `nightly`, both use `Read::read_buf`.

Read compares `FramedRead`, `BufReader`, and direct `BytesMut` buffering.
`FramedRead` retains partial frames separately and refills whole chunks.
`decode_view()` joins crossing frames; `decode_owned()` copies the selected
parts directly into its result.
Write compares `FramedWrite` and `BufWriter`.

| Workload              | Item sizes             |
| --------------------- | ---------------------- |
| Small                 | 20–500 B, step 20 B    |
| Medium                | 520–1000 B, step 20 B  |
| Large                 | 1020–1500 B, step 20 B |
| Oversized (read only) | See sizes below        |

Oversized Items repeat 16 KiB + 3 B, 64 KiB + 5 B, 100 KiB + 7 B,
256 KiB + 11 B, and 1 MiB + 13 B.

Read sizes include the final newline. Write sizes are the bytes written per
Item. Initial buffer capacity is 8 KiB.

To check read results without timing, append `-- --test` to either read command.

Reading the tables:

- `call` shows the operation being measured.
- `median ns/Item`, `min`, and `max` are nanoseconds per Item; lower is faster.
  Green marks the lowest median within each comparison group.
- Read tables also show `median MiB/s`; higher throughput is better.
- Read rows are grouped by source and result type: borrowed slice, `Vec<u8>`,
  or owned bytes. Both owned-byte paths return one contiguous `BytesMut`.
  Write rows are grouped by sink.
- `Lines` keeps partial lines in the reader; `AccumulatingLines` stores them
  in its own `Vec`. `false` transfers that Vec; `true` copies it and reuses it.

File reads use cached data. File writes include flushing but not `sync_all()`.
