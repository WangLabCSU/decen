# Benchmarks

Run from the repository root:

```sh
cargo bench --bench read
cargo bench --bench write
```

Read compares `FramedRead`, `BufReader`, and direct `BytesMut` buffering.
Write compares `FramedWrite` and `BufWriter`.

| Workload | Item sizes (20 B steps) |
| -------- | ---------------------- |
| Small    | 20–500 B               |
| Medium   | 520–1000 B             |
| Large    | 1020–1500 B            |

Read sizes include the final newline. Write sizes are the bytes written per
Item. Initial buffer capacity is 8 KiB.

Reading the tables:

- `call` shows the operation being measured.
- `median ns/Item`, `min`, and `max` are nanoseconds per Item; lower is faster.
  Green marks the lowest median within each comparison group.
- Read rows are grouped by source and result type: borrowed slice, `Vec<u8>`,
  or `BytesMut`. Write rows are grouped by sink.
- `Lines` keeps partial lines in the reader; `AccumulatingLines` stores them
  in its own `Vec`. `false` transfers that Vec; `true` copies it and reuses it.

File reads use cached data. File writes include flushing but not `sync_all()`.
