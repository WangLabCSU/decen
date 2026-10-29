use std::io::{self, IsTerminal};

pub struct Row<const N: usize> {
    pub cells: [String; N],
    pub fastest: bool,
}

// All cells are ASCII. Keep ANSI escapes outside the padded cells so color
// does not change column widths. Formatting happens after timing.
pub fn print<const N: usize>(
    title: &str,
    headers: [&str; N],
    rows: &[Row<N>],
    numeric_start: usize,
    group_columns: usize,
) {
    let color = io::stdout().is_terminal()
        && std::env::var_os("NO_COLOR").is_none_or(|value| value.is_empty());
    if color {
        println!("\n\x1b[1;36m## {title}\x1b[0m\n");
    } else {
        println!("\n## {title}\n");
    }
    let widths: [usize; N] = std::array::from_fn(|column| {
        rows.iter()
            .map(|row| row.cells[column].len())
            .chain([headers[column].len(), 3])
            .max()
            .unwrap()
    });
    let print_row = |cells: [&str; N], header: bool, fastest: bool| {
        print!("|");
        for (column, cell) in cells.into_iter().enumerate() {
            let width = widths[column];
            let style = if color && header {
                "\x1b[1m"
            } else if color && fastest && column == numeric_start {
                "\x1b[32m"
            } else {
                ""
            };
            let reset = if style.is_empty() { "" } else { "\x1b[0m" };
            if column >= numeric_start {
                print!(" {style}{cell:>width$}{reset} |");
            } else {
                print!(" {style}{cell:<width$}{reset} |");
            }
        }
        println!();
    };
    print_row(headers, true, false);
    let separators: [String; N] = std::array::from_fn(|column| {
        if column >= numeric_start {
            format!("{}:", "-".repeat(widths[column] - 1))
        } else {
            "-".repeat(widths[column])
        }
    });
    print_row(
        std::array::from_fn(|column| separators[column].as_str()),
        false,
        false,
    );
    for (index, row) in rows.iter().enumerate() {
        if index > 0 && row.cells[..group_columns] != rows[index - 1].cells[..group_columns] {
            print_row(
                std::array::from_fn(|column| separators[column].as_str()),
                false,
                false,
            );
        }
        print_row(
            std::array::from_fn(|column| row.cells[column].as_str()),
            false,
            row.fastest,
        );
    }
}
