pub fn term_width() -> usize {
    terminal_size::terminal_size()
        .map(|(w, _)| w.0 as usize)
        .unwrap_or(80)
}

pub fn print_box(lines: &[&str]) {
    let w = term_width();
    // inner = usable content width; box total = inner + 4 columns
    let inner = w.saturating_sub(4).max(10);

    println!("\x1b[1;33m╔{}╗\x1b[0m", "═".repeat(inner + 2));
    for line in lines {
        let display = if line.len() > inner {
            &line[..inner]
        } else {
            line
        };
        let total_pad = inner - display.len();
        let left_pad = total_pad / 2;
        let right_pad = total_pad - left_pad;
        println!(
            "\x1b[1;33m║\x1b[1;34m {}{}{} \x1b[1;33m║\x1b[0m",
            " ".repeat(left_pad),
            display,
            " ".repeat(right_pad)
        );
    }
    println!("\x1b[1;33m╚{}╝\x1b[0m", "═".repeat(inner + 2));
}
