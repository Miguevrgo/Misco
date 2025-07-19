use entry::{Date, StockEntry};
use stock::Stock;

mod entry;
mod stock;

fn main() {
    println!("\x1b[1;33m╔══════════════════════════════════════════════╗\x1b[0m");
    println!(
        "\x1b[1;33m║\x1b[1;34m           Misco Stock Predictor V0           \x1b[1;33m║\x1b[0m"
    );
    println!(
        "\x1b[1;33m║\x1b[1;34m    Miguel Angel De la Vega | Gonzalo Olmo    \x1b[1;33m║\x1b[0m"
    );
    println!("\x1b[1;33m╚══════════════════════════════════════════════╝\x1b[0m");
    let mut stock = Stock::new("APPL", "Apple");
    for i in 0..10 {
        stock.push(
            Date::from(format!("28-02-200{i}").as_str()).unwrap(),
            StockEntry::new(222220 + i * 1500, i, 11340, 1234),
        );
    }
    print!("{stock}");
}
