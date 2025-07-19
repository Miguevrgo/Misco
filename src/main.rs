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
    let s: StockEntry = StockEntry::new(10, 10, 10, 1234);
    let mut stock = Stock::new("APPL");
    stock.push(Date::new(20, 10, 2000), s);
    print!("{stock}");
}
