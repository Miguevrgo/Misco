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

    let asdf = "2014-07-20 00:00:00:04:00";
    let fecha:Date = Date::from_csv(asdf).unwrap();
    println!("{}",fecha);
}
