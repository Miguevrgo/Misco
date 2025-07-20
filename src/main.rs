use network::{Layer, Network};
use portfolio::Portfolio;
use std::path::Path;

mod entry;
mod network;
mod portfolio;
mod stock;

const LEARN_TICKER: [&str; 3] = ["BP", "E", "EQNR"];
const LEARN_NAME: [&str; 3] = ["BP", "ENI", "EQUINOR"];
#[allow(dead_code)]
const TEST_TICKER: [&str; 3] = ["REPYF", "SHEL", "TTE"];
#[allow(dead_code)]
const TEST_NAME: [&str; 3] = ["REPSOL", "SHELL", "TOTAL ENERGY"];

fn main() {
    println!("\x1b[1;33m╔══════════════════════════════════════════════╗\x1b[0m");
    println!(
        "\x1b[1;33m║\x1b[1;34m           Misco Stock Predictor V0           \x1b[1;33m║\x1b[0m"
    );
    println!(
        "\x1b[1;33m║\x1b[1;34m    Miguel Angel De la Vega | Gonzalo Olmo    \x1b[1;33m║\x1b[0m"
    );
    println!("\x1b[1;33m╚══════════════════════════════════════════════╝\x1b[0m");

    let mut portfolio = Portfolio::new();
    for (ticker, name) in LEARN_TICKER.iter().zip(LEARN_NAME.iter()) {
        let filename = format!("data/{ticker}.csv");
        let path = Path::new(&filename);
        portfolio.load_stock(ticker, name, path);
    }

    for ticker in LEARN_TICKER {
        if let Some(stock) = portfolio.stock(ticker) {
            println!("{stock}");
        }
    }

    let layer = Layer::new(16, 8);
    #[allow(unused)]
    let network = Network::new(4, 4);
    println!("{layer}")
}
