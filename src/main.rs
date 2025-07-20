use entry::Date;
use network::{Layer, Network};
use portfolio::{Data, Portfolio, StockData};
use std::path::Path;

mod entry;
mod network;
mod portfolio;
mod stock;

const TICKERS: [&str; 6] = ["BP", "E", "EQNR", "REPYF", "TTE", "SHEL"];
const TICKERS_NAME: [&str; 6] = ["BP", "ENI", "EQUINOR", "REPSOL", "TOTAL ENERGY", "SHELL"];

const LEARN_TICKER: [&str; 3] = ["BP", "E", "EQNR"];
const LEARN_NAME: [&str; 3] = ["BP", "ENI", "EQUINOR"];

#[allow(dead_code)]
const TEST_TICKER: [&str; 3] = ["REPYF", "SHEL", "TTE"];
#[allow(dead_code)]
const TEST_NAME: [&str; 3] = ["REPSOL", "SHELL", "TOTAL ENERGY"];

fn print_header() {
    println!("\x1b[1;33m╔══════════════════════════════════════════════╗\x1b[0m");
    println!(
        "\x1b[1;33m║\x1b[1;34m           Misco Stock Predictor V0           \x1b[1;33m║\x1b[0m"
    );
    println!(
        "\x1b[1;33m║\x1b[1;34m    Miguel Angel De la Vega | Gonzalo Olmo    \x1b[1;33m║\x1b[0m"
    );
    println!("\x1b[1;33m╚══════════════════════════════════════════════╝\x1b[0m");
}

fn main() {
    print_header();
    let mut portfolio = Portfolio::new();
    for (ticker, name) in TICKERS.iter().zip(TICKERS_NAME.iter()) {
        let filename = format!("data/{ticker}.csv");
        let path = Path::new(&filename);
        portfolio.load_stock(ticker, name, path);
    }

    let mut training_data = Data::new();
    for ticker in &LEARN_TICKER[0..1] {
        if let Some(stock) = portfolio.stock(ticker) {
            training_data.push(stock.data(
                Date::from("2024-08-28").unwrap(),
                Date::from("2024-10-28").unwrap(),
            ));
        }
    }

    println!("{training_data}");
}
