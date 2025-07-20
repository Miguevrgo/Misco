use entry::Date;
use portfolio::Portfolio;
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
    println!("\x1b[1;33m╔═════════════════════════════════════════════╗\x1b[0m");
    println!(
        "\x1b[1;33m║\x1b[1;34m           Misco Stock Predictor V0          \x1b[1;33m║\x1b[0m"
    );
    println!(
        "\x1b[1;33m║\x1b[1;34m    Miguel Angel De la Vega | Gonzalo Olmo   \x1b[1;33m║\x1b[0m"
    );
    println!("\x1b[1;33m╚═════════════════════════════════════════════╝\x1b[0m");
}

fn main() {
    print_header();
    let mut portfolio = Portfolio::new();
    for (ticker, name) in TICKERS.iter().zip(TICKERS_NAME.iter()) {
        let filename = format!("data/{ticker}.csv");
        let path = Path::new(&filename);
        portfolio.load_stock(ticker, name, path);
    }

    let training_data = portfolio.get_data(
        &LEARN_TICKER,
        Date::new(2024, 7, 28),
        Date::new(2024, 8, 25),
    );

    println!("{training_data}");
}
