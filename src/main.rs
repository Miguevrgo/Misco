use entry::Date;
use ndarray::Array1;
use network::Network;
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
    for (ticker, name) in LEARN_TICKER[0..1].iter().zip(LEARN_NAME[0..1].iter()) {
        let filename = format!("data/{ticker}.csv");
        let path = Path::new(&filename);
        portfolio.load_stock(ticker, name, path);
    }

    let training_data = portfolio.get_data(
        &LEARN_TICKER,
        Date::new(2015, 7, 28),
        Date::new(2025, 6, 20),
    );
    let mut n_network = Network::new(365, [512, 512].to_vec());

    let network = Network::load_from_file("./data/network.bin").unwrap();
    let test_data =
        portfolio.get_data(&LEARN_TICKER, Date::new(2024, 1, 5), Date::new(2025, 6, 20));

    use ndarray::Array1;

    let test_entry = &test_data.data[0];
    assert_eq!(test_entry.training_input.len(), 365);

    let input: Array1<f32> = Array1::from(
        test_entry
            .training_input
            .iter()
            .map(|e| e.close)
            .collect::<Vec<f32>>(),
    );

    let prediction = network.feed_forward(&input)[[0]];
    let prediction_ini = n_network.feed_forward(&input)[[0]];
    let real = test_entry.real_value;

    println!(
        "\x1b[1;32mPrediction: {:.3} €\n\x1b[1;32mPrediction_no_train: {:.3} €\nReal value: {:.3} €\nError: {:.3} €\x1b[0m",
        prediction,
        prediction_ini,
        real,
        (prediction - real).abs()
    );
}
