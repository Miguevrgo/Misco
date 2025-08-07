use entry::Date;
use ndarray::Array1;
use network::Network;
use portfolio::Portfolio;
use std::fs::File;
use std::io::Write;
use std::path::Path;

mod entry;
mod network;
mod portfolio;
mod stock;

#[allow(dead_code)]
const TICKERS: [&str; 6] = ["BP", "E", "EQNR", "REPYF", "TTE", "SHEL"];

#[allow(dead_code)]
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

#[cfg(feature = "train")]
fn train() {
    use network::Activation;

    let mut portfolio = Portfolio::new();
    for (ticker, name) in LEARN_TICKER.iter().zip(LEARN_NAME.iter()) {
        let filename = format!("data/{ticker}.csv");
        let path = Path::new(&filename);
        portfolio.load_stock(ticker, name, path);
    }

    let mut training_data = portfolio.get_data(
        &LEARN_TICKER,
        Date::new(2008, 7, 28),
        Date::new(2024, 6, 20),
    );
    training_data.normalize();
    let mut network = Network::new(512, [256, 256, 256].to_vec(), Activation::ReLU);
    network.sgd(0.005, 200, 32, training_data);
    network.save_to_file("./data/networks/network.bin").unwrap();
}

#[cfg(feature = "predict")]
fn predict() {
    let mut portfolio = Portfolio::new();
    for (ticker, name) in LEARN_TICKER[0..1].iter().zip(LEARN_NAME[0..1].iter()) {
        let filename = format!("data/{ticker}.csv");
        let path = Path::new(&filename);
        portfolio.load_stock(ticker, name, path);
    }

    let network = Network::load_from_file("./data/networks/network.bin").unwrap();
    let mut test_data =
        portfolio.get_data(&LEARN_TICKER, Date::new(2023, 6, 6), Date::new(2025, 6, 20));
    test_data.normalize();

    let test_entry = &test_data.data[0];
    assert_eq!(test_entry.training_input.len(), 512);

    let input: Array1<f32> = Array1::from(
        test_entry
            .training_input
            .iter()
            .map(|e| e.close)
            .collect::<Vec<f32>>(),
    );

    let prediction = test_entry.denormalize(network.feed_forward(&input)[[0]]);
    let real = test_entry.denormalize(test_entry.real_value);

    println!(
        "\x1b[1;32mPrediction: {:.3} €\n\x1b[1;32m\nReal value: {:.3} €\nError: {:.3} €\x1b[0m",
        prediction,
        real,
        (prediction - real).abs()
    );
}

#[cfg(feature = "test")]
fn test() {
    let mut portfolio = Portfolio::new();
    for (ticker, name) in LEARN_TICKER.iter().zip(LEARN_NAME.iter()) {
        let filename = format!("data/{ticker}.csv");
        let path = Path::new(&filename);
        portfolio.load_stock(ticker, name, path);
    }

    let network = Network::load_from_file("./data/networks/network.bin").unwrap();
    const CHUNK_SIZE: usize = 512;
    let num_predictions = 300;

    let stock = portfolio.stock(LEARN_TICKER[0]).expect("Stock not found");

    let total_entries = stock.entries.len();
    if total_entries < CHUNK_SIZE {
        panic!("Not enough historical data for prediction");
    }

    let mut file = File::create("predictions.csv").unwrap();
    writeln!(file, "Date,Predicted Value,Real Value,Error").unwrap();
    for i in (0..num_predictions).rev() {
        let entries =
            &stock.entries[(stock.entries.len() - CHUNK_SIZE - i - 1)..stock.entries.len() - i - 1];

        let mut test_data =
            portfolio.get_data(&LEARN_TICKER, entries[0].0, entries.last().unwrap().0);
        test_data.normalize();

        let test_entry = &test_data.data[0];
        assert_eq!(test_entry.training_input.len(), 512);

        let input: Array1<f32> = Array1::from(
            test_entry
                .training_input
                .iter()
                .map(|e| e.close)
                .collect::<Vec<f32>>(),
        );

        let prediction = test_entry.denormalize(network.feed_forward(&input)[[0]]);
        let real = test_entry.denormalize(test_entry.real_value);
        writeln!(
            file,
            "{},{},{},{}",
            entries.last().unwrap().0,
            prediction,
            real,
            (prediction - real).abs()
        )
        .unwrap();
    }
}

#[cfg(feature = "stonks")]
fn stonks() {
    let mut portfolio = Portfolio::new();
    let mut funds = 1000.0;
    let initial_funds = funds;
    let mut shares = 0.0;
    for (ticker, name) in LEARN_TICKER[0..1].iter().zip(LEARN_NAME[0..1].iter()) {
        let filename = format!("data/{ticker}.csv");
        let path = Path::new(&filename);
        portfolio.load_stock(ticker, name, path);
    }

    let network = Network::load_from_file("./data/networks/network.bin").unwrap();
    const CHUNK_SIZE: usize = 512;
    let num_predictions = 300;

    let stock = portfolio.stock(LEARN_TICKER[0]).expect("Stock not found");

    let total_entries = stock.entries.len();
    if total_entries < CHUNK_SIZE {
        panic!("Not enough historical data for prediction");
    }

    let mut file = File::create("predictions.csv").unwrap();
    writeln!(file, "Date,Predicted Value,Real Value,Error").unwrap();
    let mut last_real: f32 = 0.0;
    for i in (0..num_predictions).rev() {
        let entries =
            &stock.entries[(stock.entries.len() - CHUNK_SIZE - i - 1)..stock.entries.len() - i - 1];

        let mut test_data =
            portfolio.get_data(&LEARN_TICKER, entries[0].0, entries.last().unwrap().0);
        test_data.normalize();

        let test_entry = &test_data.data[0];
        assert_eq!(test_entry.training_input.len(), 512);

        let input: Array1<f32> = Array1::from(
            test_entry
                .training_input
                .iter()
                .map(|e| e.close)
                .collect::<Vec<f32>>(),
        );

        let prediction = test_entry.denormalize(network.feed_forward(&input)[[0]]);
        let real = test_entry.denormalize(test_entry.real_value);

        if i == num_predictions - 1 {
            last_real = real;
            continue;
        }

        println!("Today prize: {last_real}");
        println!("Tomorrow prediction: {prediction}");

        if i == 0 {
            funds += shares * last_real;
            shares = 0.0;
            println!("Simulation ended");
            println!("Funds: ${funds}");
            println!("Performance: {}%", ((funds/initial_funds)-1.0)*100.0);

        } else if prediction < last_real && shares != 0.0 {
            funds = last_real * shares;
            shares = 0.0;
            println!("Selling all shares");
            println!("Funds: ${funds}");
            println!("Shares: {shares}");
        } else if prediction > last_real && funds > 0.0 {
            shares = funds / last_real;
            funds = 0.0;
            println!("Buying shares");
            println!("Funds: ${funds}");
            println!("Shares: {shares}");
        }
        last_real = real;
        println!("---")
    }
}

fn main() {
    print_header();
    #[cfg(feature = "train")]
    train();
    #[cfg(feature = "predict")]
    predict();
    #[cfg(feature = "test")]
    test();
    #[cfg(feature = "stonks")]
    stonks();
}
