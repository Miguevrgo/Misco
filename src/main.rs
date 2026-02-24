use portfolio::Portfolio;

mod display;
mod entry;
mod network;
mod portfolio;
mod stock;

#[cfg(feature = "train")]
const LEARN_TICKER: [&str; 3] = ["BP", "E", "EQNR"];
#[cfg(feature = "train")]
const LEARN_NAME: [&str; 3] = ["BP", "ENI", "EQUINOR"];

#[cfg(any(feature = "test", feature = "predict", feature = "stonks"))]
const TEST_TICKER: [&str; 3] = ["REPYF", "SHEL", "TTE"];
#[cfg(any(feature = "test", feature = "predict", feature = "stonks"))]
const TEST_NAME: [&str; 3] = ["REPSOL", "SHELL", "TOTAL ENERGY"];

#[cfg(any(
    feature = "train",
    feature = "test",
    feature = "predict",
    feature = "stonks"
))]
fn load_portfolio(tickers: &[&str], names: &[&str]) -> Portfolio {
    let mut portfolio = Portfolio::new();
    for (ticker, name) in tickers.iter().zip(names.iter()) {
        let filename = format!("data/{ticker}.csv");
        portfolio.load_stock(ticker, name, std::path::Path::new(&filename));
    }
    portfolio
}

fn print_header() {
    display::print_box(&[
        "Misco Stock Predictor V0",
        "Miguel Angel De la Vega | Gonzalo Olmo",
    ]);
}

#[cfg(feature = "train")]
fn train() {
    use entry::Date;
    use network::{Activation, Network, Optimizer};

    let portfolio = load_portfolio(&LEARN_TICKER, &LEARN_NAME);
    let training_data = portfolio.get_data(
        &LEARN_TICKER,
        Date::new(2008, 7, 28),
        Date::new(2024, 6, 20),
    );
    // NOTE: no global normalize() — build_training_pairs does per-window normalization
    // to match test-time behavior
    let mut network = Network::new(512, vec![256, 256, 1], Activation::ReLU);
    network.train(Optimizer::Adam, 0.001, 200, 32, training_data);
    network.save_to_file("./models/network.bin").unwrap();
}

#[cfg(feature = "predict")]
fn predict() {
    use network::Network;

    let portfolio = load_portfolio(&TEST_TICKER[0..1], &TEST_NAME[0..1]);
    let network = Network::load_from_file("./models/network.bin").unwrap();

    let stock = portfolio.stock(TEST_TICKER[0]).expect("Stock not found");
    let n = stock.entries.len();
    const CHUNK: usize = 512;
    assert!(n > CHUNK, "Not enough data for prediction");

    // Last CHUNK entries as input, entry after them as real value
    let begin = stock.entries[n - CHUNK - 1].0;
    let end = stock.entries[n - 2].0;
    let mut test_data = portfolio.get_data(&[TEST_TICKER[0]], begin, end);
    test_data.normalize();

    let test_entry = &test_data.data[0];
    let input = test_entry.to_input();

    let prediction = test_entry.denormalize(network.feed_forward(&input)[[0]]);
    let real = test_entry.denormalize(test_entry.real_value);

    display::print_box(&[
        &format!("Prediction: {:.3} EUR", prediction),
        &format!("Real value: {:.3} EUR", real),
        &format!("Error:      {:.3} EUR", (prediction - real).abs()),
    ]);
}

#[cfg(feature = "test")]
fn test() {
    use network::Network;
    use std::io::Write;

    let portfolio = load_portfolio(&TEST_TICKER, &TEST_NAME);
    let network = Network::load_from_file("./models/network.bin").unwrap();
    const CHUNK: usize = 512;
    let num_predictions = 300;

    let stock = portfolio.stock(TEST_TICKER[0]).expect("Stock not found");
    let total = stock.entries.len();
    assert!(
        total > CHUNK + num_predictions,
        "Not enough data for {num_predictions} predictions"
    );

    let mut file = std::fs::File::create("predictions.csv").unwrap();
    writeln!(file, "Date,Predicted Value,Real Value,Error").unwrap();

    let mut total_error = 0.0f32;
    let mut total_loss = 0.0f32;
    let mut count = 0usize;

    for i in (0..num_predictions).rev() {
        let start_idx = total - CHUNK - i - 1;
        let end_idx = total - i - 1;
        let entries = &stock.entries[start_idx..end_idx];

        let mut test_data =
            portfolio.get_data(&[TEST_TICKER[0]], entries[0].0, entries.last().unwrap().0);
        test_data.normalize();

        let test_entry = &test_data.data[0];
        let input = test_entry.to_input();

        let output = network.feed_forward(&input)[[0]];
        let prediction = test_entry.denormalize(output);
        let real = test_entry.denormalize(test_entry.real_value);
        let error = (prediction - real).abs();

        total_error += error;
        total_loss += (output - test_entry.real_value).powi(2);
        count += 1;

        writeln!(
            file,
            "{},{prediction},{real},{error}",
            entries.last().unwrap().0,
        )
        .unwrap();
    }

    let avg_mae = total_error / count as f32;
    let avg_loss = total_loss / (2.0 * count as f32);

    display::print_box(&[
        &format!("Test Results ({count} predictions on {})", TEST_TICKER[0]),
        &format!("Average MAE:  {avg_mae:.4} EUR"),
        &format!("Average Loss: {avg_loss:.6}"),
    ]);
}

#[cfg(feature = "stonks")]
fn stonks() {
    use network::Network;
    use std::io::Write;

    let portfolio = load_portfolio(&TEST_TICKER[0..2], &TEST_NAME[0..2]);
    let mut funds = 1000.0f32;
    let initial_funds = funds;
    let mut shares = 0.0f32;

    let network = Network::load_from_file("./models/network.bin").unwrap();
    const CHUNK: usize = 512;
    let num_predictions = 450;

    let stock = portfolio.stock(TEST_TICKER[1]).expect("Stock not found");
    let total = stock.entries.len();
    assert!(
        total > CHUNK + num_predictions,
        "Not enough data for simulation"
    );

    let mut file = std::fs::File::create("predictions.csv").unwrap();
    writeln!(file, "Date,Predicted Value,Real Value,Error").unwrap();
    let mut last_real: f32 = 0.0;

    for i in (0..num_predictions).rev() {
        let start_idx = total - CHUNK - i - 1;
        let end_idx = total - i - 1;
        let entries = &stock.entries[start_idx..end_idx];

        let mut test_data =
            portfolio.get_data(&[TEST_TICKER[1]], entries[0].0, entries.last().unwrap().0);
        test_data.normalize();

        let test_entry = &test_data.data[0];
        let input = test_entry.to_input();

        let prediction = test_entry.denormalize(network.feed_forward(&input)[[0]]);
        let real = test_entry.denormalize(test_entry.real_value);

        if i == num_predictions - 1 {
            last_real = real;
            continue;
        }

        println!("Today price: {last_real:.3}");
        println!("Tomorrow prediction: {prediction:.3}");

        if i == 0 {
            funds += shares * last_real;
            shares = 0.0;
        } else if prediction < last_real && shares != 0.0 {
            funds = last_real * shares;
            shares = 0.0;
            println!("Selling all shares | Funds: ${funds:.2}");
        } else if prediction > last_real && funds > 0.0 {
            shares = funds / last_real;
            funds = 0.0;
            println!("Buying shares | Shares: {shares:.4}");
        }
        last_real = real;
        println!("---");
    }

    let performance = ((funds / initial_funds) - 1.0) * 100.0;
    display::print_box(&[
        "Simulation Complete",
        &format!("Final Funds: ${funds:.2}"),
        &format!("Performance: {performance:.2}%"),
    ]);
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
