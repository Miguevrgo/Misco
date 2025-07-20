use std::{fmt, path::Path};

use crate::{
    entry::{Date, StockEntry},
    stock::Stock,
};

#[derive(Default)]
pub struct StockData {
    pub training_input: Vec<StockEntry>, // Close value for each day in [start, end[
    pub real_value: f32,                 // Close value in end day
}

impl StockData {
    pub fn new(data: Vec<(Date, StockEntry)>, real_value: f32) -> Self {
        let mut input = Vec::with_capacity(data.len());
        for day in data {
            input.push(day.1);
        }
        Self {
            training_input: input,
            real_value,
        }
    }
}

impl fmt::Display for StockData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Training Input (Close values):")?;
        for (i, entry) in self.training_input.iter().enumerate() {
            writeln!(f, "  Day {:>2}: {entry}", i + 1)?;
        }
        writeln!(
            f,
            "Real Value (Close on target date): {:>8.3}",
            self.real_value
        )
    }
}

pub struct Data {
    pub data: Vec<StockData>,
}

impl Data {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn push(&mut self, stock_data: StockData) {
        self.data.push(stock_data);
    }
}

impl fmt::Display for Data {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, stock_data) in self.data.iter().enumerate() {
            writeln!(f, "──────── StockData #{i} ────────")?;
            writeln!(f, "{stock_data}")?;
        }
        Ok(())
    }
}

pub struct Portfolio {
    stocks: Vec<Stock>,
}

impl Portfolio {
    pub fn new() -> Self {
        Self { stocks: Vec::new() }
    }

    pub fn load_stock(&mut self, ticker: &str, name: &str, path: &Path) {
        let mut rdr = csv::Reader::from_path(path).expect("Invalid path");
        let mut stock = Stock::new(ticker, name);

        for result in rdr.records() {
            let entry = result.unwrap();
            let date = Date::from_csv(entry.get(0).unwrap()).expect("Invalid Date");
            let entry = StockEntry::new(
                entry.get(1).unwrap().parse::<f32>().expect("Invalid open"),
                entry.get(2).unwrap().parse::<f32>().expect("Invalid high"),
                entry.get(3).unwrap().parse::<f32>().expect("Invalid low"),
                entry.get(4).unwrap().parse::<f32>().expect("Invalid close"),
            );
            stock.push(date, entry);
        }

        self.stocks.push(stock);
    }

    pub fn get_data(&self, tickers: &[&str], begin: Date, end: Date) -> Data {
        let mut data = Data::new();
        for stock in &self.stocks {
            if tickers.contains(&stock.ticker.as_str()) {
                data.push(stock.data(begin, end));
            }
        }
        data
    }

    /// Returns the stock identified by the ticket
    pub fn stock(&self, ticker: &str) -> Option<&Stock> {
        self.stocks.iter().find(|stock| stock.ticker == ticker)
    }
}
