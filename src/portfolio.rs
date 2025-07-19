use std::path::Path;

use crate::{
    entry::{Date, StockEntry},
    stock::Stock,
};

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
                (entry.get(1).unwrap().parse::<f64>().expect("Invalid open") * 1000.0) as u32,
                (entry.get(2).unwrap().parse::<f64>().expect("Invalid high") * 1000.0) as u32,
                (entry.get(3).unwrap().parse::<f64>().expect("Invalid low") * 1000.0) as u32,
                (entry.get(4).unwrap().parse::<f64>().expect("Invalid close") * 1000.0) as u32,
            );
            stock.push(date, entry);
        }

        self.stocks.push(stock);
    }

    /// Returns the stock identified by the ticket
    pub fn stock(&self, ticker: &str) -> Option<&Stock> {
        self.stocks.iter().find(|stock| stock.ticker == ticker)
    }
}
