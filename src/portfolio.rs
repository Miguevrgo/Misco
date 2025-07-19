use std::path::Path;

use crate::{
    entry::{Date, StockEntry},
    stock::Stock,
};

struct Portfolio {
    pub stocks: Vec<Stock>,
}

impl Portfolio {
    pub fn new() -> Self {
        Self { stocks: Vec::new() }
    }

    pub fn load_stock(ticker: &str, name: &str, path: &Path) {
        let mut rdr = csv::Reader::from_path(path).expect("Invalid path");
        let mut stock = Stock::new(ticker, name);

        for result in rdr.records() {
            let entry = result.unwrap();
            let date = Date::from_csv(entry.get(0).unwrap()).expect("Invalid Date");
            let entry = StockEntry::new(
                entry.get(1).unwrap().parse::<u32>().expect("Invalid open"),
                entry.get(2).unwrap().parse::<u32>().expect("Invalid High"),
                entry.get(3).unwrap().parse::<u32>().expect("Invalid Low"),
                entry.get(4).unwrap().parse::<u32>().expect("Invalid Close"),
            );
            stock.push(date, entry);
        }
    }
}
