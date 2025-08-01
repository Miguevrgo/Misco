use super::entry::{Date, StockEntry};
use core::f32;
use std::fmt;

/// Represents the time series of a single stock (e.g. "REPYF", "SHEL")
#[derive(Debug, Clone)]
pub struct Stock {
    /// Ticker symbol (e.g. "AAPL", "TTE", etc.)
    pub ticker: String,
    /// Name of the company (e.g "Apple", "Google")
    pub name: String,
    /// Chronological list of daily stock entries (oldest to newest)
    pub entries: Vec<(Date, StockEntry)>,
}

impl Stock {
    pub fn new(ticker: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            ticker: ticker.into(),
            name: name.into(),
            entries: Vec::new(),
        }
    }

    pub fn push(&mut self, date: Date, entry: StockEntry) {
        self.entries.push((date, entry));
    }

    pub fn data(&self, begin: Date, end: Date) -> StockData {
        let begin = self
            .entries
            .iter()
            .position(|date| date.0 >= begin)
            .expect("Invalid begin date");
        let end = self
            .entries
            .iter()
            .position(|date| date.0 > end)
            .expect("Invalid end date");
        StockData::new(self.entries[begin..end].to_vec(), self.entries[end].1.close)
    }
}

#[derive(Default)]
pub struct StockData {
    pub training_input: Vec<StockEntry>, // Close value for each day in [start, end[
    pub real_value: f32,                 // Close value in end day
    pub max: f32,
    pub min: f32,
}

impl StockData {
    pub fn new(data: Vec<(Date, StockEntry)>, real_value: f32) -> Self {
        let mut input = Vec::with_capacity(data.len());
        data.iter().for_each(|day| input.push(day.1));
        Self {
            training_input: input,
            real_value,
            max: f32::NEG_INFINITY,
            min: f32::INFINITY,
        }
    }

    pub fn normalize(&mut self) {
        let closes: Vec<f32> = self
            .training_input
            .iter()
            .map(|entry| entry.close)
            .collect();

        let min = closes
            .iter()
            .fold(f32::INFINITY, |a, &b| a.min(b))
            .min(self.real_value);
        let max = closes
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b))
            .max(self.real_value);

        // NOTE: Division by 0 possible :)
        self.training_input.iter_mut().for_each(|entry| {
            entry.open = (entry.open - min) / (max - min);
            entry.high = (entry.high - min) / (max - min);
            entry.low = (entry.low - min) / (max - min);
            entry.close = (entry.close - min) / (max - min);
        });

        self.real_value = (self.real_value - min) / (max - min);

        self.min = min;
        self.max = max;
    }

    pub fn denormalize(&self, value: f32) -> f32 {
        value * (self.max - self.min) + self.min
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

    pub fn normalize(&mut self) {
        for stock_data in &mut self.data {
            stock_data.normalize();
        }
    }
}

impl std::fmt::Display for Stock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\x1b[1;33m\t\t Ticker: {} [{}]", self.ticker, self.name)?;
        writeln!(
            f,
            "\x1b[1;36m     Date        Open |   󰁝 High |    󰁅 Low |   Close"
        )?;
        for (date, entry) in &self.entries {
            writeln!(f, "\x1b[1;32m󰃭 {date}  {entry}")?;
        }
        Ok(())
    }
}

impl fmt::Display for StockData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.training_input.len();
        let half = n.div_ceil(2);

        writeln!(
            f,
            "\x1b[1;34m╔═════════╦════════════╦═════════╦════════════╗\x1b[0m"
        )?;
        writeln!(
            f,
            "\x1b[1;34m║ Day (#) ║  Close (€) ║ Day (#) ║  Close (€) ║\x1b[0m"
        )?;
        writeln!(
            f,
            "\x1b[1;34m╟─────────║────────────╫─────────╫────────────╢\x1b[0m"
        )?;

        for i in 0..half {
            let left = self.training_input.get(i);
            let right = self.training_input.get(i + half);

            match (left, right) {
                (Some(l), Some(r)) => writeln!(
                    f,
                    "\x1b[1;34m║ \x1b[0m{:>7} \x1b[1;34m║ \x1b[1;37m{:>10.3} \x1b[1;34m║ \x1b[0m{:>7} \x1b[1;34m║ \x1b[0m{:>10.3}\x1b[0m \x1b[1;34m║",
                    i + 1,
                    self.denormalize(l.close),
                    i + half + 1,
                    self.denormalize(r.close),
                )?,
                (Some(l), None) => writeln!(
                    f,
                    "\x1b[1;34m║ \x1b[0m{:>7} \x1b[1;34m║ \x1b[1;37m{:>10.3} \x1b[1;34m║ \x1b[0m{:>7} \x1b[1;34m║ \x1b[0m{:>10.3}\x1b[0m \x1b[1;34m║",
                    i + 1,
                    self.denormalize(l.close),
                    "",
                    ""
                )?,
                _ => break,
            }
        }

        writeln!(
            f,
            "\x1b[1;34m╚═════════╩════════════╩═════════╩════════════╝\x1b[0m"
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "\x1b[1;32m╔═════════════════════════════════════════════╗\n\
             ║  Real Value → {:>26.3} €  ║\n\
             ╚═════════════════════════════════════════════╝\x1b[0m",
            self.denormalize(self.real_value)
        )
    }
}

impl fmt::Display for Data {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, stock_data) in self.data.iter().enumerate() {
            writeln!(
                f,
                "\x1b[1;34m╔═════════════════════════════════════════════╗\n\
                 ║ \x1b[1;34mStockData #{i}\x1b[1;34m                                ║\n\
                 ╚═════════════════════════════════════════════╝\x1b[0m"
            )?;
            writeln!(f, "{stock_data}")?;
        }
        Ok(())
    }
}
