use crate::portfolio::{Data, StockData};

use super::entry::{Date, StockEntry};

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

    // TODO: If end is not found, find previous valid date @Gonzalo :)
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
        StockData::new(
            self.entries[begin..end-1].to_vec(),
            self.entries[end].1.close,
        )
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
