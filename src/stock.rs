use super::entry::{Date, StockEntry};

/// Represents the time series of a single stock (e.g. "REPYF", "SHEL")
#[derive(Debug, Clone)]
pub struct Stock {
    /// Ticker symbol (e.g. "AAPL", "TTE", etc.)
    pub ticker: String,
    /// Chronological list of daily stock entries (oldest to newest)
    pub entries: Vec<(Date, StockEntry)>,
}

impl Stock {
    pub fn new(ticker: impl Into<String>) -> Self {
        Self {
            ticker: ticker.into(),
            entries: Vec::new(),
        }
    }

    pub fn push(&mut self, date: Date, entry: StockEntry) {
        self.entries.push((date, entry));
    }
}

impl std::fmt::Display for Stock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\x1b[1;33mTicker: {}", self.ticker)?;
        for (date, entry) in &self.entries {
            writeln!(f, "{} || {}", date, entry)?;
        }
        Ok(())
    }
}
