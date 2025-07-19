/// A simple date structure in the format DD-MM-YYYY
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Date {
    /// Full year (e.g. 2024)
    pub year: u16,
    /// Month of the year (1–12)
    pub month: u8,
    /// Day of the month (1–31)
    pub day: u8,
}

impl Date {
    /// Creates a new [`Date`] from numeric day, month and year values.
    pub fn new(year: u16, month: u8, day: u8) -> Self {
        Self { year, month, day }
    }

    /// Parses a [`Date`] from a string in `"DD-MM-YYYY"` format.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the string does not match the format or contains invalid numbers.
    pub fn from(date: &str) -> Result<Self, &'static str> {
        let parts: Vec<&str> = date.split('-').collect();
        if parts.len() != 3 {
            return Err("Expected format: YYYY-MM-DD");
        }

        let year = parts[0].parse::<u16>().map_err(|_| "Invalid year")?;
        let month = parts[1].parse::<u8>().map_err(|_| "Invalid month")?;
        let day = parts[2].parse::<u8>().map_err(|_| "Invalid day")?;

        Ok(Self::new(year, month, day))
    }

    pub fn from_csv(date: &str) -> Result<Self, &'static str> {
        let parts: Vec<&str> = date.split(' ').collect();
        if parts.len() != 2 {
            return Err("Expected date format from CSV: YYYY-MM-DD hh:mm:ss-mm:ss");
        }
        let date = Date::from(parts[0])?;
        Ok(date)
    }
}

impl std::fmt::Display for Date {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:02}-{:02}-{}", self.year, self.month, self.day)
    }
}

/// Contains price information of a stock for a given `Date`.
///
/// All prices are expressed in cents * 10 (i.e. 123456 = 123.456).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StockEntry {
    /// Opening price at the start of the day
    open: u32,
    /// Highest price reached during the day
    high: u32,
    /// Lowest price during the day
    low: u32,
    /// Price at market close
    close: u32,
}

impl StockEntry {
    /// Creates a new [`StockEntry`] instance with given values.
    ///
    /// All prices must be expressed in cents.
    pub fn new(open: u32, high: u32, low: u32, close: u32) -> Self {
        Self {
            open,
            high,
            low,
            close,
        }
    }
}

impl std::fmt::Display for StockEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:>8.3} | {:>8.3} | {:>8.3} | {:>8.3}",
            (self.open as f32) / 1000.0,
            (self.high as f32) / 1000.0,
            (self.low as f32) / 1000.0,
            (self.close as f32) / 1000.0
        )
    }
}
