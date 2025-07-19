/// A simple date structure in the format DD-MM-YYYY
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Date {
    /// Day of the month (1–31)
    pub day: u8,
    /// Month of the year (1–12)
    pub month: u8,
    /// Full year (e.g. 2024)
    pub year: u16,
}

impl Date {
    /// Creates a new [`Date`] from numeric day, month and year values.
    pub fn new(day: u8, month: u8, year: u16) -> Self {
        Self { day, month, year }
    }

    /// Parses a [`Date`] from a string in `"DD-MM-YYYY"` format.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the string does not match the format or contains invalid numbers.
    pub fn from(date: &str) -> Result<Self, &'static str> {
        let parts: Vec<&str> = date.split('-').collect();
        if parts.len() != 3 {
            return Err("Expected format: DD-MM-YYYY");
        }

        let day = parts[0].parse::<u8>().map_err(|_| "Invalid day")?;
        let month = parts[1].parse::<u8>().map_err(|_| "Invalid month")?;
        let year = parts[2].parse::<u16>().map_err(|_| "Invalid year")?;

        Ok(Self::new(day, month, year))
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
