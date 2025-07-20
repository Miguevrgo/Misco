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
            return Err("Expected date format: YYYY-MM-DD");
        }

        let year = parts[0].parse::<u16>().map_err(|_| "Invalid year")?;
        if year < 1970 {
            return Err("Year must be 1970 or later");
        }
        let month = parts[1].parse::<u8>().map_err(|_| "Invalid month")?;
        if !(1..=12).contains(&month) {
            return Err("Month must be between 1 and 12");
        }
        let day = parts[2].parse::<u8>().map_err(|_| "Invalid day")?;
        if !(1..=31).contains(&day) {
            return Err("Day must be between 1 and 31");
        }

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
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

/// Contains price information of a stock for a given `Date`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StockEntry {
    /// Opening price at the start of the day
    open: f32,
    /// Highest price reached during the day
    high: f32,
    /// Lowest price during the day
    low: f32,
    /// Price at market close
    close: f32,
}

impl StockEntry {
    /// Creates a new [`StockEntry`] instance with given values.
    ///
    /// All prices must be expressed in cents.
    pub fn new(open: f32, high: f32, low: f32, close: f32) -> Self {
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
            self.open, self.high, self.low, self.close,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_date_from_csv_valid() {
        let date = Date::from_csv("2024-07-19 00:00:00-00:00").unwrap();
        assert_eq!(date, Date::new(2024, 7, 19));
    }
    #[test]
    fn test_date_from_csv_invalid() {
        assert!(Date::from_csv("2024-07-19").is_err());
        assert!(Date::from_csv("2024-07 00:00:00").is_err());
        assert!(Date::from_csv("invalid-date").is_err());
    }
    #[test]
    fn test_date_from_invalid_format() {
        assert!(Date::from("19/07/2024").is_err());
        assert!(Date::from("2024-07").is_err());
        assert!(Date::from("invalid").is_err());
        assert!(Date::from("34-3-2000").is_err());
        assert!(Date::from("06-13-2000").is_err());
        assert!(Date::from("06-11-1800").is_err());
    }
    #[test]
    fn test_date_from_valid() {
        let date = Date::from("2024-07-19").unwrap();
        assert_eq!(date, Date::new(2024, 7, 19));
    }
    #[test]
    fn test_date_display() {
        let date = Date::new(2024, 7, 19);
        assert_eq!(format!("{date}"), "2024-07-19");
    }
}
