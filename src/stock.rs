struct Date {
    day: u32,
    month: u32,
    year: u32,
}

struct Stock {
    date: Date,
    open: u32,
    high: u32,
    low: u32,
    close: u32,
}

impl Stock {
    fn new(date: Date, open: u32, high: u32, low: u32, close: u32) -> Self {
        Self {
            date,
            open,
            high,
            low,
            close,
        }
    }

    fn day(&self) -> u32 {
        self.date.day
    }
    fn month(&self) -> u32 {
        self.date.month
    }
    fn year(&self) -> u32 {
        self.date.year
    }
}

impl Date {
    fn new(day: u32, month: u32, year: u32) -> Self {
        Self { day, month, year }
    }

    fn from(date: &str) -> Option<Self> {
        let parts: Vec<&str> = date.split('-').collect();
        if parts.len() != 3 {
            return None;
        }
        let day = parts[0].parse::<u32>().ok()?;
        let month = parts[1].parse::<u32>().ok()?;
        let year = parts[2].parse()::<u32>.ok()?;
        Some(Self::new(day, month, year))
    }
}