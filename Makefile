.PHONY: kickstart train test predict stonks graph clean

# First-time setup: create venv, install deps, download stock data
kickstart:
	@echo "🐍 Creating Python environment..."
	@python3 -m venv .venv
	@echo "📦 Installing dependencies..."
	@.venv/bin/pip install yfinance matplotlib numpy pandas
	@echo "📥 Downloading stock data..."
	@.venv/bin/python3 scripts/download_data.py
	@mkdir -p models
	@echo "✅ Done. Run 'make train' to start training."

# Train the network on LEARN tickers (BP, ENI, EQUINOR)
train:
	@mkdir -p models
	@RUSTFLAGS="-C target-cpu=native" cargo run --release --features train

# Evaluate on TEST tickers and write predictions.csv
test:
	@RUSTFLAGS="-C target-cpu=native" cargo run --release --features test

# Run a single prediction on a TEST ticker
predict:
	@RUSTFLAGS="-C target-cpu=native" cargo run --release --features predict

# Simulate a buy/sell strategy using predictions
stonks:
	@RUSTFLAGS="-C target-cpu=native" cargo run --release --features stonks

# Plot predictions.csv (run after 'make test')
graph:
	@.venv/bin/python3 scripts/plot_results.py

# Remove generated artifacts
clean:
	@echo "🧹 Cleaning up..."
	@rm -rf .venv
	@rm -f data/*.csv
	@rm -f predictions.csv
	@rm -rf models/checkpoint*
	@echo "✅ Done."
