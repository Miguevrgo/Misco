.PHONY: kickstart clean graph

kickstart:
	@echo "Creating pythnic environment..."
	@python3 -m venv .venv
	@echo "Installing dependencies..."
	@.venv/bin/pip install yfinance matplotlib
	@echo "Downloading data..."
	@.venv/bin/python3 data.py 
	@echo "Done."

graph:
	@.venv/bin/python graph.py

test:
	@RUSTFLAGS="-C target-cpu=native" cargo run --release --features test

train:
	@RUSTFLAGS="-C target-cpu=native" cargo run --release --features train

predict:
	@RUSTFLAGS="-C target-cpu=native" cargo run --release --features predict

clean:
	@echo "Cleaning up..."
	@rm -rf .venv
	@rm -rf data/
	@echo "Done."
