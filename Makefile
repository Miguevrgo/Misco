.PHONY: kickstart clean

kickstart:
	@echo "Creating pythnic environment..."
	@python3 -m venv .venv
	@echo "Installing dependencies..."
	@.venv/bin/pip install yfinance 
	@echo "Downloading data..."
	@.venv/bin/python3 data.py 
	@echo "Done."

clean:
	@echo "Cleaning up..."
	@rm -rf .venv
	@rm -rf data/
	@echo "Done."
