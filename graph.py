import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

def load_preditions(ruta: str) -> list:
    with open(ruta, 'r') as f:
        content = f.read()
        values = content.strip().split(',')
        return [float(x) for x in values]

def plot_stock_with_predictions(ticker_symbol: str, start_date: str, end_date: str, predictions: list):
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date, auto_adjust=False)

    if stock_data.empty:
        print("No data found for the given dates.")
        return

    fechas = stock_data.index

    if len(predictions) != len(fechas):
        print("Error: number of predictions does not match with number of real values")
        print("Real values:", len(fechas))
        print("Predictions: ", len(predictions))
        return

    pred_series = pd.Series(predictions, index=fechas)

    # Graph
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Close'], label='Real price')
    plt.plot(pred_series, label='Prediction', linestyle='--', color='orange')
    plt.title(f"{ticker_symbol.upper()} - Realprice vs Predition\n{start_date} to {end_date}")
    plt.xlabel("date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("stock_prediction_plot.png")
    print("Graph saved as stock_prediction_plot.png")

predictions = load_preditions("predictions.csv")
plot_stock_with_predictions("BP", "2025-06-23", "2025-07-31", predictions)

