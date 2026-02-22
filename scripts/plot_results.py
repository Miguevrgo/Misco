import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    if not all(col in df.columns for col in ['Date', 'Predicted Value', 'Real Value', 'Error']):
        raise ValueError("CSV must contain columns: Date, Predicted Value, Real Value, Error")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def plot_predictions(df: pd.DataFrame, ticker_symbol: str):
    pred_series = df['Predicted Value']
    real_series = df['Real Value']
    error_series = df['Error']

    plt.figure(figsize=(12, 6))
    plt.plot(real_series, label='Real price')
    plt.plot(pred_series, label='Prediction', linestyle='--', color='orange')
    plt.title(f"{ticker_symbol.upper()} - Real vs Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price (EUR)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("docs/prediction_plot.png")
    print("Graph saved as docs/prediction_plot.png")

    plt.figure(figsize=(12, 4))
    plt.plot(error_series, label='Prediction Error', color='red')
    plt.title(f"{ticker_symbol.upper()} - Prediction Error")
    plt.xlabel("Date")
    plt.ylabel("Error (EUR)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("docs/prediction_error.png")
    print("Graph saved as docs/prediction_error.png")

df = load_predictions("predictions.csv")
rmse = np.sqrt((df['Error'] ** 2).mean())
print(f"RMSE: {rmse:.4f} EUR")
plot_predictions(df, "BP")
