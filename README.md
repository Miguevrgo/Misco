# Misco
Misco is a collaborative project by gonzaloolmo19 and miguevrgo to build a performant, reproducible stock market forecasting pipeline in Rust. The initial goal is to implement a customizable neural network that learns from historical OHLCV (Open, High, Low, Close, Volume) data (and optionally engineered features) for a training universe of tickers, then evaluates generalization on a separate test ticker set.

# Dataset
*Training*
Repsol S.A.
BP PLC
Shell PLC
*Test*
TotalEnergies SE
Equinor ASA
Eni S.p.A.
<div align="center">
  <img src="stock_prediction_plot.png" alt="test" />
</div>

The plot above shows a comparison between predicted and real stock prices for BP over a certain time window. While the model captures some trends, there is still room for improvement before it can be relied upon.

## Architecture
- Input: 512 days of historical data
- Network: 3 fully connected layers, each with 512 neurons
- Activation: ReLU
- Training: Gradient descent with backpropagation
- Framework: 100% handcrafted in Rust (no external ML libraries)

## Disclaimer
⚠️ This project is experimental and should not be used for real-world investment decisions. It is a research prototype.

## Future Work
We are actively working on improving the system (to learn). Planned improvements include:
- ✅ Debugging and fixing any remaining logic bugs
- ⚡ Speeding up training time (e.g. through SIMD or parallelization)
- Using GPU acceleration (CUDA)
- Expanding to deeper and wider networks
- 🕒 Transitioning to temporal models like RNNs, LSTMs, or GRUs
- Training on the full dataset and evaluating generalization on unseen tickers
- Exploring attention-based models such as Transformers
