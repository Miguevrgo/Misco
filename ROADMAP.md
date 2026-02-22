# Misco Roadmap: Modern AI Improvements

This document outlines concrete, incremental improvements to evolve Misco from a vanilla
feedforward network with SGD into a state-of-the-art time-series forecasting system —
all handcrafted in Rust.

---

## Phase 1 — Training Foundation (Current target)

These are high-impact, low-effort changes to the existing architecture that dramatically
improve convergence and robustness without touching the network topology.

| # | Improvement | Why it matters |
|---|-------------|---------------|
| 1 | **Adam optimizer** | Adaptive per-parameter learning rates. Converges 5-10x faster than vanilla SGD on most problems. Tracks first moment (mean) and second moment (variance) of gradients with bias correction. |
| 2 | **Learning rate scheduling** | Cosine annealing with linear warmup. Prevents early divergence (warmup) and enables fine-grained convergence (annealing). Standard in all modern training pipelines. |
| 3 | **Gradient clipping** | Clips gradients by global L2 norm. Prevents exploding gradients that cause NaN losses, especially important with deep networks and financial data with occasional large moves. |
| 4 | **Huber (Smooth L1) loss** | Quadratic for small errors, linear for large ones. Much more robust to outliers than pure MSE — critical for financial data where a single earnings day can spike 10%+. |
| 5 | **Dropout regularization** | Randomly zeros neurons during training (e.g. p=0.2). Forces the network to learn redundant representations, reducing overfitting on the small training set. Disabled at inference. |

**Expected impact**: Faster convergence, more stable training, better generalization to test tickers.

---

## Phase 2 — Architecture Improvements

Enhance the network topology to better capture the structure of financial time series.

| # | Improvement | Why it matters |
|---|-------------|---------------|
| 6 | **Batch normalization** | Normalizes layer activations to zero mean / unit variance per mini-batch. Stabilizes training, allows higher learning rates, acts as light regularization. |
| 7 | **Residual (skip) connections** | `output = layer(x) + x`. Allows gradients to flow directly through the network, enabling training of much deeper architectures (10+ layers) without vanishing gradients. |
| 8 | **Multi-feature input (OHLCV)** | Currently only Close prices are fed to the network. Using Open, High, Low, Close, and Volume gives the model intraday structure and liquidity information — much richer signal. |
| 9 | **Learned normalization (LayerNorm)** | Unlike BatchNorm, LayerNorm normalizes across features rather than the batch dimension. More stable for sequence models and small batch sizes. |
| 10 | **Early stopping + validation split** | Hold out a portion of training data for validation. Stop training when validation loss stops improving. Prevents overfitting systematically. |

**Expected impact**: Deeper networks, richer inputs, systematic overfitting prevention.

---

## Phase 3 — Temporal Architecture

The fundamental limitation of the current system: a feedforward network treats the 512-day
window as a flat vector with no notion of time ordering. Temporal layers fix this.

| # | Improvement | Why it matters |
|---|-------------|---------------|
| 11 | **LSTM (Long Short-Term Memory)** | Gated recurrent cells with forget/input/output gates. Explicitly designed to capture long-range temporal dependencies — the bread and butter of sequence modeling. |
| 12 | **GRU (Gated Recurrent Unit)** | Simplified LSTM with only 2 gates (reset + update). Fewer parameters, faster training, comparable performance on many tasks. Good baseline temporal model. |
| 13 | **Bidirectional variants** | Process the sequence both forward and backward. For financial data, future context isn't available in real-time, but bidirectional training can improve pattern recognition during training. |
| 14 | **Stacked temporal layers** | Multiple LSTM/GRU layers. First layer captures short-term patterns, deeper layers capture increasingly abstract temporal features. |

**Expected impact**: The model finally understands that day 1 comes before day 2. Dramatic improvement in capturing trends, momentum, and mean-reversion patterns.

---

## Phase 4 — Attention & Transformers

The most impactful architecture family of the 2020s, adapted for time-series forecasting.

| # | Improvement | Why it matters |
|---|-------------|---------------|
| 15 | **Scaled dot-product attention** | Core building block. Allows each timestep to attend to every other timestep with learned relevance weights. Captures arbitrary long-range dependencies without recurrence. |
| 16 | **Positional encoding** | Sinusoidal or learned position embeddings. Since attention has no inherent ordering, positional encoding injects time-step information. |
| 17 | **Multi-head attention** | Multiple parallel attention heads, each learning different aspects of temporal relationships (one head for short-term, another for seasonal patterns, etc.). |
| 18 | **Transformer encoder** | Full encoder block: multi-head attention + feedforward + LayerNorm + residual connections. Stack multiple blocks for a complete Transformer for time series. |
| 19 | **Temporal Fusion Transformer (TFT) concepts** | Variable selection networks, static covariate encoders, interpretable attention. State-of-the-art for multi-horizon financial forecasting with built-in interpretability. |

**Expected impact**: State-of-the-art sequence modeling. Parallel training (unlike RNNs). Interpretable attention maps show which past days the model finds most relevant.

---

## Phase 5 — Data & Features

Better data processing and feature engineering for financial time series.

| # | Improvement | Why it matters |
|---|-------------|---------------|
| 20 | **Returns-based input** | Feed log-returns `ln(P_t / P_{t-1})` instead of raw prices. Returns are stationary and scale-invariant — the network doesn't need to learn that 100→101 is the same as 50→50.50. |
| 21 | **Technical indicators as features** | RSI, MACD, Bollinger Bands, moving averages. Hand-engineered features that encode momentum, volatility, and trend information that raw OHLCV alone takes many epochs to learn. |
| 22 | **Time-series augmentation** | Jittering (add small noise), magnitude warping, window slicing. Increases effective dataset size and improves generalization, especially valuable with only 3 training tickers. |
| 23 | **Multi-horizon prediction** | Predict next 1, 5, and 20 days simultaneously. Multi-task learning provides a richer training signal and more useful outputs for trading strategies. |
| 24 | **Cross-sectional features** | Relative strength vs. sector average, correlation features. Lets the model learn that "BP is outperforming the energy sector" rather than just "BP went up". |

**Expected impact**: The model works with more informative, stationary inputs. Augmentation multiplies effective training data.

---

## Phase 6 — Production & Performance

Optimize for real-world deployment and computational efficiency.

| # | Improvement | Why it matters |
|---|-------------|---------------|
| 25 | **SIMD vectorization** | Use Rust's `std::simd` or `packed_simd` for matrix operations. 4-8x speedup on CPU for dot products, activation functions, and element-wise operations. |
| 26 | **Rayon parallelism** | Parallelize mini-batch gradient computation across CPU cores. Each sample in a batch is independent — perfect for data parallelism with `rayon::par_iter`. |
| 27 | **GPU acceleration (wgpu)** | Move matrix multiplications to the GPU via wgpu compute shaders. 10-100x speedup for large matrices, making Transformer-scale models feasible. |
| 28 | **Quantization (f32 → f16/i8)** | Reduce model size and inference time by using half-precision or integer weights. Minimal accuracy loss for inference, 2-4x memory reduction. |
| 29 | **Ensemble methods** | Train multiple models (different seeds, architectures, windows) and average predictions. Reduces variance and consistently improves accuracy by 5-15%. |
| 30 | **Online learning / incremental training** | Update the model with new data daily without full retraining. Essential for adapting to regime changes in financial markets. |

**Expected impact**: Production-ready performance. Real-time inference. Continuous adaptation.

---

## Implementation Priority

```
Phase 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ NOW
Phase 2 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Next
Phase 3 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ After architecture is solid
Phase 4 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ When temporal base works
Phase 5 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Parallel with any phase
Phase 6 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ When model is finalized
```

Each phase builds on the previous one. Phase 1 changes can be applied immediately
to the existing codebase with minimal refactoring.
