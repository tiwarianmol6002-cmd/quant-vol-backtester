# Volatility-Aware Mean-Reversion Backtester (with ML-Based Position Sizing)

A from-scratch backtesting engine for a mean-reversion equity strategy, extended
with a LightGBM volatility forecasting model used for risk-based position sizing.
Built to be readable end-to-end: every module is commented to explain not just
*what* the code does but *why* — the trading logic, the statistics, and the
backtesting pitfalls (lookahead bias, overfitting) that make naive backtests
misleading.

## What this project shows

- A complete, modular quant research pipeline: data → strategy signal → backtest
  engine → ML forecasting → risk-based sizing → evaluation
- Deliberate, documented handling of the two most common backtesting mistakes:
  **lookahead bias** (using information you wouldn't actually have had yet) and
  **overfitting** (tuning a strategy until it looks good on the one dataset you
  tested it on)
- A genuine before/after comparison: does an ML volatility forecast, fed into
  position sizing, actually improve risk-adjusted returns — tested honestly,
  not cherry-picked

## How it works

**1. Data pipeline** (`src/data_pipeline.py`)
Pulls ~5 years of daily OHLCV data for SPY plus 7 liquid large-caps via
`yfinance`, using split/dividend-adjusted prices (so a stock split doesn't look
like a 50% crash in the data). Cleans the data and computes daily log returns
and rolling (20-day) volatility.

**2. Strategy signal — mean reversion** (`src/strategy_mean_reversion.py`)
The bet: when a price is unusually far below its recent average (a z-score of
its own price, `(price - rolling_mean) / rolling_std`), it tends to drift back
toward that average rather than keep falling — this is a well-documented effect
in broad indices over short horizons, and much less reliable for single stocks
in a genuine downtrend. Entry at a configurable z-score threshold, exit on
reversion, a stop-loss, or a max holding period as a safety valve. All position
decisions are explicitly lagged one day (`signal.shift(1)`) so nothing trades on
information it wouldn't have had yet.

**3. Backtest engine** (`src/backtest_engine.py`)
A vectorized PnL simulator: applies the strategy's daily returns net of
transaction costs (a flat basis-point charge per trade), and computes cumulative
equity, Sharpe ratio, max drawdown, win rate, and trade count — benchmarked
against simply buying and holding the same asset.

**4. ML volatility forecasting + risk-based sizing** (`src/ml_vol_model.py`,
`src/vol_targeting.py`)
Volatility is far more predictable than price direction (it clusters — calm
periods tend to stay calm, turbulent periods tend to stay turbulent). A LightGBM
regressor is trained on lagged returns, multi-horizon rolling volatility, volume
anomalies, and RSI to predict next-day volatility, using a strict chronological
train/test split (never a random shuffle — see "Avoiding lookahead bias" below).
The prediction is then used for **volatility targeting**: position size scales
up when predicted volatility is low and scales down when predicted volatility is
high, aiming for a roughly constant risk contribution over time instead of a
fixed position size regardless of market conditions.

**5. Evaluation** (`src/run_pipeline.py`)
Runs the full pipeline across every ticker, and saves a summary metrics table
plus equity curve, drawdown, and predicted-vs-actual-volatility plots for each.

## Avoiding lookahead bias (the #1 backtesting pitfall)

Every module explicitly documents and enforces this, but the short version:

- Rolling features (mean, std, RSI, etc.) at row `t` only use data through `t`
  — trailing windows only, never centered windows.
- The trading **signal** computed from day `t`'s close is not tradeable until
  day `t+1` — `position = signal.shift(1)` makes this literal in the code
  rather than an assumption.
- The ML model's **target** (next-day volatility) is intentionally shifted
  *forward* relative to its features — that's correct for a training label, but
  the **train/test split is strictly chronological**, never a random shuffle,
  so the model is never trained on data from after the period it's evaluated on.
- The baseline (flat sizing) and ML-sized variant are evaluated on the exact
  same backtest engine and the exact same test-period dates, so any performance
  difference reflects the sizing method, not a difference in mechanics or dates.

## Avoiding overfitting

The mean-reversion strategy's parameters (entry/exit z-score, stop-loss, max
holding period) were chosen by intuition up front, not tuned against backtest
results — tuning them to maximize Sharpe on the same data you're evaluating on
is a classic way to produce a strategy that looks great in a backtest and fails
live. The LightGBM model is deliberately kept shallow (max depth 4, few leaves)
because volatility features are highly autocorrelated, making it easy for a
flexible model to memorize training-period noise rather than learn a
generalizable pattern.

## Project structure

```
vol_backtest/
├── src/
│   ├── data_pipeline.py          # Step 1: fetch, clean, feature-engineer
│   ├── strategy_mean_reversion.py # Step 2: entry/exit signal logic
│   ├── backtest_engine.py         # Step 3: PnL simulation + metrics
│   ├── ml_vol_model.py            # Step 4a: LightGBM volatility forecasting
│   ├── vol_targeting.py           # Step 4b: vol-targeted position sizing
│   └── run_pipeline.py            # Step 5: orchestrates everything, saves results
├── data/                          # downloaded OHLCV CSVs (gitignored)
├── outputs/                       # generated plots & summary metrics (gitignored)
├── requirements.txt
└── README.md
```

## Running it

```bash
git clone <this-repo>
cd vol_backtest
pip install -r requirements.txt
python src/run_pipeline.py
```

This downloads data for the default 8-ticker universe, runs the full pipeline
for each, and writes `outputs/summary_metrics.csv` plus per-ticker plots.
Runtime is a few minutes, dominated by the yfinance downloads.

## Results

*Populate this section after running `run_pipeline.py` locally* — paste in
`outputs/summary_metrics.csv` as a table, and embed 2-3 of the generated plots
(equity curve, drawdown, predicted-vs-actual volatility) from `outputs/`.

Suggested table format:

| Ticker | Baseline Sharpe | Vol-Targeted Sharpe | Buy & Hold Sharpe | Baseline Max DD | Vol-Targeted Max DD | ML R² |
|--------|----------------:|---------------------:|-------------------:|-----------------:|----------------------:|------:|
| SPY    |                 |                       |                     |                   |                        |       |
| ...    |                 |                       |                     |                   |                        |       |

During development (this sandbox had no live internet access to Yahoo Finance),
every module was validated end-to-end against synthetic mean-reverting and
regime-switching price series to confirm the signal logic, anti-lookahead
shifting, and ML pipeline all behave correctly — see inline module docstrings
for what was checked. Those synthetic-data numbers are **not** representative
of real market performance and are intentionally excluded from this README;
run the pipeline on real data before reporting results.

## What this project is *not*, and what not to claim from it

Being upfront about this matters more than the results themselves:

- **Not production-ready.** No live order execution, no real-time data feed, no
  handling of corporate actions beyond what yfinance's adjusted prices already
  cover.
- **Transaction costs are simplified.** A flat basis-point charge per trade does
  not capture slippage, market impact, or the fact that spreads widen exactly
  when volatility (and this strategy's own signal) spikes — real costs on a
  volatility-sensitive strategy are likely higher than modeled here.
- **Single-asset backtests, not a portfolio.** Running the strategy on 8
  tickers independently doesn't show what happens when they're combined into
  one portfolio (correlated drawdowns, capital allocation across names).
- **No claim that this beats real trading costs or would be profitable live.**
  A backtest — even a carefully de-biased one — is evidence about a strategy's
  historical logic, not a guarantee about future or live performance.
- **The volatility forecast's value is conditional.** Vol-targeted sizing only
  helps if the forecast is actually decent; in some regimes/tickers it may
  reduce Sharpe by increasing volatility more than it increases return (I saw
  this in development testing) — report results as observed, not as a
  uniform improvement.

## License

MIT — see `LICENSE`.
