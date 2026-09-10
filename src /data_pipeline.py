"""
data_pipeline.py
-----------------
Step 1 of the volatility backtesting project.

Responsibilities:
  1. Download daily OHLCV data for a basket of liquid tickers via yfinance
  2. Clean it (handle missing data, ensure split/dividend adjustment is correct)
  3. Compute log returns and rolling volatility features

FINANCE CONCEPTS (read before the code — they explain WHY we do things this way)
----------------------------------------------------------------------------
- OHLCV = Open, High, Low, Close, Volume — the standard daily bar for a stock.

- "Adjusted Close" vs "Close": when a company pays a dividend or does a stock
  split, the raw price series has artificial jumps that have nothing to do
  with the market's view of the company's value (e.g. a 2:1 split halves the
  price overnight even though nothing economically changed). The "Adjusted
  Close" backward-adjusts historical prices so the series is continuous and
  comparable over time. We ALWAYS compute returns from adjusted prices,
  never raw Close — using raw Close would inject fake -50% "crashes" into
  your data on split dates and misstate returns around dividend dates.

- Log returns vs simple returns: we use log returns r_t = ln(P_t / P_{t-1})
  instead of simple returns (P_t / P_{t-1} - 1) because log returns are
  time-additive (an N-day return is just the sum of N daily log returns,
  which makes rolling-window math clean) and are approximately symmetric
  for small moves. For volatility estimation specifically, log returns are
  the market-standard choice.

- Rolling volatility: "volatility" in quant finance almost always means the
  standard deviation of returns over some lookback window, annualized. A
  20-day rolling std of daily log returns tells you "how much has this
  asset been bouncing around over roughly the last calendar month." We
  annualize by multiplying by sqrt(252) because there are ~252 trading
  days in a year and variance scales linearly with time (so std scales
  with sqrt(time)) under a random-walk assumption. Annualized vol is what
  lets you compare a stock's "riskiness" to standard benchmarks (e.g. SPY
  historically runs ~15-20% annualized vol; a spike to 40%+ signals stress).

LOOKAHEAD BIAS WARNING (read this — it will matter again in Step 3):
  Every feature computed here (returns, rolling vol, etc.) at row t uses
  ONLY data up to and including day t. That's correct for feature
  engineering. The bias trap comes later, in the backtest, if you ever let
  a decision at time t use information that wasn't actually available
  until after time t (e.g. today's close before today's close is known,
  or a rolling window that accidentally centers instead of trails). We'll
  flag this explicitly again in Step 3.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# A mix of a broad index ETF + large-cap single names across sectors.
# Diversity of names matters later for the ML step (more data, varied regimes).
DEFAULT_TICKERS = [
    "SPY",   # S&P 500 ETF — broad market benchmark
    "AAPL",  # Tech
    "MSFT",  # Tech
    "JPM",   # Financials
    "XOM",   # Energy
    "JNJ",   # Healthcare / defensive
    "AMZN",  # Consumer discretionary
    "NVDA",  # High-vol tech (useful for volatility strategy testing)
]


def download_ohlcv(tickers=None, period_years=5, interval="1d"):
    """
    Download daily OHLCV for each ticker via yfinance.

    Returns a dict[ticker] -> DataFrame with columns:
    Open, High, Low, Close, Volume  (Close is ALREADY split/dividend adjusted
    when we pass auto_adjust=True, which is yfinance's default behavior now —
    we set it explicitly so this doesn't silently change on a library upgrade).
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    period_str = f"{period_years}y"
    raw = {}

    for tkr in tickers:
        df = yf.download(
            tkr,
            period=period_str,
            interval=interval,
            auto_adjust=True,   # adjusts OHLC for splits & dividends -> "Close" is adjusted close
            progress=False,
        )
        if df.empty:
            print(f"[WARN] No data returned for {tkr} — skipping.")
            continue

        # yfinance sometimes returns a MultiIndex column (ticker, field) even
        # for a single ticker depending on version — flatten defensively.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"
        raw[tkr] = df

    return raw


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a single ticker's OHLCV frame.

    Steps:
      - Drop rows where Close is NaN (can't do anything without a price)
      - Forward-fill isolated gaps in Volume only (a missing volume print
        doesn't mean the price is unknown) — we do NOT forward-fill price,
        since a faked/copied price would corrupt return calculations.
      - Drop duplicate index entries (rare yfinance artifact)
      - Sanity check: High >= Low, High >= Close >= Low etc. Drop violators.
    """
    df = df[~df.index.duplicated(keep="first")].sort_index()

    df = df.dropna(subset=["Close"])

    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].ffill()

    # Basic OHLC sanity checks — corrupt/erroneous bars do occasionally appear
    valid = (
        (df["High"] >= df["Low"])
        & (df["High"] >= df["Close"])
        & (df["Low"] <= df["Close"])
        & (df["Open"] > 0)
    )
    n_bad = (~valid).sum()
    if n_bad > 0:
        print(f"[INFO] Dropping {n_bad} rows failing OHLC sanity checks.")
    df = df[valid]

    return df


def add_return_and_vol_features(df: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """
    Add log returns and rolling volatility to a cleaned OHLCV frame.

    New columns:
      log_return        : daily log return, ln(Close_t / Close_{t-1})
      rolling_vol_{N}    : rolling std of log_return over `vol_window` days,
                            NOT yet annualized (raw daily-scale vol)
      rolling_vol_{N}_ann: the annualized version (x sqrt(252)), which is the
                            number people usually mean when they say
                            "20-day volatility is 18%"

    IMPORTANT (lookahead bias): rolling().std() in pandas is a TRAILING
    window by default (window ending at the current row), which is what we
    want — row t only uses data from t-window+1 .. t. Never call
    .rolling(..., center=True) for features used in a live/backtest signal,
    since that would use future data.
    """
    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    col_raw = f"rolling_vol_{vol_window}"
    col_ann = f"rolling_vol_{vol_window}_ann"
    df[col_raw] = df["log_return"].rolling(window=vol_window).std()
    df[col_ann] = df[col_raw] * np.sqrt(252)

    # First `vol_window` rows will have NaN vol (not enough history yet) —
    # this is expected and correct, not a bug. We'll drop/handle these
    # explicitly downstream rather than silently filling them.
    return df


def build_dataset(tickers=None, period_years=5, vol_window=20, save=True):
    """
    Full pipeline: download -> clean -> feature-engineer, for each ticker.
    Returns dict[ticker] -> feature DataFrame. Optionally saves each to CSV.
    """
    raw = download_ohlcv(tickers=tickers, period_years=period_years)
    out = {}
    for tkr, df in raw.items():
        df = clean_ohlcv(df)
        df = add_return_and_vol_features(df, vol_window=vol_window)
        out[tkr] = df
        if save:
            path = DATA_DIR / f"{tkr}.csv"
            df.to_csv(path)
            print(f"[OK] {tkr}: {len(df)} rows -> {path}")
    return out


if __name__ == "__main__":
    datasets = build_dataset()
    if not datasets:
        print("\n[ERROR] No tickers downloaded successfully. Check your network "
              "connection / yfinance access and try again.")
    else:
        sample = list(datasets.keys())[0]
        print(f"\nSample ({sample}) tail:")
        print(datasets[sample].tail())
