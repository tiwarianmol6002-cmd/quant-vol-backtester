"""
strategy_mean_reversion.py
---------------------------
Step 2: A z-score based mean-reversion strategy.

FINANCE LOGIC
-------------
- rolling_mean, rolling_std : computed on PRICE (Close), over `lookback` days.
  This is different from Step 1's rolling_vol, which was computed on RETURNS.
  Both are "volatility-ish" but price-std answers "how far does price
  typically wander from its recent average level", while return-std answers
  "how big are daily percentage moves". We use price-std here because the
  entry rule ("price is N std devs below its mean") is naturally a
  price-level concept (this is the classic Bollinger Band construction).

- z-score: z_t = (Close_t - rolling_mean_t) / rolling_std_t
  A z of -2 means "today's close is 2 standard deviations below its recent
  average" — i.e., unusually cheap relative to its own recent history.

- Entry: z_t <= -entry_z  ->  signal to go long
- Exit:  z_t >= exit_z (default 0, i.e. price reverted to its mean)
         OR held for `max_hold_days` (time-stop: cuts losses if the
         "cheap" price never reverts)
         OR a hard stop-loss beyond `stop_loss_z` standard deviations further
         down (protects against a real, non-reverting decline)

THIS STEP ONLY GENERATES SIGNALS — it does NOT simulate PnL. That's Step 3's
job (the backtest engine), kept deliberately separate so you can unit-test
"is my signal logic correct" independently of "is my PnL simulation correct."

LOOKAHEAD BIAS — READ THIS:
  `signal` at row t is computed from data available through t (Close_t,
  rolling stats through t). But you cannot ACT on that information until
  the NEXT bar, because in live trading you only know today's close after
  the market closes — there's no way to trade at today's close using
  today's close as your decision input. We therefore create a separate
  `position` column that is `signal.shift(1)`: the position you actually
  HOLD on day t is decided using information through day t-1. This is the
  single most common bug in first backtests — getting this shift backwards
  silently makes strategies look far better than they'd ever perform live.
"""

import numpy as np
import pandas as pd


def generate_mean_reversion_signals(
    df: pd.DataFrame,
    lookback: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    stop_loss_z: float = 3.5,
    max_hold_days: int = 15,
) -> pd.DataFrame:
    """
    Given a cleaned OHLCV+features DataFrame (from Step 1), add:
      rolling_mean, rolling_std, zscore  : the indicator
      signal   : instantaneous state machine output at time t (uses info
                 through t) -- 1 = "should be long", 0 = "should be flat"
      position : signal.shift(1) -- what you ACTUALLY hold at time t,
                 based only on information through t-1. THIS is the column
                 the backtest engine (Step 3) must use, never `signal`.

    We implement entry/exit as an explicit stateful loop rather than a
    vectorized one-liner. A vectorized "z <= -entry_z" mask alone can't
    express "stay in the trade until exit conditions are met" (that
    requires memory of whether you're currently in a position and how long
    you've held it) -- so a loop is the correct and honest tool here, not
    a shortcut. It's O(n) and plenty fast for daily data.
    """
    df = df.copy()
    df["rolling_mean"] = df["Close"].rolling(lookback).mean()
    df["rolling_std"] = df["Close"].rolling(lookback).std()
    df["zscore"] = (df["Close"] - df["rolling_mean"]) / df["rolling_std"]

    signal = np.zeros(len(df), dtype=int)
    in_position = False
    days_held = 0
    entry_idx = None

    z = df["zscore"].values

    for i in range(len(df)):
        if np.isnan(z[i]):
            signal[i] = 0
            continue

        if not in_position:
            if z[i] <= -entry_z:
                in_position = True
                days_held = 0
                entry_idx = i
            signal[i] = 1 if in_position else 0
        else:
            days_held += 1
            hit_exit = z[i] >= exit_z
            hit_stop = z[i] <= -stop_loss_z
            hit_time = days_held >= max_hold_days

            if hit_exit or hit_stop or hit_time:
                in_position = False
                signal[i] = 0  # flat starting this bar's signal (exit decided using today's info)
            else:
                signal[i] = 1

    df["signal"] = signal
    # THE critical anti-lookahead line: what we hold today is decided by
    # yesterday's information.
    df["position"] = df["signal"].shift(1).fillna(0).astype(int)

    return df


if __name__ == "__main__":
    # Quick smoke test on synthetic data so this runs standalone.
    import sys
    sys.path.insert(0, ".")
    from data_pipeline import build_dataset

    datasets = build_dataset()
    if datasets:
        tkr = list(datasets.keys())[0]
        out = generate_mean_reversion_signals(datasets[tkr])
        n_trades_entries = (out["signal"].diff() == 1).sum()
        print(f"{tkr}: {n_trades_entries} entry signals generated")
        print(out[["Close", "zscore", "signal", "position"]].dropna().head(30))
