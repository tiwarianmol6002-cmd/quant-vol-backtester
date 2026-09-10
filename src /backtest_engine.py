"""
backtest_engine.py
-------------------
Step 3: turn a `position` column (0/1, already lagged to avoid lookahead —
see strategy_mean_reversion.py) into a realistic PnL simulation, and compute
standard performance metrics.

FINANCE LOGIC
-------------
- Strategy daily log return = position_t * log_return_t
  Because position_t was built as signal.shift(1) upstream, this line does
  NOT look ahead: the return you "earn" on day t is the market's actual
  day-t return, but you only ever apply it when `position` says you were
  already holding going into day t based on t-1 information.

- Transaction costs: applied on the days the position CHANGES (i.e., you
  paid the bid-ask spread / commission to enter or exit), sized as
  `cost_bps * abs(position_t - position_{t-1})`. Because position is 0/1
  here (no partial sizing yet in this baseline), a change is always a full
  entry or exit, so cost = cost_bps on those days, 0 otherwise. This is a
  simplification -- real cost also scales with your trade size (added in
  Step 4 when we vary position size) and with how illiquid/volatile the
  name is at that moment (a spike-vol day often means a wider spread than
  your flat bps estimate assumes).

- Equity curve = exp(cumsum(net log returns)), net of costs, starting at 1.0
  (i.e., "$1 grows to X"). Buy-and-hold equity curve uses the raw asset
  log returns the same way, for a like-for-like comparison.

- Sharpe ratio here uses a 0% risk-free rate assumption (common
  simplification for a first-pass backtest) -- explicitly NOT the same as
  an excess-return Sharpe over T-bills. Note this in your README.

- Max drawdown = min over time of (equity / running_max(equity) - 1),
  i.e. the worst percentage decline from any prior peak.

- Trades are identified as contiguous runs of position == 1. A trade's
  return is the compounded (net-of-cost) return over its holding period;
  "win" = trade return > 0.

LOOKAHEAD-BIAS NOTE: this module trusts that `position` was constructed
correctly upstream. It does not re-derive or re-check that. If you ever
write a NEW strategy module, re-verify (like we did in Step 2's smoke
test) that position_t only reflects information available through t-1
before plugging it into this engine -- this engine cannot detect that bug
for you.
"""

import numpy as np
import pandas as pd


def run_backtest(df: pd.DataFrame, cost_bps: float = 5.0, position_col: str = "position") -> pd.DataFrame:
    """
    df must contain: log_return, and `position_col` (already lagged --
    either the 0/1 baseline position, or a continuous vol-targeted size
    from vol_targeting.py). Passing position_col lets Step 4 reuse this
    exact same engine for the ML-sized variant -- important, since running
    the baseline and the ML variant through two DIFFERENT engines would
    make any performance difference partly an artifact of inconsistent
    PnL mechanics rather than a genuine comparison.

    cost_bps: round-trip-agnostic per-side cost in basis points (e.g. 5 = 0.05%),
              charged proportional to the CHANGE in position size each day --
              this generalizes correctly to continuous position sizes, not
              just 0/1 (a bigger change in size = a bigger trade = more cost).

    Returns df with added columns:
      trade_flag        : abs(position change) that day (0 if unchanged)
      cost               : the log-return cost paid that day
      strategy_ret_gross : position * log_return (before costs)
      strategy_ret_net   : strategy_ret_gross - cost  (what we actually earned)
      equity_strategy    : cumulative equity curve, net of costs, start = 1.0
      equity_buyhold     : cumulative equity curve of just holding the asset
    """
    df = df.copy()
    pos = df[position_col].fillna(0)

    pos_change = pos.diff().abs().fillna(pos.abs())
    df["trade_flag"] = pos_change

    cost_frac = cost_bps / 10000.0
    df["cost"] = df["trade_flag"] * cost_frac

    df["strategy_ret_gross"] = pos * df["log_return"]
    df["strategy_ret_net"] = df["strategy_ret_gross"] - df["cost"]

    df["equity_strategy"] = np.exp(df["strategy_ret_net"].fillna(0).cumsum())
    df["equity_buyhold"] = np.exp(df["log_return"].fillna(0).cumsum())

    return df


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return drawdown.min()


def _extract_trades(df: pd.DataFrame, position_col: str = "position") -> pd.DataFrame:
    """
    Identify contiguous nonzero-position runs and compute each trade's net
    compounded return. Works for both 0/1 baseline positions and
    continuous vol-targeted sizes (a "trade" = any contiguous stretch of
    nonzero exposure). Returns a DataFrame: entry_date, exit_date,
    n_days, trade_return.
    """
    pos = df[position_col].values
    dates = df.index
    net_ret = df["strategy_ret_net"].values

    trades = []
    in_trade = False
    start_i = None

    for i in range(len(pos)):
        if pos[i] != 0 and not in_trade:
            in_trade = True
            start_i = i
        end_of_trade = in_trade and (i == len(pos) - 1 or pos[i + 1] == 0)
        # NOTE: for continuous sizing, this counts a full sizing episode
        # (any nonzero stretch) as one "trade", even if size fluctuated
        # within it -- appropriate for win-rate/trade-count purposes.
        if end_of_trade:
            seg_ret = net_ret[start_i:i + 1].sum()  # sum of log returns = log of compounded return
            trades.append({
                "entry_date": dates[start_i],
                "exit_date": dates[i],
                "n_days": i - start_i + 1,
                "trade_return": np.exp(seg_ret) - 1,
            })
            in_trade = False

    return pd.DataFrame(trades)


def compute_metrics(df: pd.DataFrame, periods_per_year: int = 252, position_col: str = "position") -> dict:
    """
    Computes headline metrics for the STRATEGY (net of costs) and, for
    comparison, the buy-and-hold benchmark over the same period.
    """
    strat_ret = df["strategy_ret_net"].dropna()
    bh_ret = df["log_return"].dropna()

    def sharpe(returns):
        if returns.std() == 0 or len(returns) == 0:
            return np.nan
        return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)

    trades = _extract_trades(df, position_col=position_col)
    n_trades = len(trades)
    win_rate = (trades["trade_return"] > 0).mean() if n_trades > 0 else np.nan

    total_years = len(df) / periods_per_year

    metrics = {
        "strategy": {
            "total_return": df["equity_strategy"].iloc[-1] - 1,
            "cagr": df["equity_strategy"].iloc[-1] ** (1 / total_years) - 1,
            "annualized_vol": strat_ret.std() * np.sqrt(periods_per_year),
            "sharpe": sharpe(strat_ret),
            "max_drawdown": _max_drawdown(df["equity_strategy"]),
            "n_trades": n_trades,
            "win_rate": win_rate,
            "avg_trade_return": trades["trade_return"].mean() if n_trades > 0 else np.nan,
            "avg_hold_days": trades["n_days"].mean() if n_trades > 0 else np.nan,
        },
        "buy_and_hold": {
            "total_return": df["equity_buyhold"].iloc[-1] - 1,
            "cagr": df["equity_buyhold"].iloc[-1] ** (1 / total_years) - 1,
            "annualized_vol": bh_ret.std() * np.sqrt(periods_per_year),
            "sharpe": sharpe(bh_ret),
            "max_drawdown": _max_drawdown(df["equity_buyhold"]),
        },
    }
    return metrics, trades


def plot_equity_curve(df: pd.DataFrame, title: str = "Strategy vs Buy & Hold", save_path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["equity_strategy"], label="Mean-Reversion Strategy (net of costs)")
    ax.plot(df.index, df["equity_buyhold"], label="Buy & Hold", linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"[OK] Saved plot -> {save_path}")
    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_pipeline import build_dataset
    from strategy_mean_reversion import generate_mean_reversion_signals

    datasets = build_dataset()
    if datasets:
        tkr = list(datasets.keys())[0]
        sig_df = generate_mean_reversion_signals(datasets[tkr])
        bt_df = run_backtest(sig_df, cost_bps=5.0)
        metrics, trades = compute_metrics(bt_df)
        print(f"\n=== {tkr} backtest ===")
        for k, v in metrics["strategy"].items():
            print(f"  {k}: {v}")
        print(f"\nBuy & hold Sharpe: {metrics['buy_and_hold']['sharpe']:.2f}, "
              f"CAGR: {metrics['buy_and_hold']['cagr']:.2%}")
