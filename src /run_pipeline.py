"""
run_pipeline.py
----------------
Step 5: orchestrates the full pipeline across every ticker and produces the
artifacts a resume/portfolio writeup needs:
  - outputs/summary_metrics.csv   (strategy vs buy-hold, all tickers)
  - outputs/equity_curve_{TICKER}.png
  - outputs/drawdown_{TICKER}.png
  - outputs/predicted_vs_actual_vol_{TICKER}.png
  - outputs/trades_{TICKER}.csv

Run with:  python src/run_pipeline.py
Requires real internet access to Yahoo Finance (this repo's dev sandbox did
not have that access when the code was written -- see README's "Validation"
section for how each module was tested against synthetic data instead).
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from data_pipeline import build_dataset, DEFAULT_TICKERS
from strategy_mean_reversion import generate_mean_reversion_signals
from ml_vol_model import build_features, chronological_split, train_vol_model, evaluate_model
from vol_targeting import apply_vol_targeting
from backtest_engine import run_backtest, compute_metrics, _max_drawdown

from pathlib import Path
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def run_for_ticker(ticker: str, raw_df: pd.DataFrame, cost_bps: float = 5.0, train_frac: float = 0.7):
    """Runs mean-reversion baseline + ML vol-targeted variant for one ticker."""
    sig_df = generate_mean_reversion_signals(raw_df)
    feat_df = build_features(raw_df)
    train_df, test_df = chronological_split(feat_df, train_frac=train_frac)

    if len(train_df) < 100 or len(test_df) < 50:
        print(f"[SKIP] {ticker}: not enough history after warmup/split.")
        return None

    model = train_vol_model(train_df)
    ml_results = evaluate_model(model, test_df)

    # Evaluate BOTH variants on the exact same (test-period) rows, so the
    # comparison isn't confounded by different date ranges.
    merged = sig_df.loc[test_df.index]
    sized = apply_vol_targeting(merged, ml_results["predictions"])

    bt_base = run_backtest(sized, cost_bps=cost_bps, position_col="position")
    bt_sized = run_backtest(sized, cost_bps=cost_bps, position_col="position_sized")

    m_base, trades_base = compute_metrics(bt_base, position_col="position")
    m_sized, trades_sized = compute_metrics(bt_sized, position_col="position_sized")

    # --- plots ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bt_base.index, bt_base["equity_strategy"], label="Mean-Reversion (flat sizing)")
    ax.plot(bt_sized.index, bt_sized["equity_strategy"], label="Mean-Reversion (vol-targeted sizing)")
    ax.plot(bt_base.index, bt_base["equity_buyhold"], label="Buy & Hold", linestyle="--", color="gray")
    ax.set_title(f"{ticker}: Equity Curves (test period)")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"equity_curve_{ticker}.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    eq = bt_sized["equity_strategy"]
    dd = eq / eq.cummax() - 1
    ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.4)
    ax.set_title(f"{ticker}: Drawdown (vol-targeted variant)")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"drawdown_{ticker}.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test_df.index, test_df["target_vol_next"], label="Actual next-day vol (20d rolling)", alpha=0.8)
    ax.plot(ml_results["predictions"].index, ml_results["predictions"], label="Predicted", alpha=0.8)
    ax.set_title(f"{ticker}: Predicted vs Actual Volatility (test period)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"predicted_vs_actual_vol_{ticker}.png", dpi=120)
    plt.close(fig)

    trades_sized.to_csv(OUT_DIR / f"trades_{ticker}.csv", index=False)

    return {
        "ticker": ticker,
        "ml_mae": ml_results["mae"],
        "ml_r2": ml_results["r2"],
        "baseline_sharpe": m_base["strategy"]["sharpe"],
        "baseline_cagr": m_base["strategy"]["cagr"],
        "baseline_maxdd": m_base["strategy"]["max_drawdown"],
        "baseline_trades": m_base["strategy"]["n_trades"],
        "baseline_winrate": m_base["strategy"]["win_rate"],
        "volsized_sharpe": m_sized["strategy"]["sharpe"],
        "volsized_cagr": m_sized["strategy"]["cagr"],
        "volsized_maxdd": m_sized["strategy"]["max_drawdown"],
        "buyhold_sharpe": m_base["buy_and_hold"]["sharpe"],
        "buyhold_cagr": m_base["buy_and_hold"]["cagr"],
        "buyhold_maxdd": m_base["buy_and_hold"]["max_drawdown"],
    }


def main():
    print(f"Pulling data for {len(DEFAULT_TICKERS)} tickers...")
    datasets = build_dataset()

    rows = []
    for tkr, df in datasets.items():
        print(f"Running pipeline for {tkr}...")
        row = run_for_ticker(tkr, df)
        if row:
            rows.append(row)

    if not rows:
        print("[ERROR] No tickers produced results.")
        return

    summary = pd.DataFrame(rows).set_index("ticker")
    summary.to_csv(OUT_DIR / "summary_metrics.csv")
    print(f"\n[OK] Saved summary_metrics.csv and per-ticker plots/trades -> {OUT_DIR}")
    print(summary.round(3))


if __name__ == "__main__":
    main()
