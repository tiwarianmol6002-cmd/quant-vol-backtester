""
vol_targeting.py
-----------------
Step 4b: turn the LightGBM predicted-volatility series into a CONTINUOUS
position size, replacing the baseline's flat 0/1 sizing.

FINANCE LOGIC -- volatility targeting
----------"----------------------------
The idea: risk, not price exposure, is what you actually want to control.
Holding a fixed 100% position size means your REALIZED risk swings around
with the market's volatility -- you're taking much more risk in a turbulent
period than in a calm one, usually without intending to.

Volatility targeting instead scales the position so that your EXPECTED
risk contribution stays roughly constant over time:

    size_t = target_vol / predicted_vol_t      (capped at max_leverage)

If predicted next-day vol is LOW, size_t is large (you can take a bigger
position for the same risk budget). If predicted vol is HIGH, size_t
shrinks -- you deliberately back off exposure heading into turbulence,
which is exactly the scenario where our mean-reversion baseline's flat
sizing is most likely to get hurt (a "cheap" price in a high-vol regime is
more likely to keep falling, not revert).

This is a genuinely standard institutional technique (risk parity / vol
targeting funds are a real, large category), not something invented for
this project -- but note two things clearly for your README:
  1. It only helps if the volatility FORECAST is actually decent (garbage
     predictions -> garbage sizing -> this can hurt as easily as help).
  2. We're capping leverage (`max_leverage`) specifically because an
     unconstrained 1/predicted_vol size can blow up to enormous, unrealistic
     positions whenever the model predicts a near-zero vol -- a real failure
     mode you should watch for in your results, not just a defensive code
     comment.

ALIGNMENT WITH THE PREDICTION'S TIMING (re-confirming the lookahead logic
from ml_vol_model.py): a prediction made using features through day t is a
forecast of day t+1's volatility. It becomes available at t's close, so
it's valid to use for sizing the position HELD on day t+1 -- i.e. it needs
to be shifted forward by one day to align with the `position` column
(itself already lagged by one day relative to the raw signal). We do that
shift explicitly here rather than assuming the caller got it right.
"""

import numpy as np
import pandas as pd


def apply_vol_targeting(
    df: pd.DataFrame,
    predicted_vol: pd.Series,
    target_vol_annual: float = 0.15,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """
    df: backtest-ready frame containing `position` (0/1 baseline signal,
        already lagged -- see strategy_mean_reversion.py).
    predicted_vol: Series indexed like df, the model's prediction of
        NEXT-DAY's (unannualized, daily-scale) rolling volatility, as
        produced by ml_vol_model.evaluate_model()['predictions'].
    target_vol_annual: the annualized volatility level we're sizing toward
        (0.15 = "size as if this were a 15%-annual-vol asset").

    Adds:
      predicted_vol_aligned : predicted_vol shifted +1 day to line up with
                               the day it should influence sizing on
      raw_size_multiplier   : target_vol / predicted_vol (daily-scale, uncapped)
      size_multiplier       : same, capped to [0, max_leverage]
      position_sized         : position * size_multiplier -- the CONTINUOUS
                               position to feed into run_backtest()
    """
    df = df.copy()

    target_vol_daily = target_vol_annual / np.sqrt(252)

    # Shift the prediction forward by 1: a prediction computed from day t's
    # features (predicting t+1's vol) becomes usable for sizing on t+1.
    aligned = predicted_vol.reindex(df.index).shift(1)
    df["predicted_vol_aligned"] = aligned

    raw_mult = target_vol_daily / aligned.replace(0, np.nan)
    df["raw_size_multiplier"] = raw_mult
    df["size_multiplier"] = raw_mult.clip(lower=0, upper=max_leverage).fillna(0)

    df["position_sized"] = df["position"] * df["size_multiplier"]

    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_pipeline import build_dataset
    from strategy_mean_reversion import generate_mean_reversion_signals
    from ml_vol_model import build_features, chronological_split, train_vol_model, evaluate_model
    from backtest_engine import run_backtest, compute_metrics

    datasets = build_dataset()
    if datasets:
        tkr = list(datasets.keys())[0]
        base_df = datasets[tkr]

        sig_df = generate_mean_reversion_signals(base_df)
        feat_df = build_features(base_df)
        train_df, test_df = chronological_split(feat_df)
        model = train_vol_model(train_df)
        results = evaluate_model(model, test_df)

        merged = sig_df.loc[test_df.index]
        sized = apply_vol_targeting(merged, results["predictions"])

        bt_baseline = run_backtest(sized, cost_bps=5.0, position_col="position")
        bt_sized = run_backtest(sized, cost_bps=5.0, position_col="position_sized")

        m_base, _ = compute_metrics(bt_baseline, position_col="position")
        m_sized, _ = compute_metrics(bt_sized, position_col="position_sized")

        print(f"Baseline (0/1) Sharpe: {m_base['strategy']['sharpe']:.2f}")
        print(f"Vol-targeted Sharpe:   {m_sized['strategy']['sharpe']:.2f}")
