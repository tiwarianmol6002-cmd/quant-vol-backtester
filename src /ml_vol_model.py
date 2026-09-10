"""
ml_vol_model.py
----------------
Step 4: predict NEXT-DAY realized volatility using engineered features, then
use that prediction to size positions (volatility targeting).

FINANCE LOGIC
-------------
Why predict volatility instead of price/return direction? Volatility is
substantially more predictable than direction -- it clusters (high-vol days
tend to follow high-vol days; "volatility clustering" is one of the most
robust stylized facts in finance) whereas next-day price direction is close
to a coin flip for liquid names. This is why real quant shops spend a lot
of effort on volatility forecasting for RISK MANAGEMENT and position sizing,
rather than trying to predict direction directly.

TARGET: next-day's 20-day rolling volatility, i.e. rolling_vol_20 SHIFTED
BACKWARD by one day relative to the features (target_t = rolling_vol_20 at
t+1). We predict a forward-looking, already-smoothed quantity rather than a
single day's noisy squared return, which is a more learnable target and
closer to what you'd actually use for sizing decisions.

FEATURES (all computed using only information through day t):
  - lagged log returns (1, 5, 10 day)
  - rolling volatility at multiple horizons (10, 20, 60 day)
  - rolling volume z-score (volume spikes often precede/accompany vol spikes)
  - RSI(14) -- momentum/overbought-oversold indicator, included because
    extreme RSI readings often coincide with elevated realized vol

LOOKAHEAD BIAS -- THE MOST IMPORTANT PART OF THIS FILE:
  1. All FEATURES at row t must only use data through t (true of every
     rolling/lag computation below, since pandas .rolling()/.shift() are
     trailing).
  2. The TARGET at row t is tomorrow's volatility (t+1) -- meaning row t's
     (features, target) pair is only usable for training once row t+1 has
     actually happened. This is fine for training (we're not "trading" on
     it), but the train/test SPLIT must be chronological, never a random
     shuffle. Shuffling would let the model train on data from after the
     test period and silently "peek" at the future through the split
     itself -- a subtle and very common mistake in first ML backtests.
  3. We fit any scalers/imputers on the TRAIN split only, then apply
     (never re-fit) to the test split, for the same reason.
"""

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Standard Wilder RSI. >70 = 'overbought', <30 = 'oversold' by convention."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must already have: log_return, Close, Volume (from Step 1).
    Adds feature columns + the forward target `target_vol_next`.
    """
    df = df.copy()

    for lag in (1, 5, 10):
        df[f"ret_lag_{lag}"] = df["log_return"].rolling(lag).sum()

    for w in (10, 20, 60):
        df[f"vol_{w}"] = df["log_return"].rolling(w).std()

    vol_mean = df["Volume"].rolling(20).mean()
    vol_std = df["Volume"].rolling(20).std()
    df["volume_z"] = (df["Volume"] - vol_mean) / vol_std

    df["rsi_14"] = compute_rsi(df["Close"], 14)

    # TARGET: tomorrow's 20-day rolling vol. shift(-1) looks FORWARD -- this
    # is correct here because it's the label, not a feature used to decide
    # today's trade. Never shift(-1) anything that feeds into `position`.
    df["target_vol_next"] = df["vol_20"].shift(-1)

    return df


FEATURE_COLS = [
    "ret_lag_1", "ret_lag_5", "ret_lag_10",
    "vol_10", "vol_20", "vol_60",
    "volume_z", "rsi_14",
]


def chronological_split(df: pd.DataFrame, train_frac: float = 0.7):
    """
    Time-ordered split -- NEVER use train_test_split(shuffle=True) here.
    Returns (train_df, test_df). Also drops rows with any NaN in features
    or target (from warmup periods / the final row's shifted target).
    """
    clean = df.dropna(subset=FEATURE_COLS + ["target_vol_next"])
    split_i = int(len(clean) * train_frac)
    train_df = clean.iloc[:split_i]
    test_df = clean.iloc[split_i:]
    return train_df, test_df


def train_vol_model(train_df: pd.DataFrame, params: dict = None):
    """
    Trains a LightGBM regressor to predict target_vol_next from FEATURE_COLS.
    Kept deliberately simple (shallow trees, few leaves) -- with a modest
    number of daily rows per ticker, an aggressive model will overfit the
    training window's specific noise rather than learning a generalizable
    vol-clustering pattern. Flagging this as a real risk, not a formality:
    volatility features are highly autocorrelated, so it's easy for a
    flexible model to look great in-sample and fail out-of-sample.
    """
    import lightgbm as lgb

    default_params = dict(
        n_estimators=200,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    if params:
        default_params.update(params)

    model = lgb.LGBMRegressor(**default_params)
    model.fit(train_df[FEATURE_COLS], train_df["target_vol_next"])
    return model


def evaluate_model(model, test_df: pd.DataFrame) -> dict:
    from sklearn.metrics import mean_absolute_error, r2_score

    preds = model.predict(test_df[FEATURE_COLS])
    actual = test_df["target_vol_next"].values

    return {
        "mae": mean_absolute_error(actual, preds),
        "r2": r2_score(actual, preds),
        "predictions": pd.Series(preds, index=test_df.index, name="predicted_vol"),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_pipeline import build_dataset

    datasets = build_dataset()
    if datasets:
        tkr = list(datasets.keys())[0]
        feat_df = build_features(datasets[tkr])
        train_df, test_df = chronological_split(feat_df)
        print(f"Train: {len(train_df)} rows ({train_df.index[0].date()} -> {train_df.index[-1].date()})")
        print(f"Test:  {len(test_df)} rows ({test_df.index[0].date()} -> {test_df.index[-1].date()})")
        model = train_vol_model(train_df)
        results = evaluate_model(model, test_df)
        print(f"Test MAE: {results['mae']:.5f}, R2: {results['r2']:.3f}")
