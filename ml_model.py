# -*- coding: utf-8 -*-
"""
ml_model.py
============
用 scikit-learn 的 GradientBoostingClassifier 做「輕量級」量化模型。

為什麼選這個而不是深度學習/LLM：
    - GitHub Actions 免費 runner 只有 CPU、沒有 GPU，執行時間也有限制
    - 多篇量化文獻（含 Indonesia/Poland/Korea 股市的實證研究）都指出，
      梯度提升樹（Gradient Boosting / XGBoost 系列）在「股價方向分類」
      這種任務上，是CPU上訓練最快、且實務中表現最穩定的模型類型之一，
      優於單純的邏輯回歸，複雜度又遠低於 LSTM 等深度學習模型
    - 每次 Actions 執行都重新訓練（stateless），不需要另外存模型權重、
      不用擔心版本或環境漂移問題，架構最單純

⚠️ 重要局限（請務必知道）：
    - 訓練樣本通常只有幾百到一千多筆（yfinance 能給的歷史長度有限），
      加上是「每天重新訓練」，模型容易 overfit、也可能今天訓出的機率
      跟明天差異很大，本身不構成穩定的預測能力保證。
    - 這裡只把模型機率當成「多一個輔助訊號」跟 SMC / 技術指標一起加權，
      不建議把 ML 機率當成唯一的進出場依據。
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

import indicators as ind


FEATURE_COLUMNS = [
    "rsi", "macd_hist", "adx", "di_diff",
    "ema20_gap", "ema50_gap", "ema_trend_align",
    "atr_pct", "volume_z", "dist_to_discount",
]


def build_feature_frame(df: pd.DataFrame, ind_df: pd.DataFrame) -> pd.DataFrame:
    """把原始 OHLCV + 已算好的指標，組成給模型用的特徵矩陣（每一列都只用當下及之前的資訊，不偷看未來）。"""
    feat = pd.DataFrame(index=df.index)
    close = df["Close"]

    feat["rsi"] = ind_df["rsi"]
    feat["macd_hist"] = ind_df["macd_hist"]
    feat["adx"] = ind_df["adx"]
    feat["di_diff"] = ind_df["plus_di"] - ind_df["minus_di"]

    feat["ema20_gap"] = (close - ind_df["ema20"]) / close
    feat["ema50_gap"] = (close - ind_df["ema50"]) / close
    feat["ema_trend_align"] = (
        np.where(ind_df["ema20"] > ind_df["ema50"], 1, -1) +
        np.where(ind_df["ema50"] > ind_df["ema200"], 1, -1)
    )

    feat["atr_pct"] = ind_df["atr"] / close

    vol = df["Volume"]
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std()
    feat["volume_z"] = (vol - vol_mean) / vol_std.replace(0, np.nan)

    roll_high = df["High"].rolling(50).max()
    roll_low = df["Low"].rolling(50).min()
    mid = (roll_high + roll_low) / 2
    span = (roll_high - roll_low).replace(0, np.nan)
    feat["dist_to_discount"] = (close - mid) / span  # >0 溢價, <0 折價

    return feat


def predict_next_move_probability(
    df: pd.DataFrame,
    ind_df: pd.DataFrame,
    horizon: int = 5,
    return_threshold: float = 0.0,
    min_train_rows: int = 80,
) -> Optional[dict]:
    """
    訓練一個輕量 GradientBoosting 分類器，預測「未來 horizon 根K棒後，
    收盤價漲幅是否超過 return_threshold」的機率。

    回傳 {"prob_up": float, "trained_rows": int, "horizon": int} 或
    None（資料量不足 / 類別過度失衡時，代表模型不可靠，不提供分數）。
    """
    feat = build_feature_frame(df, ind_df)
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    label = (future_return > return_threshold).astype(int)

    data = feat.copy()
    data["label"] = label
    data = data.dropna()

    if len(data) < min_train_rows:
        return None

    y = data["label"]
    if y.nunique() < 2 or min(y.mean(), 1 - y.mean()) < 0.08:
        # 標籤幾乎都是同一類（例如過去一路噴出或一路崩），模型學不到東西
        return None

    X_train = data[FEATURE_COLUMNS]

    model = GradientBoostingClassifier(
        n_estimators=120, max_depth=2, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    model.fit(X_train, y)

    latest_feat = feat.iloc[[-1]][FEATURE_COLUMNS]
    if latest_feat.isna().any(axis=None):
        return None

    prob_up = float(model.predict_proba(latest_feat)[0, 1])
    return {"prob_up": prob_up, "trained_rows": len(data), "horizon": horizon}
