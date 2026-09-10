# -*- coding: utf-8 -*-
"""
indicators.py
==============
主流技術分析指標的純 pandas/numpy 實作，不依賴 ta-lib 之類的 C 擴充套件，
確保在 GitHub Actions 的 ubuntu runner 上能穩定 `pip install` 並執行。

包含的指標都是業界（含學術文獻）最常被拿來做「多指標共振confluence」
判斷的組合：
    - RSI  (相對強弱指標)：動能，判斷超買/超賣
    - MACD (指數平滑異同移動平均)：趨勢動能與交叉訊號
    - EMA 20 / 50 / 200：趨勢方向與多空排列
    - ADX + DI：趨勢「強度」（不判斷方向，只判斷值不值得跟）
    - ATR：波動度，用來做動態的止損/區間寬度參考
    - OBV：用成交量驗證價格走勢是否有量能支撐

注意：學術文獻（如 Chung 2013、政大商學院 2023 年研究）指出，單獨使用
RSI 或 MACD 長期不必然顯著贏過 buy-and-hold；這裡的用途是拿來跟 SMC
結構訊號做「共振過濾」，降低單一指標雜訊，而不是宣稱這些指標本身有
穩定的超額報酬。
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)  # 資料不足或無漲跌時，給中性值 50


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14):
    high, low = df["High"], df["Low"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr = true_range(df)
    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_ = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx_.fillna(0), plus_di.fillna(0), minus_di.fillna(0)


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def compute_indicator_set(df: pd.DataFrame) -> pd.DataFrame:
    """一次算好所有指標，回傳一個跟 df 同 index 的 DataFrame，供繪圖 / 評分 / ML 特徵共用。"""
    out = pd.DataFrame(index=df.index)
    close = df["Close"]

    out["rsi"] = rsi(close)
    macd_line, signal_line, hist = macd(close)
    out["macd_line"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    out["ema20"] = ema(close, 20)
    out["ema50"] = ema(close, 50)
    out["ema200"] = ema(close, 200)

    adx_, plus_di, minus_di = adx(df)
    out["adx"] = adx_
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di

    out["atr"] = atr(df)
    out["obv"] = obv(df)

    return out


def confluence_signal(ind_df: pd.DataFrame, adx_trend_threshold: float = 20.0) -> dict:
    """
    判斷「最新一根K棒」的技術指標共振方向。
    回傳 {"bull_count": int, "bear_count": int, "trending": bool, "details": [...]}
    """
    row = ind_df.iloc[-1]
    bull_count, bear_count = 0, 0
    details = []

    if row["rsi"] > 50:
        bull_count += 1
        details.append(f"RSI {row['rsi']:.1f} > 50（動能偏多）")
    elif row["rsi"] < 50:
        bear_count += 1
        details.append(f"RSI {row['rsi']:.1f} < 50（動能偏空）")

    if row["macd_hist"] > 0:
        bull_count += 1
        details.append("MACD 柱狀圖為正（多頭動能）")
    elif row["macd_hist"] < 0:
        bear_count += 1
        details.append("MACD 柱狀圖為負（空頭動能）")

    if row["ema20"] > row["ema50"] > row["ema200"]:
        bull_count += 1
        details.append("EMA20 > EMA50 > EMA200（多頭排列）")
    elif row["ema20"] < row["ema50"] < row["ema200"]:
        bear_count += 1
        details.append("EMA20 < EMA50 < EMA200（空頭排列）")

    trending = bool(row["adx"] >= adx_trend_threshold)
    if trending:
        if row["plus_di"] > row["minus_di"]:
            bull_count += 1
            details.append(f"ADX {row['adx']:.1f}（趨勢成立）且 +DI > -DI")
        else:
            bear_count += 1
            details.append(f"ADX {row['adx']:.1f}（趨勢成立）且 -DI > +DI")
    else:
        details.append(f"ADX {row['adx']:.1f} < {adx_trend_threshold}（盤整，趨勢訊號打折）")

    return {"bull_count": bull_count, "bear_count": bear_count, "trending": trending, "details": details}
