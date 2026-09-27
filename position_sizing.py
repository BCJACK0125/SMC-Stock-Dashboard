# -*- coding: utf-8 -*-
"""
position_sizing.py
===================
依「這檔標的先碰停利還是先碰停損」的歷史機率，給出保守的部位大小建議。

為什麼這個模組跟系統其他部分不同：
    它**不依賴訊號有預測力**。整個專案反覆驗證的結論是 SMC 訊號沒有
    可證實的選時能力（超額期望值 ≈ 0），但這個模組用的不是訊號，而是
    **標的本身對這套出場結構的適配度**。

    實測（22 檔、日線 10 年、隨機進場對照）：在 2:1 的停損停利結構下，
    各檔先碰停利的機率從 27% 到 62% 不等——

        AVGO 62%   MSFT 53%   0050 53%   2330 51%
        AMZN 35%   NOK  35%   LITE 36%   AAOI 27%

    這個差距（35 個百分點）反映的是各檔的漂移與波動結構，與訊號無關。
    用同一個部位大小去做 AVGO 和 AMZN 本來就不合理。

⚠️ 為什麼刻意做得保守：

    1. **用隨機進場的機率當基準，不用訊號進場的。**
       訊號進場的超額只有 +3.1 個百分點且不顯著（CI [−1.7, +8.0]），
       而隨機進場的樣本大 4 倍、估計穩定得多。把超額當真會高估。

    2. **代入信賴下界，不用點估計。**
       每檔樣本只有 14~34 筆，P 的信賴區間寬達 30~40 個百分點。

    3. **只給 1/4 Kelly。**
       Kelly 對機率估計極度敏感：高估 p 會導致過度下注，而過度下注的
       破產風險是非線性上升的。全 Kelly 在實務上幾乎沒人用。

    三層保守疊起來的結果會比「理論最適」小很多——這是刻意的。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


def kelly_fraction(p: float, rr: float) -> float:
    """
    Kelly 最適下注比例。

    p  : 獲勝機率（先碰停利）
    rr : 賠率（停利距離 ÷ 停損距離，例如 2.0 表示 2:1）

    f* = (p·rr − (1−p)) / rr
    """
    if rr <= 0:
        return 0.0
    f = (p * rr - (1 - p)) / rr
    return max(0.0, f)


def wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> Optional[float]:
    """機率的 Wilson 下界。小樣本時點估計極不穩定，一律用下界。"""
    if n <= 0:
        return None
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


def suggest_position(
    wins: int,
    n: int,
    rr: float = 2.0,
    *,
    kelly_divisor: float = 4.0,
    max_fraction: float = 0.20,
    min_samples: int = 20,
) -> dict:
    """
    回傳這檔標的的建議部位比例與判斷依據。

    wins / n : 歷史上「先碰停利」的次數與總次數（建議用隨機進場的統計，
               樣本較大且不受訊號超額的樂觀偏誤影響）
    """
    out = {
        "n": n, "wins": wins, "rr": rr, "kelly_divisor": kelly_divisor,
        "p_point": (wins / n) if n else None,
        "p_lower": wilson_lower_bound(wins, n),
        "kelly_full": None, "kelly_used": None,
        "fraction": 0.0, "reason": "",
    }

    if n < min_samples:
        out["reason"] = f"樣本不足（{n} < {min_samples} 筆），無法估計"
        return out

    p_lo = out["p_lower"]
    f_full = kelly_fraction(p_lo, rr)
    out["kelly_full"] = f_full
    out["kelly_used"] = f_full / kelly_divisor

    # 損益兩平所需的機率：低於它代表這個出場結構對這檔不利
    breakeven = 1.0 / (1.0 + rr)
    if p_lo <= breakeven:
        out["reason"] = (f"機率下界 {p_lo:.0%} 未超過 {rr:.0f}:1 的損益兩平點 "
                         f"{breakeven:.0%}，這個出場結構對本標的不利")
        return out

    out["fraction"] = min(out["kelly_used"], max_fraction)
    out["reason"] = (f"機率下界 {p_lo:.0%} > 兩平點 {breakeven:.0%}；"
                     f"全 Kelly {f_full:.0%} ÷ {kelly_divisor:.0f} = "
                     f"{out['kelly_used']:.0%}"
                     + (f"，受上限 {max_fraction:.0%} 限制" if out["kelly_used"] > max_fraction else ""))
    return out


def describe(symbol: str, s: dict) -> str:
    """一行人話說明，給通知信與儀表板用。"""
    if s["fraction"] <= 0:
        return f"{symbol}：不建議下注（{s['reason']}）"
    div = s.get("kelly_divisor", 4.0)
    return (f"{symbol}：建議部位 {s['fraction']:.0%}"
            f"（歷史 {s['wins']}/{s['n']} 先碰停利，下界 {s['p_lower']:.0%}，"
            f"1/{div:.0f} Kelly）")


def barrier_stats(outcomes: Sequence[str]) -> dict:
    """把 'target'/'stop'/'timeout' 的序列整理成勝負次數。"""
    arr = list(outcomes)
    resolved = [o for o in arr if o in ("target", "stop")]
    return {
        "n_total": len(arr),
        "n_resolved": len(resolved),
        "wins": sum(1 for o in resolved if o == "target"),
        "timeout_rate": (sum(1 for o in arr if o == "timeout") / len(arr)) if arr else 0.0,
    }


# ---------------------------------------------------------------------------
# 隨機進場基準
#
# 為什麼要用隨機進場而不是訊號進場來估 P：
#   實測訊號進場相對隨機進場的超額只有 +3.1 個百分點，而且不顯著
#   （95% CI [−1.7, +8.0]）。既然證明不了訊號有貢獻，用訊號進場的統計
#   等於把不顯著的超額當真，會系統性高估部位。
#
#   隨機進場的樣本可以任意放大（實測用 4 倍），估計穩定得多，而且它量到
#   的是**這檔標的對這套出場結構的適配度**——那是真實存在、與訊號無關的
#   性質（實測 22 檔從 27% 到 62%）。
# ---------------------------------------------------------------------------
def simulate_barrier(high, low, entry: float, stop: float, target: float) -> str:
    """
    單筆交易的三重障礙結果。high/low 是進場之後的序列（已截到時間上限）。
    同一根同時觸及一律算停損先到——用 OHLC 無法還原日內順序，
    反過來假設會系統性高估績效。
    """
    for h, lo in zip(high, low):
        if lo <= stop:
            return "stop"
        if h >= target:
            return "target"
    return "timeout"


def random_entry_barrier_stats(
    df, risk_pcts: Sequence[float], *, rr: float = 2.0,
    max_bars: int = 250, warmup: int = 250, n_samples: int = 400,
    seed: int = 42,
) -> dict:
    """
    在這檔標的上隨機挑進場日，套用同樣的停損/停利結構，統計先碰哪一邊。

    risk_pcts: 實際訊號交易的停損幅度分布（用來抽樣，讓對照組的風險
               幅度與真實交易一致，而不是憑空設一個數字）。
    """
    n = len(df)
    lo_i, hi_i = warmup, n - max_bars - 2
    if hi_i <= lo_i or len(risk_pcts) == 0:
        return {"n_total": 0, "n_resolved": 0, "wins": 0, "timeout_rate": 0.0}

    rng = np.random.default_rng(seed)
    H = df["High"].to_numpy()
    L = df["Low"].to_numpy()
    C = df["Close"].to_numpy()
    risks = np.asarray(list(risk_pcts), dtype=float)

    outcomes = []
    for t in rng.integers(lo_i, hi_i, size=n_samples):
        e = float(C[t])
        rp = float(rng.choice(risks))
        if e <= 0 or rp <= 0:
            continue
        end = min(t + 1 + max_bars, n)
        outcomes.append(simulate_barrier(
            H[t + 1:end], L[t + 1:end], e, e * (1 - rp), e * (1 + rr * rp)))
    return barrier_stats(outcomes)
