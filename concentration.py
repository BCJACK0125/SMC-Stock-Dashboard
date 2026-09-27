# -*- coding: utf-8 -*-
"""
concentration.py
=================
集中度監控：同時出現多個訊號時，提醒它們其實是同一個風險。

為什麼這個專案需要它：
    使用者的操作模式是「平常持有現金、有訊號才進場」，所以不做資金配置。
    但這不代表集中度不重要——**同一天出現 5 個訊號時，如果那 5 檔高度
    相關，實際上等於在同一個賭注上押 5 倍的錢**，而看起來像是分散了。

    實測使用者的 22 檔清單：

        日報酬平均相關 0.28（最高一對 0.92）
        AI/半導體/光通訊 13 檔彼此 0.34
        有效獨立標的數 ≈ 3.2 檔（名目 22 檔）

    分散效果只有名目的七分之一。這個數字跟「訊號有沒有效」完全無關，
    但對實際會不會虧大錢影響更直接。

核心指標是**有效獨立標的數**：

    N_eff = 1 / (wᵀ Σ w)      w = 等權重向量, Σ = 相關矩陣

    完全不相關的 N 檔 → N_eff = N
    完全相同的 N 檔   → N_eff = 1

    同時進場 5 檔但 N_eff 只有 1.4，代表實質上是 1.4 個部位的風險，
    卻用了 5 份資金。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def correlation_matrix(
    prices: Dict[str, pd.Series], lookback: int = 252
) -> pd.DataFrame:
    """用最近 lookback 根 K 棒的日報酬算相關矩陣。"""
    rets = {}
    for sym, s in prices.items():
        r = s.pct_change().dropna()
        if len(r) >= 30:
            rets[sym] = r.iloc[-lookback:]
    if len(rets) < 2:
        return pd.DataFrame()
    return pd.DataFrame(rets).dropna().corr()


def effective_independent_count(corr: pd.DataFrame,
                                symbols: Optional[Sequence[str]] = None) -> Optional[float]:
    """
    有效獨立標的數 = 1 / (wᵀΣw)，等權重。

    這個數字回答的是：「這幾檔加起來，相當於幾個互不相關的部位？」
    """
    if corr.empty:
        return None
    if symbols is not None:
        cols = [s for s in symbols if s in corr.index]
        if len(cols) < 1:
            return None
        if len(cols) == 1:
            return 1.0
        corr = corr.loc[cols, cols]
    n = len(corr)
    w = np.ones(n) / n
    denom = float(w @ corr.to_numpy() @ w)
    return (1.0 / denom) if denom > 0 else None


def assess_signals(
    active_symbols: Sequence[str],
    corr: pd.DataFrame,
    high_corr_threshold: float = 0.6,
) -> dict:
    """
    評估「今天同時出現的這些訊號」實際上是幾個獨立風險。

    回傳可直接放進通知信的診斷。
    """
    syms = [s for s in active_symbols if s in corr.index]
    out = {
        "n_signals": len(active_symbols),
        "n_measured": len(syms),
        "effective_n": None,
        "concentration_ratio": None,
        "high_corr_pairs": [],
        "avg_corr": None,
    }
    if len(syms) < 2:
        out["effective_n"] = float(len(syms)) if syms else 0.0
        return out

    sub = corr.loc[syms, syms]
    eff = effective_independent_count(sub)
    out["effective_n"] = eff
    # 集中度比率：名目檔數 ÷ 有效獨立檔數。1.0 = 完全分散
    out["concentration_ratio"] = (len(syms) / eff) if eff else None

    iu = np.triu_indices(len(syms), 1)
    vals = sub.to_numpy()[iu]
    out["avg_corr"] = float(vals.mean())
    for a, b, v in zip(np.array(syms)[iu[0]], np.array(syms)[iu[1]], vals):
        if v >= high_corr_threshold:
            out["high_corr_pairs"].append((str(a), str(b), float(v)))
    out["high_corr_pairs"].sort(key=lambda x: -x[2])
    return out


def describe(assessment: dict, max_pairs: int = 5) -> List[str]:
    """把評估結果翻成可以直接讀的提醒。"""
    n = assessment.get("n_signals", 0)
    eff = assessment.get("effective_n")
    if n < 2 or eff is None:
        return []

    lines = []
    ratio = assessment.get("concentration_ratio") or 1.0
    lines.append(
        f"今天有 {n} 個訊號，但以近一年相關性計算，"
        f"實質上只相當於 **{eff:.1f} 個獨立風險**"
        f"（平均相關 {assessment['avg_corr']:.2f}）。")

    if ratio >= 2.5:
        lines.append(
            f"⚠️ 集中度偏高（名目 {n} 檔 ÷ 有效 {eff:.1f} = {ratio:.1f}x）。"
            f"每檔用相同部位大小的話，投入的是 {n} 份資金，"
            f"但分散效果只有 {eff:.1f} 份——建議減少同時進場的檔數，"
            f"或把每檔部位縮到平常的 {1 / ratio:.0%}，"
            f"讓總風險回到 {eff:.1f} 份的水準。")
    elif ratio >= 1.6:
        lines.append(
            f"ℹ️ 集中度中等（{ratio:.1f}x）。這幾檔有明顯共同走勢，"
            f"同時進場等於加大同一個賭注。")
    else:
        lines.append(f"✅ 這幾檔相關性不高（{ratio:.1f}x），分散效果接近名目檔數。")

    pairs = assessment.get("high_corr_pairs", [])[:max_pairs]
    if pairs:
        lines.append("高度相關的配對（基本上是同一個標的的兩種買法）：")
        lines.extend(f"    {a} ↔ {b}  相關 {v:.2f}" for a, b, v in pairs)
    return lines
