# -*- coding: utf-8 -*-
"""
exit_policy.py
===============
把「你能接受多大的回撤」換算成「移動停損要多寬」。

為什麼是一張靜態校準表，而不是每天重算：
    逐檔挑最佳出場寬度**完全行不通**。實測用 2016~2020 挑、2021~2025 驗證，
    21 檔裡只有 1 檔（5%）挑對——比隨機猜 8 選 1（12.5%）還差。逐檔只有
    20~60 筆交易，從 8 個寬度裡挑最好的就是在挑雜訊。

    全域挑（所有標的共用一個寬度）才穩定：同樣的樣本外分割，寬度排名的
    Spearman 相關是 +0.74。所以這裡挑的是一個**共用**寬度。

    表是靜態的，因為若每天用當天資料重挑，選擇本身就會污染同一份資料
    算出來的回測結果——那正是這個專案一路在防的選擇偏誤。要更新表，
    重跑 EXPERIMENTS.md 第十三階段記錄的腳本。

⚠️ 這個預算是**傾向，不是保證**。見 OOS_BREACH：預算設 20% 時，樣本外
   仍有 18/22 檔的實際回撤超過 20%，中位數是 −31%。真正的結論是
   **回撤壓不下去，能調的其實是報酬**——停損收得太緊只會少賺，不會少跌。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# (回撤預算, ATR 倍數, 預期年化, 預期回撤, 樣本外實測回撤中位, 樣本外超標檔數/總數)
#
# 資料來源：22 檔 × 10 年日線、門檻 50、逐根市價評估、逐檔不重疊複利後取
# 中位數。「預期」欄是全樣本，「樣本外」欄是用 2016~2020 挑、2021~2025 驗證。
CALIBRATION: List[Tuple[float, float, float, float, float, str]] = [
    (0.20, 1.0, 0.035, -0.180, -0.311, "18/22"),
    (0.25, 1.0, 0.035, -0.180, -0.340, "18/22"),
    (0.30, 2.0, 0.049, -0.277, -0.340, "15/22"),
    (0.35, 2.5, 0.083, -0.329, -0.340, "10/22"),
    (0.40, 5.0, 0.153, -0.387, -0.340, "9/22"),
    (0.45, 5.0, 0.153, -0.387, -0.340, "5/22"),
    (0.50, 5.0, 0.153, -0.387, -0.340, "4/22"),
    (0.60, 5.0, 0.153, -0.387, -0.340, "0/22"),
]


def _row(budget: float) -> Tuple[float, float, float, float, float, str]:
    """取「不超過預算」的最後一列；預算比最小值還小時退回最保守的一列。"""
    chosen = CALIBRATION[0]
    for row in CALIBRATION:
        if row[0] <= budget + 1e-9:
            chosen = row
    return chosen


def trail_mult_for(budget: float) -> float:
    """回撤預算 → 移動停損的 ATR 倍數。"""
    return _row(budget)[1]


def describe(budget: float) -> Dict:
    """給儀表板用的完整說明，含誠實的超標揭露。"""
    b, mult, cagr, dd, oos_dd, breach = _row(budget)
    return {
        "budget": budget,
        "matched_budget": b,
        "trail_atr_mult": mult,
        "expected_cagr": cagr,
        "expected_drawdown": dd,
        "oos_drawdown": oos_dd,
        "oos_breach": breach,
        # 預算調更低也換不到更低的回撤，只會換到更低的報酬
        "floor_reached": mult <= CALIBRATION[0][1],
    }
