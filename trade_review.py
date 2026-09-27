# -*- coding: utf-8 -*-
"""
trade_review.py
================
交易覆盤：分析每一筆進出場的品質，找出「哪裡沒做好」。

跟回測的差別：
    回測問的是「這套訊號有沒有優勢」——答案已經確定是沒有。
    覆盤問的是「既然要交易，這些進出場本身做得好不好」——這是**即使訊號
    沒有預測力也依然有意義**的問題。停損放太緊會把本來會贏的單洗掉；
    目標設太近會把趨勢單提早砍掉；進場點太差會讓每筆都先套一段。
    這三件事都可以獨立改善，而且改善幅度是可以量化的。

核心工具是 MAE / MFE（業界標準的交易後分析）：
    MAE (Maximum Adverse Excursion)  持倉期間最大的浮虧
    MFE (Maximum Favorable Excursion) 持倉期間最大的浮盈

    這兩個數字能回答很具體的問題：
      - 最後獲利的單子，過程中最多套了多少？
        → 如果贏家的 MAE 都很淺，停損可以收緊，R:R 直接變好
        → 如果贏家的 MAE 常常很深，停損放太緊就會把它們洗掉
      - 最後虧損的單子，過程中曾經賺到多少？
        → 如果虧損單常常先賺一段才回吐，代表**出場太慢**
      - 實際出場價 vs MFE 的差距
        → 這是「留在桌上的錢」，衡量出場效率
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    symbol: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry: float
    exit: float
    stop: float
    side: str
    outcome: str
    bars_held: int
    ret: float              # 已扣成本
    r_multiple: float
    mae: float              # 最大浮虧（正值，以報酬率表示）
    mfe: float              # 最大浮盈（正值）
    mae_r: float            # 以初始風險為單位
    mfe_r: float
    bars_to_mae: int
    bars_to_mfe: int
    entry_heat: float       # 進場後前 3 根內的最大浮虧——衡量進場時機


def measure_excursions(
    path: pd.DataFrame, entry: float, side: str, stop: float,
    cost: float = 0.0, early_bars: int = 3,
) -> dict:
    """
    量測一筆交易持倉期間的 MAE / MFE。

    path: 進場之後、到出場為止的 OHLC（不含進場棒）。
    """
    if len(path) == 0:
        return {"mae": 0.0, "mfe": 0.0, "bars_to_mae": 0, "bars_to_mfe": 0,
                "entry_heat": 0.0}

    long_ = side == "bullish"
    # 多單：最不利是 Low、最有利是 High；空單相反
    adverse = (entry - path["Low"]) / entry if long_ else (path["High"] - entry) / entry
    favorable = (path["High"] - entry) / entry if long_ else (entry - path["Low"]) / entry

    adverse = adverse.clip(lower=0).to_numpy()
    favorable = favorable.clip(lower=0).to_numpy()

    return {
        "mae": float(adverse.max()),
        "mfe": float(favorable.max()),
        "bars_to_mae": int(adverse.argmax()) + 1,
        "bars_to_mfe": int(favorable.argmax()) + 1,
        # 進場後前幾根的浮虧：進場時機好不好的直接指標。
        # 如果每筆一進場就先套，代表訊號在追高（或摸底摸太早）。
        "entry_heat": float(adverse[:early_bars].max()),
    }


def summarize_review(trades: List[TradeRecord]) -> dict:
    """把一組交易彙總成可以行動的診斷指標。"""
    if not trades:
        return {}

    df = pd.DataFrame([t.__dict__ for t in trades])
    win = df[df["ret"] > 0]
    loss = df[df["ret"] <= 0]

    out = {
        "n_trades": len(df),
        "win_rate": float((df["ret"] > 0).mean()),
        "avg_ret": float(df["ret"].mean()),
        "outcome_mix": df["outcome"].value_counts(normalize=True).to_dict(),

        # 贏家過程中最多套多少 —— 決定停損能收多緊
        "winner_mae_median": float(win["mae"].median()) if len(win) else None,
        "winner_mae_p90": float(win["mae"].quantile(0.9)) if len(win) else None,
        "winner_mae_r_median": float(win["mae_r"].median()) if len(win) else None,

        # 輸家過程中曾經賺多少 —— 判斷是不是出場太慢
        "loser_mfe_median": float(loss["mfe"].median()) if len(loss) else None,
        "loser_ever_profitable": float((loss["mfe"] > 0.01).mean()) if len(loss) else None,

        # 出場效率：實際拿到的 vs 最好的時候。
        # 用「總實現 / 總浮盈」而不是「逐筆比值的中位數」——交易報酬是
        # 高度右偏的（少數大贏、多數小輸），逐筆比值的中位數會被一堆小額
        # 交易主導，算出接近 0 的假象。
        "mfe_capture": (float(df["ret"].sum() / df["mfe"].sum())
                        if df["mfe"].sum() > 0 else None),
        "mfe_capture_median_ratio": float((df["ret"] / df["mfe"].replace(0, np.nan)).median()),
        "median_mfe": float(df["mfe"].median()),
        "median_realized": float(df["ret"].median()),

        # 進場時機
        "entry_heat_median": float(df["entry_heat"].median()),
        "immediately_underwater": float((df["entry_heat"] > 0.01).mean()),
    }
    return out


def diagnose(
    summary: dict,
    exit_mode: str = "fixed",
    baseline: Optional[dict] = None,
) -> List[str]:
    """
    把彙總指標翻譯成具體的診斷與建議。

    exit_mode: "fixed" 或 "trailing"。兩者的健康指標完全不同——移動停損
               模式下「100% 由停損出場」是正常設計（趨勢反轉才出場），
               不是警訊；在固定目標模式下才代表停損被打爆。

    baseline:  隨機進場的對照組（同樣的標的與期間）。**進場時機一定要有
               對照才能診斷**：高波動股票任何進場點都會先吃幾個百分點的
               震盪，沒有基準就會把正常波動誤判成「訊號在追高」。
               實測就發生過：訊號 66% vs 隨機 67%，其實沒有差別。
    """
    msgs = []
    if not summary:
        return ["樣本不足，無法診斷"]

    # --- 停損 ---
    wm = summary.get("winner_mae_r_median")
    if wm is not None:
        if wm < 0.35:
            msgs.append(
                f"✅ 停損可以收緊：獲利單的 MAE 中位數只有初始風險的 {wm:.0%}，"
                f"代表贏家很少深套。把停損收到約 {max(wm * 1.5, 0.4):.0%} 風險"
                f"可以直接改善 R:R，而且只會洗掉少數贏家。")
        elif wm > 0.7:
            msgs.append(
                f"⚠️ 停損偏緊：獲利單的 MAE 中位數已達初始風險的 {wm:.0%}，"
                f"贏家在過程中經常逼近停損。再收緊會把它們洗掉。")

    # --- 出場 ---
    cap = summary.get("mfe_capture")
    if cap is not None:
        if cap < 0.35:
            msgs.append(
                f"⚠️ 出場效率偏低：總實現只有總浮盈的 {cap:.0%}。"
                f"可考慮分批出場，或收緊移動停損距離少讓回吐。")
        elif cap > 0.6:
            msgs.append(f"✅ 出場效率良好：總實現達總浮盈的 {cap:.0%}。")

    lp = summary.get("loser_ever_profitable")
    if lp is not None and lp > 0.5:
        msgs.append(
            f"⚠️ 出場太慢：{lp:.0%} 的虧損單在過程中曾經浮盈超過 1%，"
            f"最後卻是虧的（浮盈中位數 {summary['loser_mfe_median']:.1%}）。"
            f"加入保本停損（浮盈達 1R 後把停損移到成本價）是最直接的對策。")

    # --- 進場：一定要有對照組 ---
    iu = summary.get("immediately_underwater")
    if iu is None:
        pass
    elif baseline is None or baseline.get("immediately_underwater") is None:
        msgs.append(
            f"ℹ️ 進場後 3 根內浮虧 >1% 的比例為 {iu:.0%}，但**沒有隨機進場的"
            f"對照組，無法判斷這是訊號的問題還是標的本身的波動**。"
            f"高波動股票任何進場點都會先吃幾個百分點。")
    else:
        b = baseline["immediately_underwater"]
        d = iu - b
        if d > 0.08:
            msgs.append(
                f"⚠️ 進場時機偏差：{iu:.0%} 的交易一進場就明顯浮虧，"
                f"高於隨機進場的 {b:.0%}（差 {d:+.0%}）。訊號可能在追高。")
        elif d < -0.08:
            msgs.append(f"✅ 進場時機優於隨機：{iu:.0%} vs 隨機 {b:.0%}。")
        else:
            msgs.append(
                f"➖ 進場時機與隨機無異：{iu:.0%} vs 隨機 {b:.0%}（差 {d:+.0%}）。"
                f"這不是缺點也不是優點——代表這幾個百分點是標的本身的波動，"
                f"不是進場點選得差，改用限價單等回踩並不會解決什麼。")

    # --- 出場方式組成（依模式判讀）---
    mix = summary.get("outcome_mix", {})
    if exit_mode == "trailing":
        if mix.get("stop", 0) > 0.9:
            msgs.append(
                f"ℹ️ {mix['stop']:.0%} 由移動停損出場——這是移動停損模式的"
                f"正常行為（趨勢反轉才出場），不是停損被打爆。")
    else:
        if mix.get("timeout", 0) > 0.6:
            msgs.append(
                f"ℹ️ {mix['timeout']:.0%} 以時間停損出場，代表目標距離對這個"
                f"持有期設得太遠。")
        if mix.get("stop", 0) > 0.45:
            msgs.append(f"⚠️ {mix['stop']:.0%} 觸及停損，比例偏高。")

    return msgs or ["各項指標都在正常範圍，沒有明顯可改善處。"]
