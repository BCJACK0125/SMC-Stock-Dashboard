# -*- coding: utf-8 -*-
"""
backtest.py
============
Walk-forward（滾動式樣本外）回測引擎，用來回答兩個問題：
    1. 用「我們現在這套 SMC + 技術指標 + ML」的評分邏輯，在這檔標的過去的
       歷史資料上，各個分數門檻歷史上分別出現過幾次訊號、勝率各是多少？
    2. 依此幫每一檔標的挑一個「入場次數 vs 勝率」平衡後的建議門檻，
       而不是全部標的都用同一個寫死的 ALERT_THRESHOLD。

方法論（參考業界公認的 walk-forward analysis「金標準」）：
    - 對歷史上每一根K棒 t（跳過暖身期），只用「t 當下已經可以知道」的資訊
      去算 bull_score / bear_score（不可以用到 t 之後才確認的 swing／
      Order Block／FVG／ADX 等等）。
    - 用 t 之後第 horizon 根K棒的報酬，判斷「如果那時候進場，結果是贏還輸」。
    - 對一組候選門檻（例如 35, 40, 45, ... 85）分別統計「訊號次數」與「勝率」，
      在樣本數足夠、勝率達標的門檻中，選相對寬鬆（訊號較多）的那個。

⚠️ 誠實的局限（務必知道）：
    - 這是「單一次」walk-forward，不是業界更嚴謹的「多段滾動」walk-forward
      optimization（那個作法是切成好幾段 in-sample/out-of-sample 交替驗證，
      這裡因為單一標的歷史資料量有限、且要在 GitHub Actions 的時間預算內
      跑完所有標的，所以簡化成一次性、單向的樣本外回測）。
    - ML 模型為了控制運算量，只有每隔 `ml_retrain_every` 根K棒才重新訓練一次
      （而不是每根K棒都重訓），兩次重訓之間沿用同一顆模型。
    - 完全沒有考慮交易成本、滑價、稅務，勝率是「未來收盤價方向」的勝率，
      不是實際下單後的損益勝率。
    - 歷史勝率不保證未來績效，樣本數少的標的（例如新上市股票、資料很短）
      這個回測本身也不可靠——所以才需要 `min_trades` 門檻，樣本太少就不採信。
"""

from __future__ import annotations
from dataclasses import replace
from typing import List, Optional
import pandas as pd

from smc.analyzer import SMCAnalyzer, OrderBlock, FairValueGap
import ml_model
from notifier import score as score_fn


# ---------------------------------------------------------------------------
# AsOfView：把 analyzer 在「完整歷史」上算出來的結果，重新過濾成
# 「只用截至時間點 t 為止已知資訊」的快照，介面跟 SMCAnalyzer 相容，
# 這樣可以直接餵給 notifier.score()，確保回測跟正式評分用的是同一套邏輯。
# ---------------------------------------------------------------------------
class AsOfView:
    def __init__(self, analyzer: SMCAnalyzer, t_idx: int, ts: pd.Timestamp):
        self.df = analyzer.df.iloc[: t_idx + 1]
        self.current_trend = None  # 目前評分邏輯用不到，留空即可

        self._order_blocks = self._filter_obs(analyzer.order_blocks, ts)
        self._fvgs = self._filter_fvgs(analyzer.fvgs, ts)
        self._structure_events = [ev for ev in analyzer.structure_events if ev.index <= ts]
        self.liquidity_pools = [
            p for p in analyzer.liquidity_pools
            if p.confirmed_index is not None and p.confirmed_index <= ts
        ]
        self.current_zone = self._zone_as_of(analyzer, ts)

    @staticmethod
    def _filter_obs(obs: List[OrderBlock], ts: pd.Timestamp) -> List[OrderBlock]:
        out = []
        for ob in obs:
            if ob.caused_structure is None or ob.caused_structure.index > ts:
                continue  # 這個 OB 依附的結構事件，在 ts 當下還沒發生
            mitigated_as_of_t = ob.mitigated_index is not None and ob.mitigated_index <= ts
            out.append(replace(ob, mitigated=mitigated_as_of_t))
        return out

    @staticmethod
    def _filter_fvgs(fvgs: List[FairValueGap], ts: pd.Timestamp) -> List[FairValueGap]:
        out = []
        for fvg in fvgs:
            if fvg.end_index > ts:
                continue  # 第三根K棒都還沒收盤，這個缺口還不存在
            filled_as_of_t = fvg.filled_index is not None and fvg.filled_index <= ts
            out.append(replace(fvg, filled=filled_as_of_t))
        return out

    @staticmethod
    def _zone_as_of(analyzer: SMCAnalyzer, ts: pd.Timestamp) -> Optional[dict]:
        confirmed_swings = [s for s in analyzer.swings if s.confirmed_index <= ts]
        if len(confirmed_swings) < 2:
            return None
        recent_high = next((s for s in reversed(confirmed_swings) if s.kind == "high"), None)
        recent_low = next((s for s in reversed(confirmed_swings) if s.kind == "low"), None)
        if recent_high is None or recent_low is None:
            return None
        top, bottom = max(recent_high.price, recent_low.price), min(recent_high.price, recent_low.price)
        mid = (top + bottom) / 2
        return {"top": top, "bottom": bottom, "mid": mid, "zone": None}  # zone 由呼叫端依收盤價自己判斷

    # -- 跟 SMCAnalyzer 相容的介面 --
    def active_order_blocks(self, side: Optional[str] = None):
        obs = [ob for ob in self._order_blocks if not ob.mitigated]
        return [ob for ob in obs if ob.side == side] if side else obs

    def active_fvgs(self, side: Optional[str] = None):
        gaps = [g for g in self._fvgs if not g.filled]
        return [g for g in gaps if g.side == side] if side else gaps

    def last_structure_event(self):
        return self._structure_events[-1] if self._structure_events else None


def _finalize_zone(view: AsOfView, last_close: float) -> None:
    """current_zone 建好後，還要依照『t 當下的收盤價』決定是溢價/折價/均衡。"""
    if view.current_zone is None:
        return
    mid = view.current_zone["mid"]
    if last_close > mid:
        view.current_zone["zone"] = "premium"
    elif last_close < mid:
        view.current_zone["zone"] = "discount"
    else:
        view.current_zone["zone"] = "equilibrium"


def walk_forward_scores(
    df: pd.DataFrame,
    analyzer: SMCAnalyzer,
    ind_df: pd.DataFrame,
    horizon: int = 5,
    warmup: int = 90,
    ml_enabled: bool = True,
    ml_retrain_every: int = 10,
    ml_min_train_rows: int = 80,
    adx_trend_threshold: float = 20.0,
    recent_bars: int = 5,
) -> pd.DataFrame:
    """
    逐根K棒（跳過暖身期）計算「當下已知資訊」的 bull_score / bear_score，
    以及 horizon 根K棒之後的實際報酬率，回傳一個 DataFrame 方便後續做門檻掃描。
    """
    n = len(df)
    warmup = max(warmup, 30)
    end = n - horizon
    if end <= warmup:
        return pd.DataFrame(columns=["ts", "bull_score", "bear_score", "future_return"])

    records = []
    cached_ml_result = None
    last_retrain_t = -10**9

    for t in range(warmup, end):
        ts = df.index[t]
        view = AsOfView(analyzer, t, ts)
        last_close = float(df["Close"].iloc[t])
        _finalize_zone(view, last_close)

        ind_slice = ind_df.iloc[: t + 1]

        ml_result = None
        if ml_enabled:
            if cached_ml_result is None or (t - last_retrain_t) >= ml_retrain_every:
                try:
                    ml_result = ml_model.predict_next_move_probability(
                        df.iloc[: t + 1], ind_slice,
                        horizon=horizon, min_train_rows=ml_min_train_rows,
                    )
                except Exception:
                    ml_result = None
                cached_ml_result = ml_result
                last_retrain_t = t
            else:
                ml_result = cached_ml_result

        s = score_fn(
            view, recent_bars=recent_bars, ind_df=ind_slice,
            ml_result=ml_result, adx_trend_threshold=adx_trend_threshold,
        )

        future_return = float(df["Close"].iloc[t + horizon]) / last_close - 1
        records.append((ts, s["bull_score"], s["bear_score"], future_return))

    return pd.DataFrame(records, columns=["ts", "bull_score", "bear_score", "future_return"])


def evaluate_thresholds(
    wf_df: pd.DataFrame,
    candidate_thresholds: List[int],
    win_return_threshold: float = 0.0,
) -> dict:
    """
    對一組候選門檻，分別統計「多方訊號次數/勝率」與「空方訊號次數/勝率」。
    回傳 {"bull": [{"threshold":.., "n":.., "win_rate":..}, ...], "bear": [...]}
    """
    out = {"bull": [], "bear": []}
    if wf_df.empty:
        return out

    for th in candidate_thresholds:
        bull_rows = wf_df[wf_df["bull_score"] >= th]
        n_bull = len(bull_rows)
        win_rate_bull = float((bull_rows["future_return"] > win_return_threshold).mean()) if n_bull else None
        out["bull"].append({"threshold": th, "n": n_bull, "win_rate": win_rate_bull})

        bear_rows = wf_df[wf_df["bear_score"] >= th]
        n_bear = len(bear_rows)
        win_rate_bear = float((bear_rows["future_return"] < -win_return_threshold).mean()) if n_bear else None
        out["bear"].append({"threshold": th, "n": n_bear, "win_rate": win_rate_bear})

    return out


def recommend_threshold(
    stats: List[dict],
    min_trades: int,
    target_win_rate: float,
    default_threshold: int,
) -> dict:
    """
    在滿足『樣本數 >= min_trades』的門檻中：
        - 優先選「勝率達標且訊號最多（門檻最寬鬆）」的那個
        - 若沒有任何門檻達標，退而求其次選「樣本數足夠中，勝率最高」的門檻，
          但標記 confidence='low'
        - 若連樣本數足夠的門檻都沒有，回退到 default_threshold，
          標記 confidence='insufficient_data'
    """
    eligible = [s for s in stats if s["n"] >= min_trades and s["win_rate"] is not None]
    if not eligible:
        return {"threshold": default_threshold, "win_rate": None, "n": 0, "confidence": "insufficient_data"}

    meets_target = [s for s in eligible if s["win_rate"] >= target_win_rate]
    if meets_target:
        best = min(meets_target, key=lambda s: s["threshold"])  # 門檻越低代表訊號越多，優先採用
        return {**best, "confidence": "ok"}

    best_fallback = max(eligible, key=lambda s: s["win_rate"])
    return {**best_fallback, "confidence": "low"}


def run_symbol_backtest(df: pd.DataFrame, analyzer: SMCAnalyzer, ind_df: pd.DataFrame, config) -> dict:
    """整合以上步驟，回傳這檔標的的完整回測結果與建議門檻（多空分開）。"""
    wf_df = walk_forward_scores(
        df, analyzer, ind_df,
        horizon=config.BACKTEST_HORIZON_BARS,
        warmup=config.BACKTEST_WARMUP_BARS,
        ml_enabled=config.ML_ENABLED,
        ml_retrain_every=config.BACKTEST_ML_RETRAIN_EVERY,
        ml_min_train_rows=config.ML_MIN_TRAIN_ROWS,
        adx_trend_threshold=config.ADX_TREND_THRESHOLD,
        recent_bars=config.RECENT_BARS_FOR_SCORE,
    )

    stats = evaluate_thresholds(
        wf_df, config.BACKTEST_CANDIDATE_THRESHOLDS,
        win_return_threshold=config.BACKTEST_WIN_RETURN_THRESHOLD,
    )

    bull_rec = recommend_threshold(
        stats["bull"], config.BACKTEST_MIN_TRADES, config.BACKTEST_TARGET_WIN_RATE,
        config.ALERT_THRESHOLD,
    )
    bear_rec = recommend_threshold(
        stats["bear"], config.BACKTEST_MIN_TRADES, config.BACKTEST_TARGET_WIN_RATE,
        config.ALERT_THRESHOLD,
    )

    return {
        "n_evaluated_bars": len(wf_df),
        "bull": bull_rec,
        "bear": bear_rec,
        "threshold_sweep": stats,
    }
