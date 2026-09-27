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
import math
from dataclasses import replace
from typing import List, Optional
import numpy as np
import pandas as pd

from smc.analyzer import SMCAnalyzer, OrderBlock, FairValueGap, LiquidityPool
import ml_model
import performance
import trade_model
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
        self.liquidity_pools = self._filter_pools(analyzer.liquidity_pools, ts)
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
    def _filter_pools(pools: List[LiquidityPool], ts: pd.Timestamp) -> List[LiquidityPool]:
        out = []
        for p in pools:
            if p.confirmed_index is None or p.confirmed_index > ts:
                continue  # 兩個 swing 還沒都被確認，這個池子在 ts 當下還不存在
            # swept/swept_index 是用「完整歷史」算的，回測時必須切回 t 當下的
            # 狀態，否則評分會看到未來才發生的掃蕩（前視偏差）。
            swept_as_of_t = p.swept_index is not None and p.swept_index <= ts
            out.append(replace(
                p,
                swept=swept_as_of_t,
                swept_index=p.swept_index if swept_as_of_t else None,
            ))
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
    max_holding_bars: int = 20,
    adx_trend_threshold: float = 20.0,
    recent_bars: int = 5,
    atr_series: Optional[pd.Series] = None,
    trade_cfg: Optional[dict] = None,
    exit_mode: str = "fixed",
    trail_atr_mult: float = 3.0,
    entry_cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """
    逐根K棒（跳過暖身期）計算「當下已知資訊」的 bull_score / bear_score。

    每根K棒都會輸出兩組結果：
      - future_return：固定持有 horizon 根的報酬（保留下來當基準與對照組）
      - long_* / short_*：帶停損與目標的模擬交易結果（見 trade_model）

    交易計畫只用 t 當下的 AsOfView 決定，出場才往後看K棒，所以沒有前視偏差。
    """
    n = len(df)
    warmup = max(warmup, 30)
    end = n - horizon
    if end <= warmup:
        return pd.DataFrame(columns=["ts", "bull_score", "bear_score", "future_return"])

    entry_cfg = entry_cfg or {}
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

        # -------- 帶停損/目標的模擬交易（多空各一筆）--------
        rec = {"ts": ts, "bull_score": s["bull_score"], "bear_score": s["bear_score"],
               "future_return": future_return}
        atr_now = float(atr_series.iloc[t]) if atr_series is not None else 0.0
        after_signal = df.iloc[t + 1:]

        # 隔天才進場：訊號是用當根收盤算的，收盤後才看得到，不可能用它成交
        entry_plan = trade_model.plan_entry(
            last_close, atr_now,
            offset_atr=entry_cfg.get("offset_atr", 0.5),
            valid_bars=entry_cfg.get("valid_bars", 1),
            fallback=entry_cfg.get("fallback", "market"),
        ) if atr_now > 0 else None
        fill = trade_model.resolve_entry(after_signal, entry_plan) if entry_plan else None
        if fill is None:
            entry_px, entry_off = last_close, -1     # 退回舊行為（無 ATR 時）
        else:
            entry_px, entry_off = fill
        future_bars = after_signal.iloc[entry_off + 1:]

        for side, prefix in (("bullish", "long"), ("bearish", "short")):
            plan = trade_model.plan_trade(view, side, entry_px, atr_now, **(trade_cfg or {}))
            if plan is None:
                rec.update({f"{prefix}_ret": 0.0, f"{prefix}_bars": horizon,
                            f"{prefix}_r": 0.0, f"{prefix}_outcome": "none",
                            f"{prefix}_rr": 0.0})
                continue
            if exit_mode == "trailing":
                res = trade_model.simulate_trailing_trade(
                    future_bars, plan, atr_now,
                    trail_atr_mult=trail_atr_mult, max_holding_bars=max_holding_bars)
            else:
                res = trade_model.simulate_trade(
                    future_bars, plan, max_holding_bars=max_holding_bars)
            rec.update({f"{prefix}_ret": res.return_pct,
                        # 持有長度要含進場等待，否則不重疊選取會重複進場
                        f"{prefix}_bars": max(res.bars_held + entry_off + 1, 1),
                        f"{prefix}_r": res.r_multiple,
                        f"{prefix}_outcome": res.outcome,
                        f"{prefix}_rr": plan.rr})
        records.append(rec)

    return pd.DataFrame(records)


def wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> Optional[float]:
    """
    勝率的 Wilson score interval 下界。

    為什麼不直接用 wins/n：樣本數少的時候，點估計極不穩定。
    8 筆裡贏 5 筆的點估計是 62.5%，聽起來不錯，但 95% 信賴區間大約是
    [24%, 91%]——跟「完全沒有資訊」幾乎沒兩樣。用下界來選門檻，等於要求
    「有統計證據支持這個勝率」，而不是「剛好在歷史上長這樣」，可以同時
    抑制小樣本雜訊與掃描多個門檻造成的 selection bias。

    Wilson 區間比常見的常態近似 (p ± z·sqrt(p(1-p)/n)) 更適合這裡的場景：
    樣本小、p 接近 0 或 1 時仍然不會跑出 [0, 1] 之外。
    """
    if n <= 0:
        return None
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, (center - margin) / denom)


def select_non_overlapping(mask: pd.Series, holding_bars: pd.Series) -> List[int]:
    """
    把「每根K棒都可能是訊號」壓縮成一組互不重疊的進場。

    為什麼必要：原本的做法是對每一根符合門檻的K棒都計一筆，但連續 10 根
    K棒的訊號其實是在賭同一段行情，它們的未來報酬高度重疊。這會讓樣本數
    嚴重灌水（實測 n 從 295 掉到 93），連帶讓信賴區間假性收窄。

    holding_bars: 每根K棒對應的持有長度（第一優先階段是固定 horizon，
                  第二優先加入停損/停利後會變成實際持有到出場的長度）。
    """
    picks: List[int] = []
    busy_until = -1
    for i, is_signal in enumerate(mask.to_numpy()):
        if is_signal and i > busy_until:
            picks.append(i)
            busy_until = i + int(holding_bars.iloc[i]) - 1
    return picks


def bootstrap_mean_lower_bound(
    values: "np.ndarray", alpha: float = 0.05, n_boot: int = 2000, seed: int = 42
) -> Optional[float]:
    """
    平均值的 bootstrap 百分位數下界。

    為什麼不用常態近似的 mean - z*sem：單筆交易報酬的分佈是明顯右偏、
    厚尾的（少數大賺、多數小輸小贏），常態假設會低估尾部風險。bootstrap
    不對分佈做假設，直接從樣本本身重抽，對這種形狀比較誠實。
    """
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha))


def summarize_trades(returns: "np.ndarray", cost: float, alpha: float) -> dict:
    """把一組單筆報酬換算成一整套績效統計（全部已扣除來回交易成本）。"""
    net = returns - cost
    n = len(net)
    if n == 0:
        return {"n": 0, "wins": 0, "win_rate": None, "win_rate_lb": None,
                "expectancy": None, "expectancy_lb": None,
                "avg_win": None, "avg_loss": None, "payoff": None,
                "profit_factor": None}

    wins_mask = net > 0
    wins = int(wins_mask.sum())
    gains = net[wins_mask]
    losses = net[~wins_mask]
    avg_win = float(gains.mean()) if len(gains) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0

    return {
        "n": n,
        "wins": wins,
        "win_rate": wins / n,
        "win_rate_lb": wilson_lower_bound(wins, n, z=_z_for_alpha(alpha)),
        # 期望值才是「這個門檻到底會不會賺錢」的直接答案：
        # 勝率高但賠率差的組合（實測 0050 勝率 59% 卻每筆虧 0.75%）
        # 在勝率判準下會過關，在期望值判準下不會。
        "expectancy": float(net.mean()),
        "expectancy_lb": bootstrap_mean_lower_bound(net, alpha=alpha),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff": abs(avg_win / avg_loss) if avg_loss else None,
        "profit_factor": (float(gains.sum()) / abs(float(losses.sum()))) if len(losses) and losses.sum() else None,
    }


def _z_for_alpha(alpha: float) -> float:
    """把單尾顯著水準換成 z 值（只需要幾個常用點，避免引入 scipy）。"""
    table = [(0.10, 1.2816), (0.05, 1.6449), (0.025, 1.9600),
             (0.01, 2.3263), (0.005, 2.5758), (0.001, 3.0902)]
    return min(table, key=lambda kv: abs(kv[0] - alpha))[1]


def evaluate_thresholds(
    wf_df: pd.DataFrame,
    candidate_thresholds: List[int],
    cost: float = 0.0,
    holding_bars: int = 5,
    alpha: float = 0.05,
    multiple_testing_correction: bool = True,
) -> dict:
    """
    對一組候選門檻，分別統計多空兩邊「不重疊進場、扣成本後」的績效。

    multiple_testing_correction：掃描 N 個門檻挑歷史最好的那個，本身就會
    系統性高估被選中門檻的績效（Bailey & López de Prado 的 Deflated
    Sharpe Ratio 講的就是這件事）。這裡用最簡單也最保守的 Šidák 修正，
    把顯著水準除以試驗次數，讓信賴下界跟著變嚴。
    """
    out = {"bull": [], "bear": [], "base": {}}
    if wf_df.empty:
        return out

    n_trials = max(len(candidate_thresholds), 1)
    eff_alpha = (1 - (1 - alpha) ** (1 / n_trials)) if multiple_testing_correction else alpha

    wf = wf_df.reset_index(drop=True)
    # 有停損/目標的模擬交易時，用「實際報酬」與「實際持有長度」；
    # 沒有時退回固定持有 horizon 的舊行為（對照組與單元測試會走這條）。
    use_trades = "long_ret" in wf.columns

    def series_for(side: str):
        """回傳 (每根K棒的報酬, 每根K棒的持有長度, R倍數)。"""
        if use_trades:
            p = "long" if side == "bull" else "short"
            return (wf[f"{p}_ret"].to_numpy(),
                    wf[f"{p}_bars"].astype(int),
                    wf[f"{p}_r"].to_numpy())
        sign = 1.0 if side == "bull" else -1.0
        return (sign * wf["future_return"].to_numpy(),
                pd.Series(holding_bars, index=range(len(wf))),
                None)

    # 基準率：完全不看訊號，每一根K棒都進場的績效。
    # 沒有這個對照，多頭行情裡任何做多訊號的勝率看起來都很漂亮
    # （實測基準勝率就有 52.7~58.8%，訊號只高出約 3 個百分點）。
    always = pd.Series(True, index=range(len(wf)))
    out["base"] = {"alpha_used": eff_alpha, "n_trials": n_trials}
    for side, key in (("bull", "long"), ("bear", "short")):
        r, hold_s, rm = series_for(side)
        picks = select_non_overlapping(always, hold_s)
        out["base"][key] = summarize_trades(r[picks], cost, alpha)

    for th in candidate_thresholds:
        for side, col in (("bull", "bull_score"), ("bear", "bear_score")):
            r, hold_s, rm = series_for(side)
            mask = wf[col] >= th
            picks = select_non_overlapping(mask, hold_s)
            stats = summarize_trades(r[picks], cost, eff_alpha)
            stats["threshold"] = th
            stats["avg_bars_held"] = float(hold_s.iloc[picks].mean()) if picks else None
            stats["avg_r"] = float(rm[picks].mean()) if (rm is not None and picks) else None
            # 保留被選中的交易，供 run_symbol_backtest 算複利權益曲線。
            # 期望值是算術平均，回答不了「複利之後贏不贏買進持有」。
            stats["_trade_returns"] = (r[picks] - cost).tolist()
            stats["_trade_bars"] = hold_s.iloc[picks].tolist()
            # 超額：扣掉「隨便進場」本來就有的漂移之後，這個訊號還剩多少
            base_exp = out["base"]["long" if side == "bull" else "short"]["expectancy"]
            stats["excess_expectancy"] = (
                stats["expectancy"] - base_exp
                if stats["expectancy"] is not None and base_exp is not None else None
            )
            out[side].append(stats)

    return out


def recommend_threshold(
    stats: List[dict],
    min_trades: int,
    default_threshold: int,
) -> dict:
    """
    在滿足『樣本數 >= min_trades』的門檻中，挑「扣成本期望值的信賴下界
    仍然為正」且訊號最多（門檻最寬鬆）的那個。

    判準從「勝率」換成「期望值下界 > 0」的理由：
      - 勝率完全沒有考慮賠率。實測 0050 勝率 59.1%（五檔最高）但每筆
        平均虧 0.75%，因為平均獲利只有平均虧損的 0.54 倍。用勝率挑門檻
        會很有信心地推薦一個穩定虧錢的設定。
      - 期望值直接回答「這樣做會不會賺錢」，而且已經內含成本。
      - 用下界而不是點估計，是為了同時擋掉小樣本雜訊與掃描多門檻的
        selection bias。

    挑不出來時不會硬湊：沒有任何門檻的期望值下界為正，就標記 no_edge，
    代表「這檔標的在這套訊號下，歷史上看不出優勢」——這是有用的資訊，
    不該被一個看起來還行的點估計蓋掉。
    """
    eligible = [s for s in stats if s["n"] >= min_trades and s.get("expectancy_lb") is not None]
    if not eligible:
        return {"threshold": default_threshold, "n": 0, "wins": 0,
                "win_rate": None, "win_rate_lb": None,
                "expectancy": None, "expectancy_lb": None,
                "payoff": None, "profit_factor": None, "excess_expectancy": None,
                "confidence": "insufficient_data"}

    positive = [s for s in eligible if s["expectancy_lb"] > 0]
    if positive:
        best = min(positive, key=lambda s: s["threshold"])
        return {**best, "confidence": "ok"}

    # 沒有任何門檻證明得了正期望值。回報下界最高的那個，但明確標記無優勢。
    best_fallback = max(eligible, key=lambda s: s["expectancy_lb"])
    return {**best_fallback, "confidence": "no_edge"}


def entry_config(config) -> dict:
    """從 config 取出 trade_model.plan_entry 需要的參數。"""
    return {
        "offset_atr": getattr(config, "ENTRY_LIMIT_OFFSET_ATR", 0.5),
        "valid_bars": getattr(config, "ENTRY_LIMIT_VALID_BARS", 1),
        "fallback": getattr(config, "ENTRY_FALLBACK", "market"),
    }


def trade_config(config) -> dict:
    """從 config 取出 trade_model.plan_trade 需要的參數。"""
    return {
        "stop_buffer_atr": getattr(config, "TRADE_STOP_BUFFER_ATR", 0.25),
        "atr_stop_mult": getattr(config, "TRADE_ATR_STOP_MULT", 1.5),
        "fallback_target_r": getattr(config, "TRADE_FALLBACK_TARGET_R", 2.0),
        "min_rr": getattr(config, "TRADE_MIN_RR", 0.5),
        "max_rr": getattr(config, "TRADE_MAX_RR", 10.0),
    }


def run_symbol_backtest(df: pd.DataFrame, analyzer: SMCAnalyzer, ind_df: pd.DataFrame,
                        config, cost: float = 0.0) -> dict:
    """整合以上步驟，回傳這檔標的的完整回測結果與建議門檻（多空分開）。"""
    wf_df = walk_forward_scores(
        df, analyzer, ind_df,
        horizon=config.BACKTEST_HORIZON_BARS,
        warmup=config.BACKTEST_WARMUP_BARS,
        ml_enabled=config.ML_ENABLED,
        ml_retrain_every=config.BACKTEST_ML_RETRAIN_EVERY,
        ml_min_train_rows=config.ML_MIN_TRAIN_ROWS,
        max_holding_bars=(
            getattr(config, "TRADE_TRAILING_MAX_HOLDING_BARS", 250)
            if getattr(config, "TRADE_EXIT_MODE", "fixed") == "trailing"
            else getattr(config, "TRADE_MAX_HOLDING_BARS", 20)
        ),
        adx_trend_threshold=config.ADX_TREND_THRESHOLD,
        recent_bars=config.RECENT_BARS_FOR_SCORE,
        atr_series=ind_df["atr"] if "atr" in ind_df.columns else None,
        trade_cfg=trade_config(config),
        exit_mode=getattr(config, "TRADE_EXIT_MODE", "fixed"),
        trail_atr_mult=getattr(config, "TRADE_TRAIL_ATR_MULT", 3.0),
        entry_cfg=entry_config(config),
    )

    stats = evaluate_thresholds(
        wf_df, config.BACKTEST_CANDIDATE_THRESHOLDS,
        cost=cost,
        holding_bars=config.BACKTEST_HORIZON_BARS,
        alpha=getattr(config, "BACKTEST_ALPHA", 0.05),
        multiple_testing_correction=getattr(config, "BACKTEST_MULTIPLE_TESTING_CORRECTION", True),
    )

    bull_rec = recommend_threshold(stats["bull"], config.BACKTEST_MIN_TRADES, config.ALERT_THRESHOLD)
    bear_rec = recommend_threshold(stats["bear"], config.BACKTEST_MIN_TRADES, config.ALERT_THRESHOLD)

    # ---- 複利績效與買進持有對照 ----
    # 每筆期望值是算術平均，回答不了「這樣做一年賺幾%」「贏不贏買進持有」。
    # 以獲利為目標時，這兩個問題才是真正的判準。
    bars_per_year = getattr(config, "BARS_PER_YEAR", 252)
    warm = min(config.BACKTEST_WARMUP_BARS, max(len(df) - 1, 0))
    bh = performance.buy_and_hold(df, start_idx=warm)
    for rec in (bull_rec, bear_rec):
        rets = rec.pop("_trade_returns", [])
        bars = rec.pop("_trade_bars", [])
        rec["performance"] = performance.summarize_performance(
            rets, bars, total_bars=max(len(wf_df), 1), bars_per_year=bars_per_year,
            # 只有做多才拿買進持有當對照；做空沒有對應的「什麼都不做」基準
            buy_hold_return=bh["return"] if rec is bull_rec else None,
            buy_hold_max_drawdown=bh["max_drawdown"] if rec is bull_rec else None,
        )

    # 掃描結果不需要帶著每筆明細，清掉以免 JSON 爆掉
    for side in ("bull", "bear"):
        for s in stats[side]:
            s.pop("_trade_returns", None)
            s.pop("_trade_bars", None)

    return {
        "n_evaluated_bars": len(wf_df),
        "cost": cost,
        "bull": bull_rec,
        "bear": bear_rec,
        "base": stats["base"],
        "buy_hold": bh,
        "threshold_sweep": {"bull": stats["bull"], "bear": stats["bear"]},
    }
