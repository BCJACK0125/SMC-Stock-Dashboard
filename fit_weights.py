# -*- coding: utf-8 -*-
"""
fit_weights.py
===============
用真實績效擬合評分權重，取代手工配的 15/15/10/15/8/7/5/20。

為什麼值得試：
    那組權重從頭到尾沒有擬合過任何東西，純粹是憑感覺配的。既然現在有了
    正確的目標函數（扣成本後的超額期望值）與乾淨的元件向量，就可以讓
    資料自己決定每個元件值多少分。

為什麼預期不高：
    先前實測顯示，訊號相對基準的超額期望值本來就在雜訊範圍內反覆變號
    （−0.85% ~ +0.78%，隨參數選擇變號），ML 元件的樣本外 AUC 更只有
    0.518。如果元件裡根本沒有訊息，再怎麼調配權重也生不出來。

    所以這支程式的價值不只是「找到更好的權重」，更是**給出一個明確的
    答案**：如果嚴格樣本外的擬合權重也贏不過手工權重，那就證明問題出在
    特徵本身，不是權重配置——該換特徵，不是繼續調參。

方法（刻意設計成很難自欺）：
    1. 時間切分：前 60% 訓練、後 40% 測試，中間留一段 embargo 間隔，
       避免持有期橫跨切點造成資訊洩漏。
    2. 只在訓練段擬合權重，測試段完全不參與擬合。
    3. 對照組是手工權重在**同一個測試段**上的表現。
    4. 額外跑隨機權重當作對照下限——擬合權重若贏不過隨機，就是純雜訊。

用法：
    python fit_weights.py --cache <目錄>     # 用快取的 pickle 跑
    python fit_weights.py --limit 10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import backtest as B
import config
import indicators as ind
import notifier
import performance
import trade_model
from smc.analyzer import SMCAnalyzer


def build_component_matrix(
    df: pd.DataFrame, analyzer: SMCAnalyzer, ind_df: pd.DataFrame,
    warmup: int, horizon: int, atr_series: pd.Series, trade_cfg: dict,
    exit_mode: str, trail_mult: float, max_hold: int,
) -> pd.DataFrame:
    """
    逐根K棒抽出「元件向量 + 進場後的實際交易結果」。

    元件只用 t 當下已知的 AsOfView 算，出場才往後看——跟正式回測同一套
    邏輯，所以沒有前視偏差。
    """
    rows = []
    end = len(df) - horizon
    for t in range(max(warmup, 30), end):
        ts = df.index[t]
        view = B.AsOfView(analyzer, t, ts)
        last_close = float(df["Close"].iloc[t])
        B._finalize_zone(view, last_close)

        comp = notifier.score_components(
            view, recent_bars=config.RECENT_BARS_FOR_SCORE,
            ind_df=ind_df.iloc[: t + 1], ml_result=None,
            adx_trend_threshold=config.ADX_TREND_THRESHOLD,
        )

        atr_now = float(atr_series.iloc[t])
        plan = trade_model.plan_trade(view, "bullish", last_close, atr_now, **trade_cfg)
        if plan is None:
            continue
        future = df.iloc[t + 1:]
        if exit_mode == "trailing":
            res = trade_model.simulate_trailing_trade(
                future, plan, atr_now, trail_atr_mult=trail_mult,
                max_holding_bars=max_hold)
        else:
            res = trade_model.simulate_trade(future, plan, max_holding_bars=max_hold)

        row = {f"c_{k}": v for k, v in comp["bull"].items()}
        row.update({"ts": ts, "t": t, "ret": res.return_pct,
                    "bars": max(res.bars_held, 1)})
        rows.append(row)
    return pd.DataFrame(rows)


def fit_linear_weights(
    X: np.ndarray, y: np.ndarray, ridge: float = 1.0
) -> np.ndarray:
    """
    嶺回歸：用元件向量預測交易報酬，回歸係數就是權重。

    用嶺回歸而不是普通最小平方，是因為元件之間高度相關（例如 EMA 排列
    與 ADX 方向常常同時成立），OLS 在共線性下會給出很大且不穩定的係數。
    """
    Xb = np.column_stack([np.ones(len(X)), X])
    A = Xb.T @ Xb + ridge * np.eye(Xb.shape[1])
    A[0, 0] -= ridge                     # 截距不做正則化
    coef = np.linalg.solve(A, Xb.T @ y)
    return coef[1:]                      # 丟掉截距，只留元件權重


def weights_to_score_scale(raw: np.ndarray, names: List[str]) -> Dict[str, float]:
    """
    把回歸係數換算成跟原本同量級（總和約 100）的分數權重。

    負係數直接歸零：代表那個元件與報酬反向，留著只會製造反訊號；
    這裡的目的是「哪些元件該加分」，不是做多空雙向的線性模型。
    """
    w = np.clip(raw, 0, None)
    if w.sum() <= 0:
        return {n: 0.0 for n in names}
    w = w / w.sum() * 100.0
    return {n: float(v) for n, v in zip(names, w)}


def _apply(seg: pd.DataFrame, weights, comp_cols, cost, th, min_trades):
    """在某一段資料上，用固定門檻算績效。"""
    w = np.array([weights.get(c[2:], 0.0) for c in comp_cols])
    scores = seg[comp_cols].to_numpy() @ w
    mask = pd.Series(scores >= th, index=range(len(seg)))
    picks = B.select_non_overlapping(mask, seg["bars"].reset_index(drop=True))
    if len(picks) < min_trades:
        return None
    r = seg["ret"].to_numpy()[picks] - cost
    return {"threshold": th, "n": len(picks), "expectancy": float(r.mean()),
            "excess": float(r.mean() - (seg["ret"] - cost).mean()),
            "win_rate": float((r > 0).mean()),
            "total_return": float(np.prod(1 + r) - 1)}


def evaluate_weights(
    train: pd.DataFrame, test: pd.DataFrame, weights: Dict[str, float],
    comp_cols: List[str], cost: float, thresholds: List[int], min_trades: int,
) -> Optional[dict]:
    """
    **門檻在訓練段挑、在測試段套用**——測試段完全不參與任何選擇。

    第一版是在測試段上挑最佳門檻，那是選擇偏誤：會同時灌水所有比較組，
    讓三組看起來都「顯著優於基準」。改成訓練段挑門檻之後，測試段的數字
    才是真正沒有被任何後見之明污染的樣本外績效。
    """
    best_th, best_exp = None, None
    for th in thresholds:
        r = _apply(train, weights, comp_cols, cost, th, min_trades)
        if r is None:
            continue
        if best_exp is None or r["expectancy"] > best_exp:
            best_th, best_exp = th, r["expectancy"]
    if best_th is None:
        return None
    out = _apply(test, weights, comp_cols, cost, best_th,
                 max(min_trades // 2, 8))
    if out is not None:
        out["train_expectancy"] = best_exp
    return out


def run_symbol(path: str, args) -> Optional[dict]:
    sym = os.path.basename(path)[:-4]
    df = pd.read_pickle(path)
    is_etf = sym in {x["symbol"].replace(".", "_")
                     for x in config.VALIDATION_UNIVERSE if x.get("etf")}
    cost = config.transaction_cost(sym.replace("_", "."), is_etf)

    a = SMCAnalyzer(df, swing_lookback=config.SWING_LOOKBACK,
                    eq_tolerance_pct=config.EQ_TOLERANCE_PCT,
                    fvg_min_gap_atr=config.FVG_MIN_GAP_ATR).run_all()
    i_df = ind.compute_indicator_set(a.df)
    exit_mode = getattr(config, "TRADE_EXIT_MODE", "fixed")
    max_hold = (config.TRADE_TRAILING_MAX_HOLDING_BARS
                if exit_mode == "trailing" else config.TRADE_MAX_HOLDING_BARS)

    M = build_component_matrix(
        a.df, a, i_df, config.BACKTEST_WARMUP_BARS, config.BACKTEST_HORIZON_BARS,
        i_df["atr"], B.trade_config(config), exit_mode,
        getattr(config, "TRADE_TRAIL_ATR_MULT", 2.0), max_hold)
    if len(M) < 400:
        return None

    comp_cols = [c for c in M.columns if c.startswith("c_")]
    # 時間切分 + embargo：持有期最長 max_hold 根，切點前後各留這麼多根不用，
    # 否則訓練段最後幾筆交易的結果會延伸進測試段，造成資訊洩漏。
    split = int(len(M) * 0.6)
    embargo = max_hold
    train = M.iloc[:split]
    test = M.iloc[split + embargo:]
    if len(train) < 200 or len(test) < 200:
        return None

    fitted_raw = fit_linear_weights(
        train[comp_cols].to_numpy(), (train["ret"] - cost).to_numpy(), ridge=args.ridge)
    fitted = weights_to_score_scale(fitted_raw, [c[2:] for c in comp_cols])

    rng = np.random.default_rng(abs(hash(sym)) % 2**32)
    random_w = weights_to_score_scale(rng.random(len(comp_cols)), [c[2:] for c in comp_cols])

    th = config.BACKTEST_CANDIDATE_THRESHOLDS
    mt = max(config.BACKTEST_MIN_TRADES // 2, 15)   # 測試段較短，放寬樣本門檻
    return {
        "symbol": sym, "n_bars": len(M), "n_train": len(train), "n_test": len(test),
        "fitted_weights": fitted,
        "manual": evaluate_weights(train, test, notifier.DEFAULT_WEIGHTS, comp_cols, cost, th, mt),
        "fitted": evaluate_weights(train, test, fitted, comp_cols, cost, th, mt),
        "random": evaluate_weights(train, test, random_w, comp_cols, cost, th, mt),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="放 *.pkl 日線資料的目錄")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--out", default="weight_fit_results.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache, "*.pkl")))[: args.limit]
    results = []
    for i, p in enumerate(files, 1):
        r = run_symbol(p, args)
        if r:
            results.append(r)
            print(f"[{i}/{len(files)}] {r['symbol']} "
                  f"train={r['n_train']} test={r['n_test']}", flush=True)
        else:
            print(f"[{i}/{len(files)}] {os.path.basename(p)} 資料不足，略過", flush=True)

    print("\n" + "=" * 78)
    print("樣本外比較：手工權重 vs 擬合權重 vs 隨機權重")
    print("=" * 78)
    print(f"{'標的':<10}{'手工超額':>12}{'擬合超額':>12}{'隨機超額':>12}  勝方")
    print("-" * 78)
    wins = {"manual": 0, "fitted": 0, "random": 0}
    rows = []
    for r in results:
        vals = {k: (r[k]["excess"] if r[k] else None) for k in ("manual", "fitted", "random")}
        if any(v is None for v in vals.values()):
            print(f"{r['symbol']:<10}  樣本不足")
            continue
        best = max(vals, key=lambda k: vals[k])
        wins[best] += 1
        rows.append(vals)
        print(f"{r['symbol']:<10}{vals['manual']:>11.2%}{vals['fitted']:>12.2%}"
              f"{vals['random']:>12.2%}  {best}")

    print("-" * 78)
    if rows:
        for k in ("manual", "fitted", "random"):
            arr = np.array([x[k] for x in rows])
            print(f"{k:<10} 超額中位數 {np.median(arr):+.3%}   "
                  f"為正的比例 {np.mean(arr > 0):.0%}   勝出 {wins[k]}/{len(rows)} 檔")

    print("\n怎麼讀這份報告：")
    print("  · 擬合權重若贏不過隨機權重，代表元件裡根本沒有可擬合的訊息。")
    print("  · 擬合贏過手工、但三者都接近零 → 只是在雜訊裡挑到比較好看的那個。")
    print("  · 門檻在訓練段挑、測試段套用，測試段不參與任何選擇——沒有後見之明。")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n[INFO] 已寫出 {args.out}")


if __name__ == "__main__":
    main()
