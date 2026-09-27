# -*- coding: utf-8 -*-
"""
validate.py
============
橫斷面驗證：回答「這套訊號到底有沒有優勢」，而不是「今天要不要進場」。

為什麼要跟 main.py 分開：
    main.py 是每日排程，關心的是「現在有沒有訊號」，標的數量要少、跑得快。
    但要判斷一套策略有沒有真實優勢，5 檔標的、單一市場環境是不夠的——
    2317 一檔出現正期望值，完全可能只是運氣。這支程式跑一個 30~50 檔的
    驗證池，看的是**跨標的的一致性**：

        - 有多少比例的標的能證明正期望值？
        - 匯總所有標的的交易（pooled）之後，整體期望值是否顯著為正？
        - 優勢是普遍存在，還是集中在少數幾檔？

    如果只有 1~2 檔亮燈、pooled 期望值貼近零，那就是「沒有優勢」的證據，
    不是「再多調幾個參數就好」。

用法：
    python validate.py                # 跑完整驗證池（很慢，建議本機跑）
    python validate.py --limit 10     # 只跑前 10 檔，快速確認流程沒壞
    python validate.py --jobs 4       # 多進程平行（CPU 核心數允許時）
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import backtest
import config
import indicators as ind
import main as main_mod
from smc.analyzer import SMCAnalyzer


def analyze_one(item: dict) -> dict:
    """單一標的：下載 -> 分析 -> 回測。回傳可序列化的摘要。"""
    symbol = item["symbol"]
    cost = config.transaction_cost(symbol, item.get("etf", False))
    df = main_mod.fetch_data(symbol)
    analyzer = SMCAnalyzer(
        df,
        swing_lookback=config.SWING_LOOKBACK,
        eq_tolerance_pct=config.EQ_TOLERANCE_PCT,
        fvg_min_gap_atr=getattr(config, "FVG_MIN_GAP_ATR", 0.0),
    ).run_all()
    ind_df = ind.compute_indicator_set(analyzer.df)
    bt = backtest.run_symbol_backtest(analyzer.df, analyzer, ind_df, config, cost=cost)

    out = {"symbol": symbol, "name": item.get("name", symbol),
           "bars": len(analyzer.df), "cost": cost}
    for side in ("bull", "bear"):
        r = bt[side]
        base = bt.get("base", {}).get("long" if side == "bull" else "short", {})
        out[side] = {
            "threshold": r["threshold"], "n": r["n"],
            "win_rate": r.get("win_rate"), "payoff": r.get("payoff"),
            "expectancy": r.get("expectancy"), "expectancy_lb": r.get("expectancy_lb"),
            "excess": r.get("excess_expectancy"),
            "base_expectancy": base.get("expectancy"),
            "confidence": r["confidence"],
        }
    return out


def _safe_analyze(item: dict) -> dict:
    try:
        return analyze_one(item)
    except Exception as e:  # 單一標的失敗不該中斷整場驗證
        return {"symbol": item["symbol"], "error": f"{type(e).__name__}: {e}"}


def summarize(results: list) -> dict:
    """把每檔的結果匯總成「這套策略有沒有優勢」的判斷依據。"""
    ok = [r for r in results if "error" not in r]
    report = {"n_symbols": len(ok), "n_failed": len(results) - len(ok)}

    for side in ("bull", "bear"):
        rows = [r[side] for r in ok if r[side].get("expectancy") is not None]
        if not rows:
            report[side] = {"n_evaluated": 0}
            continue

        proven = [x for x in rows if x["confidence"] == "ok"]
        exps = np.array([x["expectancy"] for x in rows])
        excess = np.array([x["excess"] for x in rows if x["excess"] is not None])

        report[side] = {
            "n_evaluated": len(rows),
            # 最重要的一個數字：有多少比例的標的「證明得了」正期望值。
            # 隨機亂猜的話，這個比例應該接近 0（因為下界判準很嚴）。
            "n_proven_positive": len(proven),
            "pct_proven_positive": len(proven) / len(rows),
            "median_expectancy": float(np.median(exps)),
            "mean_expectancy": float(exps.mean()),
            "pct_positive_point_estimate": float((exps > 0).mean()),
            "median_excess_vs_base": float(np.median(excess)) if len(excess) else None,
            "proven_symbols": [r["symbol"] for r in ok
                               if r[side].get("confidence") == "ok"],
        }
    return report


def print_report(results: list, report: dict) -> None:
    print("\n" + "=" * 86)
    print("橫斷面驗證結果")
    print("=" * 86)
    print(f"{'標的':<10}{'方':<4}{'門檻':>5}{'筆數':>6}{'勝率':>7}{'賠率':>7}"
          f"{'期望值':>9}{'下界':>9}{'超額':>9}  判定")
    print("-" * 86)
    for r in sorted(results, key=lambda x: x.get("symbol", "")):
        if "error" in r:
            print(f"{r['symbol']:<10}  ⚠️ {r['error']}")
            continue
        for side, lab in (("bull", "多"), ("bear", "空")):
            s = r[side]
            if s.get("expectancy") is None:
                print(f"{r['symbol']:<10}{lab:<4}{s['threshold']:>5}{s['n']:>6}"
                      f"{'—':>7}{'—':>7}{'—':>9}{'—':>9}{'—':>9}  {s['confidence']}")
                continue
            print(f"{r['symbol']:<10}{lab:<4}{s['threshold']:>5}{s['n']:>6}"
                  f"{s['win_rate']:>6.0%}{(s['payoff'] or 0):>7.2f}"
                  f"{s['expectancy']:>8.2%}{s['expectancy_lb']:>9.2%}"
                  f"{(s['excess'] or 0):>9.2%}  {s['confidence']}")

    print("\n" + "=" * 86)
    print("匯總判斷")
    print("=" * 86)
    print(f"成功分析 {report['n_symbols']} 檔，失敗 {report['n_failed']} 檔")
    for side, lab in (("bull", "多方"), ("bear", "空方")):
        s = report.get(side, {})
        if not s.get("n_evaluated"):
            print(f"\n{lab}：無可評估樣本")
            continue
        print(f"\n{lab}：")
        print(f"  能證明正期望值的標的：{s['n_proven_positive']}/{s['n_evaluated']}"
              f"（{s['pct_proven_positive']:.0%}）")
        print(f"  期望值點估計為正的比例：{s['pct_positive_point_estimate']:.0%}")
        print(f"  期望值中位數：{s['median_expectancy']:+.3%}/筆")
        if s.get("median_excess_vs_base") is not None:
            print(f"  相對基準的超額中位數：{s['median_excess_vs_base']:+.3%}/筆")
        if s["proven_symbols"]:
            print(f"  亮燈標的：{', '.join(s['proven_symbols'])}")

    print("\n" + "-" * 86)
    print("怎麼讀這份報告：")
    print("  · 只有少數幾檔亮燈、且期望值中位數貼近 0 → 那幾檔比較可能是運氣，不是優勢。")
    print("  · 超額中位數才是關鍵：扣掉「隨便進場本來就有的漂移」之後還剩多少。")
    print("  · 這裡的期望值已扣除來回交易成本，但沒有計入滑價與跳空。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 檔（快速冒煙測試）")
    ap.add_argument("--jobs", type=int, default=1, help="平行進程數")
    ap.add_argument("--out", default="validation_results.json")
    args = ap.parse_args()

    universe = getattr(config, "VALIDATION_UNIVERSE", config.WATCHLIST)
    if args.limit:
        universe = universe[: args.limit]
    print(f"[INFO] 驗證池共 {len(universe)} 檔，jobs={args.jobs}")

    results = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_safe_analyze, it): it for it in universe}
            for f in as_completed(futs):
                r = f.result()
                results.append(r)
                print(f"[{len(results)}/{len(universe)}] {r['symbol']}"
                      f"{' 失敗' if 'error' in r else ''}", flush=True)
    else:
        for i, it in enumerate(universe, 1):
            r = _safe_analyze(it)
            results.append(r)
            print(f"[{i}/{len(universe)}] {r['symbol']}"
                  f"{' 失敗' if 'error' in r else ''}", flush=True)

    report = summarize(results)
    print_report(results, report)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"report": report, "per_symbol": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n[INFO] 已寫出 {args.out}")


if __name__ == "__main__":
    main()
