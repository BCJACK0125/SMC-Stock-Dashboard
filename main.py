# -*- coding: utf-8 -*-
"""
main.py
========
每日排程執行的主流程：
    1. 用 yfinance 下載清單內每檔股票的歷史 K 線
    2. 丟進 SMCAnalyzer 計算 BOS/CHoCH/OB/FVG/流動性/溢價折價區
    3. 計算綜合勝率分數
    4. 畫成 Plotly 圖表，組合成 index.html
    5. 若有標的分數超過門檻，寄送 Email 通知

本機測試：
    python main.py --dry-run     # 不寄信，只產生 index.html，方便本地檢查

正式在 GitHub Actions 執行時不加參數，會依 config.py 的 ALERT_THRESHOLD 判斷是否寄信。
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

import backtest_cache
import concentration
import position_sizing
import trade_model
from smc.analyzer import SMCAnalyzer
from plot_report import build_chart_html, build_index_html
from notifier import score, send_alert_email, get_email_credentials_from_env
import indicators as ind
import ml_model
import backtest
import config


TAIPEI_TZ = timezone(timedelta(hours=8))


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """把細週期 OHLCV 資料合併成粗週期（例如 60m -> 4h）。"""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out = df.resample(rule, label="left", closed="left").agg(agg)
    # resample 會在沒有交易的區塊(收盤時段/週末)產生全 NaN 的列，直接丟掉
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out


def fetch_data(symbol: str, retries: int = 3, backoff: float = 2.0):
    """
    下載歷史K線。

    yfinance 偶發性失敗（限流、暫時性 5xx、回傳空表）相當常見，而排程
    一天只跑一次——失敗就等於這檔標的當天整個消失。所以這裡做指數退避
    重試，而不是一次失敗就放棄。
    """
    # RAW_INTERVAL 是目前 config.py 實際使用的名稱；DATA_INTERVAL 是舊名，
    # 留作相容用的 fallback（config.py 裡已經沒有它，所以必須用 getattr 取，
    # 直接寫 config.DATA_INTERVAL 會在 RAW_INTERVAL 被移除時炸 AttributeError）。
    interval = (getattr(config, "RAW_INTERVAL", None)
                or getattr(config, "DATA_INTERVAL", None)
                or "1d")

    df, last_err = None, None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                symbol, period=config.DATA_PERIOD, interval=interval,
                progress=False, auto_adjust=True,
            )
            if not df.empty:
                break
            last_err = "回傳空資料"
        except Exception as e:              # 網路/解析錯誤都視為可重試
            last_err = f"{type(e).__name__}: {e}"
        if attempt < retries:
            wait = backoff ** (attempt - 1)
            print(f"[WARN] {symbol} 第 {attempt} 次下載失敗（{last_err}），"
                  f"{wait:.0f} 秒後重試", file=sys.stderr)
            time.sleep(wait)

    if df is None or df.empty:
        raise RuntimeError(f"{symbol} 重試 {retries} 次後仍抓不到資料"
                           f"（最後錯誤：{last_err}；可能代號錯誤或 yfinance 異常）")
    # yfinance 新版有時會回傳 MultiIndex 欄位，這裡統一攤平
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    resample_rule = getattr(config, "RESAMPLE_RULE", None)
    if resample_rule:
        df = resample_ohlcv(df, resample_rule)
        if df.empty:
            raise RuntimeError(f"{symbol} resample 成 {resample_rule} 後沒有資料")

    return df


def analyze_symbol(symbol: str) -> SMCAnalyzer:
    df = fetch_data(symbol)
    analyzer = SMCAnalyzer(
        df,
        swing_lookback=config.SWING_LOOKBACK,
        eq_tolerance_pct=config.EQ_TOLERANCE_PCT,
        fvg_min_gap_atr=getattr(config, "FVG_MIN_GAP_ATR", 0.0),
    )
    analyzer.run_all()
    return analyzer


# ---------------------------------------------------------------------------
# 警報去重
#
# SMC 訊號一旦成立，通常會連續好幾根K棒都維持成立（OB 還在、結構事件還在
# 最近 N 根內）。排程每天跑一次的話，同一個訊號會天天寄一封一模一樣的信，
# 幾天之後就會被當成雜訊直接忽略——這比不寄還糟。
#
# 做法：把「標的 + 方向」記在一個小 JSON 裡，冷卻期內不重複通知。
# 分數大幅變化（跨越 ALERT_RENOTIFY_SCORE_JUMP）時視為新訊號，會重新通知。
# ---------------------------------------------------------------------------
def load_alert_state() -> dict:
    path = getattr(config, "ALERT_STATE_JSON", "alert_state.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] 讀取 {path} 失敗，視為沒有歷史紀錄：{e}", file=sys.stderr)
        return {}


def save_alert_state(state: dict) -> None:
    path = getattr(config, "ALERT_STATE_JSON", "alert_state.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] 寫入 {path} 失敗（不影響本次寄信）：{e}", file=sys.stderr)


def filter_new_alerts(alerts: list, now: Optional[datetime] = None):
    """回傳 (需要寄出的訊號, 更新後的狀態)。"""
    now = now or datetime.now(TAIPEI_TZ)
    cooldown_days = getattr(config, "ALERT_COOLDOWN_DAYS", 5)
    jump = getattr(config, "ALERT_RENOTIFY_SCORE_JUMP", 15)

    state = load_alert_state()
    fresh = []
    for a in alerts:
        key = f"{a['symbol']}|{a['side']}"
        prev = state.get(key)
        if prev:
            try:
                last_at = datetime.fromisoformat(prev["at"])
                age_days = (now - last_at).total_seconds() / 86400
            except Exception:
                age_days = 1e9                      # 紀錄壞掉就當作很久以前
            score_jumped = abs(a["score"] - prev.get("score", 0)) >= jump
            if age_days < cooldown_days and not score_jumped:
                continue                            # 冷卻中且分數沒大變 -> 不重寄
        fresh.append(a)
        state[key] = {"at": now.isoformat(), "score": a["score"],
                      "threshold": a.get("threshold_used")}
    return fresh, state


def run(dry_run: bool = False) -> None:
    results = []
    alerts = []
    backtest_all_results = {}
    close_series = {}          # 供集中度分析用
    bt_cache = backtest_cache.load(getattr(config, "BACKTEST_CACHE_JSON", "backtest_cache.json"))
    n_cached = n_computed = 0
    generated_at = datetime.now(TAIPEI_TZ).strftime("%Y-%m-%d %H:%M (台北時間)")

    for item in config.WATCHLIST:
        symbol, name = item["symbol"], item["name"]
        print(f"[INFO] 分析中：{symbol} {name} ...")
        try:
            analyzer = analyze_symbol(symbol)
        except Exception as e:
            print(f"[WARN] {symbol} 分析失敗，跳過：{e}", file=sys.stderr)
            continue

        # 技術指標（RSI/MACD/EMA/ADX/ATR/OBV）
        ind_df = ind.compute_indicator_set(analyzer.df)

        # 輕量 ML 模型（資料量不足或類別失衡時，內部會回傳 None，score() 會自動略過這一項）
        ml_result = None
        if getattr(config, "ML_ENABLED", True):
            try:
                ml_result = ml_model.predict_next_move_probability(
                    analyzer.df, ind_df,
                    horizon=config.ML_HORIZON_BARS,
                    min_train_rows=config.ML_MIN_TRAIN_ROWS,
                    min_auc=getattr(config, "ML_MIN_AUC", 0.55),
                )
                if ml_result is None:
                    print(f"[INFO] {symbol} ML 樣本外預測力不足，本次不計 ML 分數。")
            except Exception as e:
                print(f"[WARN] {symbol} ML 模型訓練失敗，跳過該項評分：{e}", file=sys.stderr)

        s = score(
            analyzer,
            recent_bars=config.RECENT_BARS_FOR_SCORE,
            ind_df=ind_df,
            ml_result=ml_result,
            adx_trend_threshold=config.ADX_TREND_THRESHOLD,
        )
        last_ev = analyzer.last_structure_event()
        last_event_str = f"{last_ev.type}({'多' if last_ev.side=='bullish' else '空'})" if last_ev else "-"

        # -------------------- Walk-forward 回測：算這檔標的的建議門檻 --------------------
        cost = config.transaction_cost(symbol, item.get("etf", False))
        try:
            # 完整 walk-forward 很重但結果每天幾乎不變，所以快取；
            # 設定一改，指紋就變，快取自動失效。
            bt, hit = backtest_cache.get_or_compute(
                bt_cache, symbol, config,
                lambda: backtest.run_symbol_backtest(
                    analyzer.df, analyzer, ind_df, config, cost=cost),
                max_age_days=getattr(config, "BACKTEST_CACHE_DAYS", 7))
            if hit:
                bt = backtest_cache.restore_timestamps(bt)
                n_cached += 1
            else:
                n_computed += 1
        except Exception as e:
            print(f"[WARN] {symbol} 回測失敗，改用預設門檻：{e}", file=sys.stderr)
            _empty = {"threshold": config.ALERT_THRESHOLD, "n": 0, "wins": 0,
                      "win_rate": None, "win_rate_lb": None,
                      "expectancy": None, "expectancy_lb": None,
                      "payoff": None, "profit_factor": None, "excess_expectancy": None,
                      "confidence": "insufficient_data"}
            bt = {"n_evaluated_bars": 0, "cost": cost,
                  "bull": dict(_empty), "bear": dict(_empty),
                  "base": {}, "threshold_sweep": {"bull": [], "bear": []}}
        bull_threshold = bt["bull"]["threshold"]
        bear_threshold = bt["bear"]["threshold"]
        backtest_all_results[symbol] = bt

        # 把回測選中的歷史交易畫進圖裡，方便用肉眼驗證訊號品質
        chart_html = build_chart_html(symbol, analyzer, ind_df=ind_df,
                                      trades=bt.get("trades"))
        closes = analyzer.df["Close"]
        close_series[symbol] = closes
        prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else None

        # 進場計畫：訊號是收盤後才看到的，所以給的是「隔天怎麼下單」
        atr_now = float(ind_df["atr"].iloc[-1]) if "atr" in ind_df.columns else 0.0
        entry_plan = trade_model.plan_entry(
            float(closes.iloc[-1]), atr_now,
            offset_atr=getattr(config, "ENTRY_LIMIT_OFFSET_ATR", 0.5),
            valid_bars=getattr(config, "ENTRY_LIMIT_VALID_BARS", 1),
            fallback=getattr(config, "ENTRY_FALLBACK", "market"),
        )
        trade_plan = trade_model.plan_trade(
            analyzer, "bullish", float(closes.iloc[-1]), atr_now,
            **backtest.trade_config(config)) if atr_now > 0 else None

        # 部位建議：用隨機進場的三重障礙統計，量「這檔對這套出場結構的
        # 適配度」。不依賴訊號有預測力——訊號超額不顯著，所以不採計。
        sizing = None
        if getattr(config, "SIZING_ENABLED", True) and trade_plan is not None:
            try:
                risks = [t.get("risk_pct") for t in (bt.get("trades") or [])
                         if t.get("risk_pct")]
                if not risks:
                    risks = [abs(trade_plan.entry - trade_plan.stop) / trade_plan.entry]
                st = position_sizing.random_entry_barrier_stats(
                    analyzer.df, risks,
                    rr=config.SIZING_RR,
                    max_bars=getattr(config, "TRADE_TRAILING_MAX_HOLDING_BARS", 250),
                    warmup=config.BACKTEST_WARMUP_BARS,
                    n_samples=config.SIZING_SAMPLES)
                sizing = position_sizing.suggest_position(
                    st["wins"], st["n_resolved"], rr=config.SIZING_RR,
                    kelly_divisor=config.SIZING_KELLY_DIVISOR,
                    max_fraction=config.SIZING_MAX_FRACTION,
                    min_samples=config.SIZING_MIN_SAMPLES)
            except Exception as e:
                print(f"[WARN] {symbol} 部位建議計算失敗（不影響其他輸出）：{e}",
                      file=sys.stderr)

        results.append({
            "symbol": symbol,
            "name": name,
            "chart_html": chart_html,
            "bull_score": s["bull_score"],
            "bear_score": s["bear_score"],
            "zone": analyzer.current_zone["zone"] if analyzer.current_zone else "-",
            "last_close": float(closes.iloc[-1]),
            "prev_close": prev_close,
            "last_event": last_event_str,
            "ml_prob_up": ml_result["prob_up"] if ml_result else None,
            "alert": s["bull_score"] >= bull_threshold or s["bear_score"] >= bear_threshold,
            "generated_at": generated_at,
            "bull_threshold": bull_threshold,
            "bear_threshold": bear_threshold,
            "backtest": bt,
            "performance": bt["bull"].get("performance"),
            "entry_plan": entry_plan,
            "trade_plan": trade_plan,
            "sizing": sizing,
        })

        only_proven = getattr(config, "ALERT_ONLY_WHEN_EDGE_PROVEN", False)

        def worth_alerting(side_key: str) -> bool:
            return (not only_proven) or bt[side_key]["confidence"] == "ok"

        if s["bull_score"] >= bull_threshold and worth_alerting("bull"):
            alerts.append({
                "symbol": symbol, "name": name, "side": "bullish",
                "score": s["bull_score"], "reasons": s["reasons_bull"],
                "last_close": float(analyzer.df["Close"].iloc[-1]),
                "threshold_used": bull_threshold,
                "backtest_win_rate": bt["bull"]["win_rate"],
                "backtest_win_rate_lb": bt["bull"].get("win_rate_lb"),
                "backtest_expectancy": bt["bull"].get("expectancy"),
                "backtest_expectancy_lb": bt["bull"].get("expectancy_lb"),
                "backtest_payoff": bt["bull"].get("payoff"),
                "backtest_n": bt["bull"]["n"],
                "backtest_confidence": bt["bull"]["confidence"],
                "performance": bt["bull"].get("performance"),
                "entry_plan": entry_plan,
                "trade_plan": trade_plan,
                "sizing": sizing,
            })
        if s["bear_score"] >= bear_threshold and worth_alerting("bear"):
            alerts.append({
                "symbol": symbol, "name": name, "side": "bearish",
                "score": s["bear_score"], "reasons": s["reasons_bear"],
                "last_close": float(analyzer.df["Close"].iloc[-1]),
                "threshold_used": bear_threshold,
                "backtest_win_rate": bt["bear"]["win_rate"],
                "backtest_win_rate_lb": bt["bear"].get("win_rate_lb"),
                "backtest_expectancy": bt["bear"].get("expectancy"),
                "backtest_expectancy_lb": bt["bear"].get("expectancy_lb"),
                "backtest_payoff": bt["bear"].get("payoff"),
                "backtest_n": bt["bear"]["n"],
                "backtest_confidence": bt["bear"]["confidence"],
                "performance": bt["bear"].get("performance"),
            })

    if not results:
        print("[ERROR] 沒有任何標的分析成功，中止。", file=sys.stderr)
        sys.exit(1)

    # 集中度：同時出現的訊號是否其實是同一個風險。
    # 使用者不做資金配置，但同一天 5 個高度相關的訊號，等於在同一個賭注
    # 上押 5 倍——看起來分散，實際不是。
    concentration_note = []
    if len(alerts) >= 2:
        try:
            corr = concentration.correlation_matrix(
                close_series, lookback=getattr(config, "CORRELATION_LOOKBACK", 252))
            assessment = concentration.assess_signals(
                [a["symbol"] for a in alerts], corr,
                high_corr_threshold=getattr(config, "HIGH_CORRELATION_THRESHOLD", 0.6))
            concentration_note = concentration.describe(assessment)
            for line in concentration_note:
                print(f"[INFO] {line}")
        except Exception as e:
            print(f"[WARN] 集中度分析失敗（不影響寄信）：{e}", file=sys.stderr)

    if backtest_cache.save(getattr(config, "BACKTEST_CACHE_JSON", "backtest_cache.json"), bt_cache):
        print(f"[INFO] 回測快取：沿用 {n_cached} 檔、重算 {n_computed} 檔")

    build_index_html(results, output_path=config.OUTPUT_HTML,
                     concentration_note=concentration_note)
    print(f"[INFO] 已產生 {config.OUTPUT_HTML}")

    try:
        with open(config.BACKTEST_RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(
                {sym: {
                    "n_evaluated_bars": bt["n_evaluated_bars"],
                    "cost": bt.get("cost"),
                    "bull": bt["bull"], "bear": bt["bear"],
                    "base": bt.get("base", {}),
                } for sym, bt in backtest_all_results.items()},
                f, ensure_ascii=False, indent=2,
            )
        print(f"[INFO] 已產生 {config.BACKTEST_RESULTS_JSON}")
    except Exception as e:
        print(f"[WARN] 回測結果 JSON 寫入失敗（不影響網頁與寄信）：{e}", file=sys.stderr)

    # 去重：訊號通常會連續好幾根K棒維持成立，排程每天跑一次就會天天
    # 重複寄同一封信。這裡記住「已經通知過的訊號」，冷卻期內不再重寄。
    fresh_alerts, state = filter_new_alerts(alerts)
    suppressed = len(alerts) - len(fresh_alerts)
    if suppressed:
        print(f"[INFO] {suppressed} 筆訊號在冷卻期內（先前已通知過），本次不重複寄信。")

    if fresh_alerts and not dry_run:
        creds = get_email_credentials_from_env()
        if not all(creds.values()):
            print("[WARN] 缺少 Email 環境變數（GMAIL_SENDER / GMAIL_APP_PASSWORD / ALERT_RECIPIENT），略過寄信。",
                  file=sys.stderr)
        else:
            send_alert_email(fresh_alerts, **creds,
                             concentration_note=concentration_note)
            save_alert_state(state)      # 寄成功才記錄，失敗時下次仍會重試
            print(f"[INFO] 已寄出警報信，共 {len(fresh_alerts)} 筆訊號。")
    elif fresh_alerts and dry_run:
        print(f"[DRY-RUN] 有 {len(fresh_alerts)} 筆新訊號會觸發寄信，但因 --dry-run 略過。")
    elif alerts:
        print("[INFO] 有訊號達到門檻，但都在冷卻期內，不重複寄信。")
    else:
        print("[INFO] 本次無標的達到警報門檻，不寄信。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只產生網頁，不寄送 Email")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
