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
import sys
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf

from smc.analyzer import SMCAnalyzer
from plot_report import build_chart_html, build_index_html
from notifier import score, send_alert_email, get_email_credentials_from_env
import config


TAIPEI_TZ = timezone(timedelta(hours=8))


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """把細週期 OHLCV 資料合併成粗週期（例如 60m -> 4h）。"""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out = df.resample(rule, label="left", closed="left").agg(agg)
    # resample 會在沒有交易的區塊(收盤時段/週末)產生全 NaN 的列，直接丟掉
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out


def fetch_data(symbol: str):
    interval = getattr(config, "RAW_INTERVAL", None) or config.DATA_INTERVAL
    df = yf.download(
        symbol, period=config.DATA_PERIOD, interval=interval,
        progress=False, auto_adjust=True,
    )
    if df.empty:
        raise RuntimeError(f"{symbol} 抓不到資料（可能代號錯誤、yfinance 暫時異常，"
                            f"或此代號不支援 {interval} 這個 intraday 區間）")
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
    )
    analyzer.run_all()
    return analyzer


def run(dry_run: bool = False) -> None:
    results = []
    alerts = []
    generated_at = datetime.now(TAIPEI_TZ).strftime("%Y-%m-%d %H:%M (台北時間)")

    for item in config.WATCHLIST:
        symbol, name = item["symbol"], item["name"]
        print(f"[INFO] 分析中：{symbol} {name} ...")
        try:
            analyzer = analyze_symbol(symbol)
        except Exception as e:
            print(f"[WARN] {symbol} 分析失敗，跳過：{e}", file=sys.stderr)
            continue

        s = score(analyzer, recent_bars=config.RECENT_BARS_FOR_SCORE)
        last_ev = analyzer.last_structure_event()
        last_event_str = f"{last_ev.type}({'多' if last_ev.side=='bullish' else '空'})" if last_ev else "-"

        chart_html = build_chart_html(symbol, analyzer)

        results.append({
            "symbol": symbol,
            "name": name,
            "chart_html": chart_html,
            "bull_score": s["bull_score"],
            "bear_score": s["bear_score"],
            "zone": analyzer.current_zone["zone"] if analyzer.current_zone else "-",
            "last_close": float(analyzer.df["Close"].iloc[-1]),
            "last_event": last_event_str,
            "alert": max(s["bull_score"], s["bear_score"]) >= config.ALERT_THRESHOLD,
            "generated_at": generated_at,
        })

        if s["bull_score"] >= config.ALERT_THRESHOLD:
            alerts.append({
                "symbol": symbol, "name": name, "side": "bullish",
                "score": s["bull_score"], "reasons": s["reasons_bull"],
                "last_close": float(analyzer.df["Close"].iloc[-1]),
            })
        if s["bear_score"] >= config.ALERT_THRESHOLD:
            alerts.append({
                "symbol": symbol, "name": name, "side": "bearish",
                "score": s["bear_score"], "reasons": s["reasons_bear"],
                "last_close": float(analyzer.df["Close"].iloc[-1]),
            })

    if not results:
        print("[ERROR] 沒有任何標的分析成功，中止。", file=sys.stderr)
        sys.exit(1)

    build_index_html(results, output_path=config.OUTPUT_HTML)
    print(f"[INFO] 已產生 {config.OUTPUT_HTML}")

    if alerts and not dry_run:
        creds = get_email_credentials_from_env()
        if not all(creds.values()):
            print("[WARN] 缺少 Email 環境變數（GMAIL_SENDER / GMAIL_APP_PASSWORD / ALERT_RECIPIENT），略過寄信。",
                  file=sys.stderr)
        else:
            send_alert_email(alerts, **creds)
            print(f"[INFO] 已寄出警報信，共 {len(alerts)} 筆訊號。")
    elif alerts and dry_run:
        print(f"[DRY-RUN] 有 {len(alerts)} 筆訊號會觸發寄信，但因 --dry-run 略過。")
    else:
        print("[INFO] 本次無標的達到警報門檻，不寄信。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只產生網頁，不寄送 Email")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
