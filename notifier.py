# -*- coding: utf-8 -*-
"""
notifier.py
============
1. score() : 依據多個 SMC 訊號，計算「多方 / 空方綜合分數」(0~100)。
2. send_alert_email() : 分數超過門檻時，用 Gmail SMTP 寄出通知信。

評分規則（滿分 100，可自行在 config.py 調整權重／門檻）：

    A. SMC 結構訊號（滿分 60）
        +15  目前位於折價區 (discount) / 溢價區 (premium)
        +15  存在未被緩解的 Order Block，且價格在最近 N 根K棒內回踩過
        +10  存在未被回補的 FVG，且價格在最近 N 根K棒內觸碰過
        +15 / +8  最新結構事件為 CHoCH（反轉，權重較高）/ BOS（延續）
        +7   最近出現流動性掃蕩（EQH/EQL 被穿透後收回）

    B. 技術指標共振（滿分 20，見 indicators.confluence_signal）
        RSI 相對50的位置、MACD柱狀圖方向、EMA20/50/200排列、
        ADX+DI（只有 ADX 過門檻代表「趨勢有效」時才計分），每項 5 分。

    C. 輕量 ML 模型機率（滿分 20，見 ml_model.predict_next_move_probability）
        用 GradientBoostingClassifier 預測未來 N 根K棒上漲機率，
        機率偏離 50% 越多、給分越高（線性映射，上限 20分）。

    多空分數各自獨立累加封頂 100；同時偏高代表訊號矛盾，不建議進場。
    B、C 兩項為「選配」：呼叫 score() 時若不提供 ind_df / ml_result，
    則只計算 A 項（等同於只用 SMC，分數上限會跟著變成 60）。
"""

from __future__ import annotations
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional

from smc.analyzer import SMCAnalyzer
import indicators as ind


def score(
    analyzer: SMCAnalyzer,
    recent_bars: int = 5,
    ind_df=None,
    ml_result: Optional[dict] = None,
    adx_trend_threshold: float = 20.0,
) -> Dict:
    """回傳 {'bull_score': int, 'bear_score': int, 'reasons_bull': [...], 'reasons_bear': [...]}"""
    df = analyzer.df
    recent_df = df.tail(recent_bars)
    bull_score, bear_score = 0, 0
    reasons_bull, reasons_bear = [], []

    # ------------------------------------------------------------ A. SMC --
    # 1) 溢價 / 折價區
    if analyzer.current_zone:
        if analyzer.current_zone["zone"] == "discount":
            bull_score += 15
            reasons_bull.append("價格位於折價區 (Discount Zone)")
        elif analyzer.current_zone["zone"] == "premium":
            bear_score += 15
            reasons_bear.append("價格位於溢價區 (Premium Zone)")

    # 2) 未緩解 OB 是否近期被回踩
    for ob in analyzer.active_order_blocks("bullish"):
        if ob.start_index in recent_df.index or (
            recent_df["Low"].min() <= ob.top and recent_df["High"].max() >= ob.bottom
        ):
            bull_score += 15
            reasons_bull.append(f"價格回踩看漲 Order Block（{ob.bottom:.2f}~{ob.top:.2f}）")
            break

    for ob in analyzer.active_order_blocks("bearish"):
        if recent_df["High"].max() >= ob.bottom and recent_df["Low"].min() <= ob.top:
            bear_score += 15
            reasons_bear.append(f"價格觸及看跌 Order Block（{ob.bottom:.2f}~{ob.top:.2f}）")
            break

    # 3) 未回補 FVG 是否近期被觸碰
    for fvg in analyzer.active_fvgs("bullish"):
        if recent_df["Low"].min() <= fvg.top and recent_df["High"].max() >= fvg.bottom:
            bull_score += 10
            reasons_bull.append(f"價格修補看漲 FVG（{fvg.bottom:.2f}~{fvg.top:.2f}）")
            break

    for fvg in analyzer.active_fvgs("bearish"):
        if recent_df["High"].max() >= fvg.bottom and recent_df["Low"].min() <= fvg.top:
            bear_score += 10
            reasons_bear.append(f"價格修補看跌 FVG（{fvg.bottom:.2f}~{fvg.top:.2f}）")
            break

    # 4) 最新結構事件（CHoCH 權重高於 BOS）
    last_ev = analyzer.last_structure_event()
    if last_ev and last_ev.index in recent_df.index:
        if last_ev.side == "bullish":
            pts = 15 if last_ev.type == "CHoCH" else 8
            bull_score += pts
            reasons_bull.append(f"近期出現看漲 {last_ev.type}")
        else:
            pts = 15 if last_ev.type == "CHoCH" else 8
            bear_score += pts
            reasons_bear.append(f"近期出現看跌 {last_ev.type}")

    # 5) 流動性掃蕩後反轉（簡化版：近期低點跌破 EQL 後，最新收盤又收回其上）
    last_close = float(df["Close"].iloc[-1])
    for pool in analyzer.liquidity_pools:
        if pool.kind == "EQL" and recent_df["Low"].min() < pool.price <= last_close:
            bull_score += 7
            reasons_bull.append("偵測到流動性掃蕩後收回（EQL Sweep）")
            break
    for pool in analyzer.liquidity_pools:
        if pool.kind == "EQH" and recent_df["High"].max() > pool.price >= last_close:
            bear_score += 7
            reasons_bear.append("偵測到流動性掃蕩後收回（EQH Sweep）")
            break

    # ---------------------------------------------------- B. 技術指標共振 --
    if ind_df is not None and len(ind_df) > 0:
        conf = ind.confluence_signal(ind_df, adx_trend_threshold=adx_trend_threshold)
        bull_score += conf["bull_count"] * 5
        bear_score += conf["bear_count"] * 5
        if conf["bull_count"]:
            reasons_bull.append(f"技術指標共振：{conf['bull_count']} 項偏多訊號一致")
        if conf["bear_count"]:
            reasons_bear.append(f"技術指標共振：{conf['bear_count']} 項偏空訊號一致")

    # ---------------------------------------------------- C. ML 模型機率 --
    if ml_result is not None:
        prob_up = ml_result["prob_up"]
        if prob_up > 0.5:
            pts = min(round((prob_up - 0.5) * 40), 20)
            if pts > 0:
                bull_score += pts
                reasons_bull.append(
                    f"輕量ML模型預測未來{ml_result['horizon']}根K棒上漲機率 {prob_up:.0%}"
                    f"（訓練樣本 {ml_result['trained_rows']} 筆）"
                )
        elif prob_up < 0.5:
            pts = min(round((0.5 - prob_up) * 40), 20)
            if pts > 0:
                bear_score += pts
                reasons_bear.append(
                    f"輕量ML模型預測未來{ml_result['horizon']}根K棒上漲機率僅 {prob_up:.0%}"
                    f"（訓練樣本 {ml_result['trained_rows']} 筆）"
                )

    return {
        "bull_score": min(bull_score, 100),
        "bear_score": min(bear_score, 100),
        "reasons_bull": reasons_bull,
        "reasons_bear": reasons_bear,
    }


def send_alert_email(alerts: list, sender: str, app_password: str, recipient: str) -> None:
    """
    alerts: [{"symbol":..., "name":..., "side": "bullish"/"bearish",
               "score":..., "reasons":[...], "last_close":...,
               "threshold_used":..., "backtest_win_rate":..., "backtest_n":...,
               "backtest_confidence":...}, ...]
    使用 Gmail SMTP (smtp.gmail.com:465, SSL)。
    寄件帳號需先開啟兩步驟驗證，並產生「應用程式密碼」(App Password) 供 app_password 使用，
    不要直接用登入密碼。
    """
    if not alerts:
        return

    subject = f"📈 SMC 選股警報：{len(alerts)} 檔標的觸發訊號"
    lines = []
    for a in alerts:
        side_label = "多方 🟢" if a["side"] == "bullish" else "空方 🔴"
        bt_wr = a.get("backtest_win_rate")
        bt_n = a.get("backtest_n")
        bt_conf = a.get("backtest_confidence")
        conf_label = {"ok": "", "low": "（歷史勝率未達標，僅供參考）",
                      "insufficient_data": "（回測樣本不足，門檻為預設值，僅供參考）"}.get(bt_conf, "")
        bt_line = (f"  ↳ 回測：門檻 {a.get('threshold_used')} 分過去出現 {bt_n} 次訊號，"
                   f"歷史勝率 {bt_wr:.0%}{conf_label}") if bt_wr is not None else \
                  f"  ↳ 回測：樣本不足，使用預設門檻 {a.get('threshold_used')} 分{conf_label}"
        lines.append(
            f"【{a['symbol']} {a['name']}】{side_label} 綜合分數 {a['score']}／收盤 {a['last_close']:.2f}\n"
            + bt_line + "\n"
            + "\n".join(f"  - {r}" for r in a["reasons"])
        )
    body = "\n\n".join(lines) + "\n\n（本郵件由 GitHub Actions 自動發送，僅供研究參考，非投資建議）"

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, app_password)
        server.sendmail(sender, recipient, msg.as_string())


def get_email_credentials_from_env():
    """從環境變數（GitHub Actions Secrets 注入）讀取寄信帳密。"""
    return {
        "sender": os.environ.get("GMAIL_SENDER", ""),
        "app_password": os.environ.get("GMAIL_APP_PASSWORD", ""),
        "recipient": os.environ.get("ALERT_RECIPIENT", ""),
    }
