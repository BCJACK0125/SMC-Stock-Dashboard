# -*- coding: utf-8 -*-
"""
notifier.py
============
1. score() : 依據多個 SMC 訊號，計算「多方 / 空方綜合分數」(0~100)。
2. send_alert_email() : 分數超過門檻時，用 Gmail SMTP 寄出通知信。

評分規則（可自行調整權重）：
    多方分數（bull_score）加分項：
        +25  目前位於折價區 (discount)
        +25  存在未被緩解的看漲 Order Block，且價格在最近 N 根K棒內回踩過
        +20  存在未被回補的看漲 FVG，且價格在最近 N 根K棒內觸碰過
        +20  最新結構事件為看漲 CHoCH（早期反轉訊號，權重較高）
        +10  最新結構事件為看漲 BOS（趨勢延續）
        +10  最近出現流動性掃蕩（EQL 被跌破後又收回，屬於獵取流動性後反轉）

    空方分數（bear_score）為上述鏡像規則。

    兩者獨立計算，可能同時偏高（代表訊號矛盾，不建議進場）。
"""

from __future__ import annotations
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict

from smc.analyzer import SMCAnalyzer


def score(analyzer: SMCAnalyzer, recent_bars: int = 5) -> Dict:
    """回傳 {'bull_score': int, 'bear_score': int, 'reasons_bull': [...], 'reasons_bear': [...]}"""
    df = analyzer.df
    recent_df = df.tail(recent_bars)
    bull_score, bear_score = 0, 0
    reasons_bull, reasons_bear = [], []

    # 1) 溢價 / 折價區
    if analyzer.current_zone:
        if analyzer.current_zone["zone"] == "discount":
            bull_score += 25
            reasons_bull.append("價格位於折價區 (Discount Zone)")
        elif analyzer.current_zone["zone"] == "premium":
            bear_score += 25
            reasons_bear.append("價格位於溢價區 (Premium Zone)")

    # 2) 未緩解 OB 是否近期被回踩
    for ob in analyzer.active_order_blocks("bullish"):
        if ob.start_index in recent_df.index or (
            recent_df["Low"].min() <= ob.top and recent_df["High"].max() >= ob.bottom
        ):
            bull_score += 25
            reasons_bull.append(f"價格回踩看漲 Order Block（{ob.bottom:.2f}~{ob.top:.2f}）")
            break

    for ob in analyzer.active_order_blocks("bearish"):
        if recent_df["High"].max() >= ob.bottom and recent_df["Low"].min() <= ob.top:
            bear_score += 25
            reasons_bear.append(f"價格觸及看跌 Order Block（{ob.bottom:.2f}~{ob.top:.2f}）")
            break

    # 3) 未回補 FVG 是否近期被觸碰
    for fvg in analyzer.active_fvgs("bullish"):
        if recent_df["Low"].min() <= fvg.top and recent_df["High"].max() >= fvg.bottom:
            bull_score += 20
            reasons_bull.append(f"價格修補看漲 FVG（{fvg.bottom:.2f}~{fvg.top:.2f}）")
            break

    for fvg in analyzer.active_fvgs("bearish"):
        if recent_df["High"].max() >= fvg.bottom and recent_df["Low"].min() <= fvg.top:
            bear_score += 20
            reasons_bear.append(f"價格修補看跌 FVG（{fvg.bottom:.2f}~{fvg.top:.2f}）")
            break

    # 4) 最新結構事件（CHoCH 權重高於 BOS）
    last_ev = analyzer.last_structure_event()
    if last_ev and last_ev.index in recent_df.index:
        if last_ev.side == "bullish":
            pts = 20 if last_ev.type == "CHoCH" else 10
            bull_score += pts
            reasons_bull.append(f"近期出現看漲 {last_ev.type}")
        else:
            pts = 20 if last_ev.type == "CHoCH" else 10
            bear_score += pts
            reasons_bear.append(f"近期出現看跌 {last_ev.type}")

    # 5) 流動性掃蕩後反轉（簡化版：近期低點跌破 EQL 後，最新收盤又收回其上）
    last_close = float(df["Close"].iloc[-1])
    for pool in analyzer.liquidity_pools:
        if pool.kind == "EQL" and recent_df["Low"].min() < pool.price <= last_close:
            bull_score += 10
            reasons_bull.append("偵測到流動性掃蕩後收回（EQL Sweep）")
            break
    for pool in analyzer.liquidity_pools:
        if pool.kind == "EQH" and recent_df["High"].max() > pool.price >= last_close:
            bear_score += 10
            reasons_bear.append("偵測到流動性掃蕩後收回（EQH Sweep）")
            break

    return {
        "bull_score": min(bull_score, 100),
        "bear_score": min(bear_score, 100),
        "reasons_bull": reasons_bull,
        "reasons_bear": reasons_bear,
    }


def send_alert_email(alerts: list, sender: str, app_password: str, recipient: str) -> None:
    """
    alerts: [{"symbol":..., "name":..., "side": "bullish"/"bearish",
               "score":..., "reasons":[...], "last_close":...}, ...]
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
        lines.append(
            f"【{a['symbol']} {a['name']}】{side_label} 綜合分數 {a['score']}／收盤 {a['last_close']:.2f}\n"
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
