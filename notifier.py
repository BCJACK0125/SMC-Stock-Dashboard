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
import config as _cfg
from positions import describe as describe_position


# ---------------------------------------------------------------------------
# 評分元件
#
# 把「特徵抽取」與「權重」分離，理由有二：
#   1. 權重原本是手工配的（15/15/10/15/8/7/5/20），從來沒有擬合過任何東西。
#      分離之後才能拿真實績效去擬合它們（見 fit_weights.py）。
#   2. 元件值本身是可檢視、可測試的，不會埋在一長串 += 裡面。
#
# 每個元件回傳 0~1：多數是 0/1 的布林訊號，ML 那項是連續值。
# ---------------------------------------------------------------------------
COMPONENT_NAMES = [
    "zone", "order_block", "fvg", "choch", "bos", "sweep",      # SMC
    "rsi", "macd", "ema_stack", "adx_di",                       # 技術指標
    "ml",                                                       # 機器學習
]

# 原始的手工權重。保留當作對照基準——任何擬合出來的權重都應該跟它比。
DEFAULT_WEIGHTS = {
    "zone": 15, "order_block": 15, "fvg": 10, "choch": 15, "bos": 8, "sweep": 7,
    "rsi": 5, "macd": 5, "ema_stack": 5, "adx_di": 5,
    "ml": 20,
}


def score_components(
    analyzer: SMCAnalyzer,
    recent_bars: int = 5,
    ind_df=None,
    ml_result: Optional[dict] = None,
    adx_trend_threshold: float = 20.0,
) -> Dict:
    """
    抽出多空各自的評分元件（每項 0~1），不套用任何權重。

    回傳 {"bull": {元件名: 值}, "bear": {...},
          "reasons_bull": [...], "reasons_bear": [...]}
    """
    df = analyzer.df
    recent_df = df.tail(recent_bars)
    bull = {k: 0.0 for k in COMPONENT_NAMES}
    bear = {k: 0.0 for k in COMPONENT_NAMES}
    reasons_bull, reasons_bear = [], []

    # ------------------------------------------------------------ A. SMC --
    # 1) 溢價 / 折價區
    if analyzer.current_zone:
        if analyzer.current_zone["zone"] == "discount":
            bull["zone"] = 1.0
            reasons_bull.append("價格位於折價區 (Discount Zone)")
        elif analyzer.current_zone["zone"] == "premium":
            bear["zone"] = 1.0
            reasons_bear.append("價格位於溢價區 (Premium Zone)")

    # 2) 未緩解 OB 是否近期被回踩（多空用同一條區間重疊判準）
    for ob in analyzer.active_order_blocks("bullish"):
        if recent_df["Low"].min() <= ob.top and recent_df["High"].max() >= ob.bottom:
            bull["order_block"] = 1.0
            reasons_bull.append(f"價格回踩看漲 Order Block（{ob.bottom:.2f}~{ob.top:.2f}）")
            break
    for ob in analyzer.active_order_blocks("bearish"):
        if recent_df["High"].max() >= ob.bottom and recent_df["Low"].min() <= ob.top:
            bear["order_block"] = 1.0
            reasons_bear.append(f"價格觸及看跌 Order Block（{ob.bottom:.2f}~{ob.top:.2f}）")
            break

    # 3) 未回補 FVG 是否近期被觸碰
    for fvg in analyzer.active_fvgs("bullish"):
        if recent_df["Low"].min() <= fvg.top and recent_df["High"].max() >= fvg.bottom:
            bull["fvg"] = 1.0
            reasons_bull.append(f"價格修補看漲 FVG（{fvg.bottom:.2f}~{fvg.top:.2f}）")
            break
    for fvg in analyzer.active_fvgs("bearish"):
        if recent_df["High"].max() >= fvg.bottom and recent_df["Low"].min() <= fvg.top:
            bear["fvg"] = 1.0
            reasons_bear.append(f"價格修補看跌 FVG（{fvg.bottom:.2f}~{fvg.top:.2f}）")
            break

    # 4) 最新結構事件（CHoCH 與 BOS 分成兩個獨立元件，才能各自擬合權重）
    last_ev = analyzer.last_structure_event()
    if last_ev and last_ev.index in recent_df.index:
        key = "choch" if last_ev.type == "CHoCH" else "bos"
        if last_ev.side == "bullish":
            bull[key] = 1.0
            reasons_bull.append(f"近期出現看漲 {last_ev.type}")
        else:
            bear[key] = 1.0
            reasons_bear.append(f"近期出現看跌 {last_ev.type}")

    # 5) 流動性掃蕩後反轉：近期穿透 EQL/EQH 後，收盤又站回來
    last_close = float(df["Close"].iloc[-1])
    recent_start = recent_df.index[0]

    def swept_recently(pool) -> bool:
        return pool.swept and pool.swept_index is not None and pool.swept_index >= recent_start

    for pool in analyzer.liquidity_pools:
        if pool.kind == "EQL" and swept_recently(pool) and last_close > pool.price:
            bull["sweep"] = 1.0
            reasons_bull.append(f"偵測到流動性掃蕩後收回（EQL Sweep @ {pool.price:.2f}）")
            break
    for pool in analyzer.liquidity_pools:
        if pool.kind == "EQH" and swept_recently(pool) and last_close < pool.price:
            bear["sweep"] = 1.0
            reasons_bear.append(f"偵測到流動性掃蕩後收回（EQH Sweep @ {pool.price:.2f}）")
            break

    # ---------------------------------------------------- B. 技術指標共振 --
    if ind_df is not None and len(ind_df) > 0:
        row = ind_df.iloc[-1]
        if row["rsi"] > 50:
            bull["rsi"] = 1.0
        elif row["rsi"] < 50:
            bear["rsi"] = 1.0
        if row["macd_hist"] > 0:
            bull["macd"] = 1.0
        elif row["macd_hist"] < 0:
            bear["macd"] = 1.0
        if row["ema20"] > row["ema50"] > row["ema200"]:
            bull["ema_stack"] = 1.0
        elif row["ema20"] < row["ema50"] < row["ema200"]:
            bear["ema_stack"] = 1.0
        if row["adx"] >= adx_trend_threshold:
            if row["plus_di"] > row["minus_di"]:
                bull["adx_di"] = 1.0
            else:
                bear["adx_di"] = 1.0

        n_bull = sum(bull[k] for k in ("rsi", "macd", "ema_stack", "adx_di"))
        n_bear = sum(bear[k] for k in ("rsi", "macd", "ema_stack", "adx_di"))
        if n_bull:
            reasons_bull.append(f"技術指標共振：{int(n_bull)} 項偏多訊號一致")
        if n_bear:
            reasons_bear.append(f"技術指標共振：{int(n_bear)} 項偏空訊號一致")

    # ---------------------------------------------------- C. ML 模型機率 --
    # 連續值：機率偏離 50% 越多，元件值越接近 1
    if ml_result is not None:
        prob_up = ml_result["prob_up"]
        strength = min(abs(prob_up - 0.5) * 2, 1.0)
        if prob_up > 0.5:
            bull["ml"] = strength
            reasons_bull.append(
                f"輕量ML模型預測未來{ml_result['horizon']}根K棒上漲機率 {prob_up:.0%}"
                f"（訓練樣本 {ml_result['trained_rows']} 筆）")
        elif prob_up < 0.5:
            bear["ml"] = strength
            reasons_bear.append(
                f"輕量ML模型預測未來{ml_result['horizon']}根K棒上漲機率僅 {prob_up:.0%}"
                f"（訓練樣本 {ml_result['trained_rows']} 筆）")

    return {"bull": bull, "bear": bear,
            "reasons_bull": reasons_bull, "reasons_bear": reasons_bear}


def score(
    analyzer: SMCAnalyzer,
    recent_bars: int = 5,
    ind_df=None,
    ml_result: Optional[dict] = None,
    adx_trend_threshold: float = 20.0,
    weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    把評分元件依權重加總成 0~100 的多空分數。

    weights 為 None 時使用 DEFAULT_WEIGHTS（原本的手工權重）。
    """
    w = weights or DEFAULT_WEIGHTS
    comp = score_components(analyzer, recent_bars, ind_df, ml_result, adx_trend_threshold)
    bull_score = sum(w.get(k, 0.0) * v for k, v in comp["bull"].items())
    bear_score = sum(w.get(k, 0.0) * v for k, v in comp["bear"].items())
    return {
        "bull_score": min(round(bull_score), 100),
        "bear_score": min(round(bear_score), 100),
        "reasons_bull": comp["reasons_bull"],
        "reasons_bear": comp["reasons_bear"],
        "components_bull": comp["bull"],
        "components_bear": comp["bear"],
    }


def send_alert_email(alerts: list, sender: str, app_password: str, recipient: str,
                     concentration_note: Optional[list] = None,
                     position_status: Optional[list] = None) -> None:
    """
    alerts: [{"symbol":..., "name":..., "side": "bullish"/"bearish",
               "score":..., "reasons":[...], "last_close":...,
               "threshold_used":..., "backtest_win_rate":..., "backtest_n":...,
               "backtest_confidence":...}, ...]
    使用 Gmail SMTP (smtp.gmail.com:465, SSL)。
    寄件帳號需先開啟兩步驟驗證，並產生「應用程式密碼」(App Password) 供 app_password 使用，
    不要直接用登入密碼。
    """
    breached = [t for t in (position_status or []) if t.get("breached")]
    # 持倉跌破停損是最需要立刻知道的事，就算當天沒有任何新訊號也要寄。
    if not alerts and not breached:
        return

    if breached:
        subject = f"🚨 SMC 持倉警報：{len(breached)} 檔跌破移動停損"
    else:
        subject = f"📈 SMC 選股警報：{len(alerts)} 檔標的觸發訊號"
    lines = []
    for a in alerts:
        side_label = "多方 🟢" if a["side"] == "bullish" else "空方 🔴"
        bt_wr = a.get("backtest_win_rate")
        bt_lb = a.get("backtest_win_rate_lb")
        bt_n = a.get("backtest_n")
        bt_conf = a.get("backtest_confidence")
        conf_label = {"ok": "",
                      "low": "（勝率下界未達標，僅供參考）",
                      "no_edge": "  ⚠️ 歷史上證明不了正期望值，僅供參考",
                      "insufficient_data": "（回測樣本不足，門檻為預設值，僅供參考）"}.get(bt_conf, "")
        # 期望值才是「這樣做會不會賺錢」的直接答案，所以擺在勝率前面。
        # 勝率單獨看會騙人：實測有勝率 59% 但每筆虧 0.75% 的組合。
        exp_ = a.get("backtest_expectancy")
        exp_lb = a.get("backtest_expectancy_lb")
        payoff = a.get("backtest_payoff")
        exp_str = (f"扣成本期望值 {exp_:+.2%}/筆"
                   + (f"（下界 {exp_lb:+.2%}）" if exp_lb is not None else "")
                   ) if exp_ is not None else "期望值 —"
        payoff_str = f"，賠率 {payoff:.2f}" if payoff else ""
        lb_str = f"（下界 {bt_lb:.0%}）" if bt_lb is not None else ""
        bt_line = (f"  ↳ 回測：門檻 {a.get('threshold_used')} 分、{bt_n} 筆不重疊樣本\n"
                   f"     {exp_str}｜勝率 {bt_wr:.0%}{lb_str}{payoff_str}{conf_label}") if bt_wr is not None else \
                  f"  ↳ 回測：樣本不足，使用預設門檻 {a.get('threshold_used')} 分{conf_label}"
        # 複利績效：期望值是算術平均，回答不了「這樣做一年賺幾%、贏不贏
        # 買進持有」。以獲利為目標時，這兩個數字才是真正的判準。
        pf = a.get("performance") or {}
        perf_line = ""
        if pf.get("n_trades"):
            bh = pf.get("buy_hold_return")
            verdict = ("" if bh is None else
                       ("，勝過買進持有" if pf.get("beats_buy_hold") else "，⚠️ 不如買進持有"))
            perf_line = ("\n     回測期間複利："
                         f"總報酬 {pf['total_return']:+.0%}"
                         f"（年化 {(pf.get('cagr') or 0):+.1%}"
                         f"、最大回撤 {pf['max_drawdown']:.0%}"
                         f"、在場 {(pf.get('time_in_market') or 0):.0%}）"
                         + (f"｜買進持有 {bh:+.0%}{verdict}" if bh is not None else ""))

        # 下單指示：你收到信時市場已經收盤，需要的是「隔天怎麼下單」，
        # 而不是「剛才應該用什麼價買」。
        ep = a.get("entry_plan")
        tp = a.get("trade_plan")
        order_lines = []
        if ep is not None:
            order_lines.append(f"  ▸ 進場：{ep.describe()}")
        if tp is not None:
            # 只給一個停損數字：結構停損與移動停損起點取較高者，本來就是
            # 同一個東西。給兩個數字只會讓人不知道該在券商掛哪一個。
            mult = getattr(_cfg, "TRADE_TRAIL_ATR_MULT", 2.0)
            atr_now = a.get("atr")
            ref = ep.limit_price if ep is not None else tp.entry
            first = max(tp.stop, ref - mult * atr_now) if atr_now else tp.stop
            src = "OB 下緣" if abs(first - tp.stop) < 1e-9 else f"{mult:g}×ATR"
            order_lines.append(
                f"  ▸ 停損：{first:.2f}（{src}），風險 {abs(ref - first) / ref:.1%}")
            order_lines.append(
                f"  ▸ 出場：不設固定目標；停損每日隨進場後最高價上調"
                f"（{mult:g}×ATR 移動停損，只升不降）")
        sz = a.get("sizing")
        if sz:
            if sz.get("fraction", 0) > 0:
                order_lines.append(
                    f"  ▸ 部位：{sz['fraction']:.1%}（1/{sz.get('kelly_divisor', 4):.0f} Kelly；"
                    f"歷史先碰停利機率下界 {sz['p_lower']:.0%}，{sz['n']} 筆隨機進場樣本）")
            else:
                order_lines.append(f"  ▸ 部位：不建議下注 — {sz.get('reason', '')}")
        order_block = ("\n" + "\n".join(order_lines)) if order_lines else ""

        lines.append(
            f"【{a['symbol']} {a['name']}】{side_label} 綜合分數 {a['score']}／收盤 {a['last_close']:.2f}"
            + order_block + "\n"
            + bt_line + perf_line + "\n"
            + "\n".join(f"  - {r}" for r in a["reasons"])
        )
    body = "\n\n".join(lines)
    # 持倉的今日停損價。移動停損每天都在動，只給規則等於要使用者自己算。
    if position_status:
        head = "🚨 持倉已跌破停損，應出場" if breached else "📌 持倉今日停損"
        body += ("\n\n" + "─" * 46 + f"\n{head}\n"
                 + "\n".join("  " + describe_position(t) for t in position_status))
    # 集中度提醒：同一天多個訊號若高度相關，等於在同一個賭注上押多倍
    if concentration_note:
        body += "\n\n" + "─" * 46 + "\n📐 集中度提醒\n" + "\n".join(concentration_note)
    body += "\n\n（本郵件由 GitHub Actions 自動發送，僅供研究參考，非投資建議）"

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
