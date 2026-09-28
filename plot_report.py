# -*- coding: utf-8 -*-
"""
plot_report.py
================
把 SMCAnalyzer 算出來的指標畫成互動式 K 線圖（Plotly），
並把所有股票的圖表 + 清單表格 + 警報，組合成單一 index.html。

視覺設計參考主流金融/量化交易儀表板的通用語言：
    - 深色底、克制的強調色，長時間盯盤不刺眼
    - tabular numbers（數字等寬對齊，方便掃視比較）
    - 語意色（漲=紅／跌=綠，符合台股習慣）一律搭配文字或圖示，不單靠顏色
    - KPI 卡片列 + 卡片系統，取代單純的一張大表格
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Dict, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot as plotly_plot

from smc.analyzer import SMCAnalyzer
import config as _cfg
import exit_policy


# ---------------------------------------------------------------------------
# 圖表本體
# ---------------------------------------------------------------------------
def _add_trade_markers(fig, df: pd.DataFrame, trades: List[Dict]) -> int:
    """
    把回測選中的歷史交易畫到 K 線上：進場三角、出場叉、兩點連線。

    為什麼值得畫：回測的統計數字是匯總結果，看不出這套邏輯在哪些位置
    進出場。畫回圖上才能用肉眼檢查訊號是否合理——尤其能一眼看出那些
    撐起全部獲利的大贏家長什麼樣（實測前 10% 的交易貢獻全部報酬）。

    配色沿用本圖表的台股習慣（紅漲綠跌）：獲利=紅、虧損=綠。
    hover 文字會明講「獲利/虧損」，避免顏色語意被誤讀。
    """
    if not trades:
        return 0
    start, end = df.index[0], df.index[-1]
    shown = 0
    ex, ey, et, xx, xy, xt = [], [], [], [], [], []

    for t in trades:
        e_ts, x_ts = t["entry_ts"], t["exit_ts"]
        if e_ts < start or e_ts > end:
            continue
        if e_ts not in df.index:
            continue
        x_ts_eff = x_ts if (x_ts in df.index and x_ts <= end) else end
        # 用交易實際的成交價，不要拿收盤價重算：進場是隔天限價成交、出場是
        # 盤中打到停損價，兩者都不是收盤。舊版用收盤畫，30 筆裡 29 筆的
        # 「圖上兩點算出來的報酬」跟標籤顯示的報酬對不上（最大差 9 個百分點）。
        e_px = t.get("entry_px") or float(df["Close"].loc[e_ts])
        x_px = t.get("exit_px") or float(df["Close"].loc[x_ts_eff])
        win = t["ret"] > 0
        color = "#f0475d" if win else "#16c784"
        label = "獲利" if win else "虧損"

        fig.add_shape(type="line", xref="x", yref="y",
                      x0=e_ts, x1=x_ts_eff, y0=e_px, y1=x_px,
                      line=dict(color=color, width=1.2, dash="dot"),
                      opacity=0.75, layer="above", row=1, col=1)
        ex.append(e_ts); ey.append(e_px)
        et.append(f"進場 {e_px:.2f}<br>{label} {t['ret']:+.2%}<br>持有 {t['bars']} 根")
        xx.append(x_ts_eff); xy.append(x_px)
        xt.append(f"出場 {x_px:.2f}（{t.get('outcome', '?')}）<br>{label} {t['ret']:+.2%}")
        shown += 1

    if shown:
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="markers", name="回測進場",
            marker=dict(symbol="triangle-up", size=9, color="#e2e8f0",
                        line=dict(color="#0f172a", width=1)),
            hovertext=et, hoverinfo="text"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=xx, y=xy, mode="markers", name="回測出場",
            marker=dict(symbol="x", size=8, color="#94a3b8"),
            hovertext=xt, hoverinfo="text"), row=1, col=1)
    return shown


def build_chart_html(
    symbol: str,
    analyzer: SMCAnalyzer,
    ind_df: Optional[pd.DataFrame] = None,
    lookback_bars: int = 180,
    trades: Optional[List[Dict]] = None,
) -> str:
    """把單一標的的 K 線 + SMC 指標 (+ RSI/MACD副圖，若有提供 ind_df) 畫成一段 HTML。"""
    df = analyzer.df.tail(lookback_bars)
    start_ts = df.index[0]

    has_sub = ind_df is not None and len(ind_df) > 0
    if has_sub:
        sub = ind_df.reindex(df.index)
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.72, 0.28], vertical_spacing=0.03,
        )
    else:
        fig = make_subplots(rows=1, cols=1)

    # --- K 線本體（台股習慣：紅漲綠跌）---
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=symbol,
        increasing_line_color="#f0475d", increasing_fillcolor="#f0475d",
        decreasing_line_color="#16c784", decreasing_fillcolor="#16c784",
    ), row=1, col=1)

    # --- Order Blocks（半透明區塊）---
    for ob in analyzer.order_blocks:
        if ob.start_index < start_ts:
            continue
        color = "rgba(22, 199, 132, 0.16)" if ob.side == "bullish" else "rgba(240, 71, 93, 0.16)"
        line_color = "#16c784" if ob.side == "bullish" else "#f0475d"
        x_end = ob.mitigated_index if ob.mitigated_index else df.index[-1]
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=ob.start_index, x1=x_end, y0=ob.bottom, y1=ob.top,
            fillcolor=color, line=dict(color=line_color, width=1),
            opacity=0.55 if not ob.mitigated else 0.15,
            layer="below", row=1, col=1,
        )

    # --- Fair Value Gaps ---
    for fvg in analyzer.fvgs:
        if fvg.start_index < start_ts:
            continue
        color = "rgba(96, 165, 250, 0.22)" if fvg.side == "bullish" else "rgba(251, 191, 36, 0.22)"
        x_end = fvg.filled_index if fvg.filled_index else df.index[-1]
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=fvg.start_index, x1=x_end, y0=fvg.bottom, y1=fvg.top,
            fillcolor=color, line=dict(width=0),
            opacity=0.65 if not fvg.filled else 0.2,
            layer="below", row=1, col=1,
        )

    # --- BOS / CHoCH 標記與虛線 ---
    for ev in analyzer.structure_events:
        if ev.index < start_ts:
            continue
        color = "#60a5fa" if ev.side == "bullish" else "#fb923c"
        dash = "dot" if ev.type == "BOS" else "dash"
        fig.add_shape(
            type="line", xref="x", yref="y",
            x0=ev.broken_index, x1=ev.index, y0=ev.broken_level, y1=ev.broken_level,
            line=dict(color=color, width=1.5, dash=dash), row=1, col=1,
        )
        fig.add_annotation(
            x=ev.index, y=ev.broken_level, text=ev.type,
            showarrow=False, yshift=10 if ev.side == "bullish" else -10,
            font=dict(size=10, color=color), row=1, col=1,
        )

    # --- 溢價 / 折價 / 均衡線 ---
    if analyzer.current_zone:
        z = analyzer.current_zone
        for label, y, dash in [("Premium 高點", z["top"], "solid"),
                                ("均衡 50%", z["mid"], "dash"),
                                ("Discount 低點", z["bottom"], "solid")]:
            fig.add_shape(type="line", xref="x", yref="y", x0=df.index[0], x1=df.index[-1], y0=y, y1=y,
                          line=dict(color="#5b6472", width=1, dash=dash), row=1, col=1)
            fig.add_annotation(x=df.index[-1], y=y, text=label, showarrow=False,
                               font=dict(size=9, color="#8a94a6"), xanchor="left", row=1, col=1)

    # --- 回測歷史進出場點 ---
    _add_trade_markers(fig, df, trades or [])

    # --- 副圖：RSI + MACD ---
    if has_sub:
        fig.add_trace(go.Scatter(
            x=sub.index, y=sub["rsi"], name="RSI(14)",
            line=dict(color="#a78bfa", width=1.4),
        ), row=2, col=1)
        fig.add_hline(y=70, line=dict(color="#5b6472", width=1, dash="dot"), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="#5b6472", width=1, dash="dot"), row=2, col=1)

        hist_colors = ["#16c784" if v >= 0 else "#f0475d" for v in sub["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=sub.index, y=sub["macd_hist"], name="MACD 柱",
            marker_color=hist_colors, opacity=0.55, yaxis="y3",
        ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        height=650 if has_sub else 480,
        margin=dict(l=44, r=100, t=16, b=30),
        font=dict(family="Inter, 'Noto Sans TC', sans-serif", size=12, color="#c8ceda"),
        hoverlabel=dict(bgcolor="#1a2029", font_size=12, font_family="Inter"),
        legend=dict(orientation="h", y=1.02, x=0),
        bargap=0.1,
    )
    fig.update_xaxes(gridcolor="#242b38", showspikes=True, spikecolor="#4a5568", spikethickness=1)
    fig.update_yaxes(gridcolor="#242b38", side="right")
    if has_sub:
        fig.update_yaxes(title_text="RSI / MACD", row=2, col=1, range=[0, 100])

    return plotly_plot(fig, include_plotlyjs=False, output_type="div")


# ---------------------------------------------------------------------------
# 小工具：分數視覺化條
# ---------------------------------------------------------------------------
def _score_bar(value: int, color_var: str) -> str:
    return f"""<div class="score-bar-track">
        <div class="score-bar-fill" style="width:{value}%; background:var({color_var});"></div>
    </div>
    <span class="score-bar-num">{value}</span>"""


def _status_badge(r: Dict) -> str:
    if r["alert"]:
        return '<span class="badge badge-alert">🔔 高分警報</span>'
    if r["bull_score"] >= 50:
        return '<span class="badge badge-bull">● 偏多</span>'
    if r["bear_score"] >= 50:
        return '<span class="badge badge-bear">● 偏空</span>'
    return '<span class="badge badge-neutral">● 中性</span>'


def _zone_label(zone: str) -> str:
    return {"premium": "溢價區 Premium", "discount": "折價區 Discount",
            "equilibrium": "均衡區 Equilibrium"}.get(zone, "—")


def _zone_class(zone: str) -> str:
    return zone if zone in ("premium", "discount", "equilibrium") else "equilibrium"


def _confidence_label(conf: str) -> str:
    return {"ok": "可信", "low": "勝率下界未達標", "no_edge": "無可證實優勢",
            "insufficient_data": "樣本不足"}.get(conf, "—")


def _confidence_class(conf: str) -> str:
    return {"ok": "conf-ok", "low": "conf-low", "no_edge": "conf-low",
            "insufficient_data": "conf-na"}.get(conf, "conf-na")


def _backtest_summary_html(bt: Optional[Dict]) -> str:
    """把某標的多空兩邊的回測結果，組成一小段摘要 HTML，用在圖表卡片內。"""
    if not bt:
        return '<div class="bt-summary muted">尚無回測資料</div>'

    def side_line(label: str, rec: Dict) -> str:
        wr = rec.get("win_rate")
        n = rec.get("n", 0)
        conf = rec.get("confidence", "insufficient_data")
        exp_ = rec.get("expectancy")
        exp_lb = rec.get("expectancy_lb")
        payoff = rec.get("payoff")

        # 期望值（已扣來回交易成本）是挑門檻時實際採用的判準，擺第一個；
        # 勝率單獨看會騙人，所以一定跟賠率並列。
        exp_txt = (f'<span class="bt-metric">扣成本期望 <b>{exp_:+.2%}</b>'
                   + (f' <span class="muted">(下界 {exp_lb:+.2%})</span>' if exp_lb is not None else "")
                   + "</span>") if exp_ is not None else ""
        wr_txt = (f'<span class="bt-metric">勝率 <b>{wr:.0%}</b>'
                  + (f" × 賠率 <b>{payoff:.2f}</b>" if payoff else "")
                  + f'（{n} 筆不重疊）</span>') if wr is not None else                  f'<span class="bt-metric muted">（{n} 筆）</span>'
        return (f'<div class="bt-row">'
                f'<span class="bt-side">{label}</span>'
                f'<span class="bt-metric">建議門檻 <b>{rec["threshold"]}</b></span>'
                + exp_txt + wr_txt +
                f'<span class="bt-conf {_confidence_class(conf)}">{_confidence_label(conf)}</span>'
                f'</div>')

    return (f'<div class="bt-summary">'
            f'<div class="bt-title">📊 Walk-forward 回測（樣本外，未看未來資料）'
            f'<span class="muted">· 共評估 {bt.get("n_evaluated_bars", 0)} 根K棒'
            f'· 來回成本 {bt.get("cost", 0):.2%}</span></div>'
            + side_line("多方 Long", bt["bull"])
            + side_line("空方 Short", bt["bear"])
            + _performance_html(bt)
            + '</div>')


def _performance_html(bt: Optional[Dict]) -> str:
    """複利績效 vs 買進持有。以獲利為目標時，這一列才是真正的判準。"""
    pf = (bt or {}).get("bull", {}).get("performance") or {}
    if not pf.get("n_trades"):
        return ""
    bh = pf.get("buy_hold_return")
    beat = pf.get("beats_buy_hold")
    verdict = ("<span class=\"bt-conf conf-ok\">勝過買進持有</span>" if beat
               else "<span class=\"bt-conf conf-low\">不如買進持有</span>") if bh is not None else ""
    return (f'<div class="bt-row">'
            f'<span class="bt-side">多方複利</span>'
            f'<span class="bt-metric">總報酬 <b>{pf["total_return"]:+.0%}</b></span>'
            f'<span class="bt-metric">年化 <b>{(pf.get("cagr") or 0):+.1%}</b></span>'
            f'<span class="bt-metric">最大回撤 <b>{pf["max_drawdown"]:.0%}</b></span>'
            f'<span class="bt-metric">在場 <b>{(pf.get("time_in_market") or 0):.0%}</b></span>'
            + (f'<span class="bt-metric muted">買進持有 {bh:+.0%}'
               f'（回撤 {pf.get("buy_hold_max_drawdown", 0):.0%}）</span>' if bh is not None else "")
            + verdict +
            f'</div>')


# ---------------------------------------------------------------------------
# 整頁組合
# ---------------------------------------------------------------------------
def _entry_plan_html(results: List[Dict]) -> str:
    """
    今日行動卡：有訊號的標的要「明天怎麼下單」。

    這是整個儀表板最實用的一塊——使用者的操作模式是平常持有現金、有訊號
    才進場，他收到通知時市場已收盤，需要的是可以直接照做的掛單指示，
    而不是「剛才的分數是幾分」。
    """
    acts = [r for r in results if r.get("alert") and r.get("entry_plan")]
    if not acts:
        return ('<div class="empty-state">今日無標的達到警報門檻 — '
                '維持現金部位，無須動作。</div>')

    cards = ""
    for r in acts:
        ep, tp = r["entry_plan"], r.get("trade_plan")
        conf = (r.get("backtest") or {}).get("bull", {}).get("confidence", "")
        warn = ('<div class="act-warn">⚠️ 此標的回測未能證實正期望值，'
                '訊號僅供參考</div>') if conf != "ok" else ""
        sz = r.get("sizing")
        size_html = ""
        if sz:
            if sz.get("fraction", 0) > 0:
                size_html = (f'<div class="act-row"><span>部位</span>'
                             f'<b>{sz["fraction"]:.1%}</b>'
                             f'<span class="muted">1/{sz.get("kelly_divisor", 4):.0f} Kelly · '
                             f'先碰停利機率下界 {sz["p_lower"]:.0%}</span></div>')
            else:
                size_html = ('<div class="act-row"><span>部位</span>'
                             '<b>不建議</b>'
                             f'<span class="muted">{sz.get("reason", "")}</span></div>')
        # 停損只給一個數字。先前分成「停損」與「出場」兩列，但那是同一個
        # 東西的兩種算法取較高者——在 5×ATR 之下兩者多半相等，看到兩個數字
        # 只會讓人不知道該在券商掛哪一個。
        mult = getattr(_cfg, "TRADE_TRAIL_ATR_MULT", 2.0)
        atr_now = r.get("atr")
        stop_html = ""
        if tp is not None:
            first = max(tp.stop, ep.limit_price - mult * atr_now) if atr_now else tp.stop
            src = ("OB 下緣" if abs(first - tp.stop) < 1e-9
                   else f"{mult:g}×ATR")
            risk = abs(ep.limit_price - first) / ep.limit_price if ep.limit_price else 0
            stop_html = (f'<div class="act-row"><span>停損</span>'
                         f'<b>{first:.2f}</b>'
                         f'<span class="muted">{src} · 風險 {risk:.1%}</span></div>'
                         f'<div class="act-note">不設固定目標；停損每日隨進場後'
                         f'最高價上調（{mult:g}×ATR 移動停損，只升不降）</div>')
        cards += f"""
        <div class="act-card">
            <div class="act-head">{r['symbol']} <span class="muted">{r['name']}</span>
                <span class="act-score">{r['bull_score']} 分</span></div>
            <div class="act-row"><span>限價</span><b>{ep.limit_price:.2f}</b>
                <span class="muted">收盤 {ep.reference_close:.2f} · 未成交轉市價</span></div>
            {stop_html}
            {size_html}
            {warn}
        </div>"""
    return f'<div class="act-grid">{cards}</div>'


def _action_banner_html(results: List[Dict],
                        position_status: Optional[List[Dict]]) -> str:
    """
    頁面最上方的一句話：今天到底要不要動作。

    這個儀表板的使用情境是「平常抱現金、每天瞄一眼」，所以第一眼要回答的
    不是分數幾分，而是「有沒有事」。出場排在進場前面——手上的部位跌破停損
    比錯過一個新訊號急迫得多。
    """
    breached = [t for t in (position_status or []) if t["breached"]]
    acts = [r for r in results if r.get("alert") and r.get("entry_plan")]

    if breached:
        names = "、".join(f"{t['symbol']} @ {t['stop']:.2f}" for t in breached)
        return (f'<div class="banner banner-exit"><span class="banner-icon">🚨</span>'
                f'<div><b>{len(breached)} 檔持倉跌破移動停損，應出場</b>'
                f'<span class="banner-sub">{names}</span></div></div>')
    if acts:
        names = "、".join(r["symbol"] for r in acts)
        return (f'<div class="banner banner-entry"><span class="banner-icon">🔔</span>'
                f'<div><b>{len(acts)} 檔達到進場門檻</b>'
                f'<span class="banner-sub">{names}　·　收盤後掛單，見下方「今日行動」</span>'
                f'</div></div>')
    return ('<div class="banner banner-calm"><span class="banner-icon">☕</span>'
            '<div><b>今天沒有事要做</b>'
            '<span class="banner-sub">無訊號、持倉也都在停損之上——維持現金部位</span>'
            '</div></div>')


def _risk_policy_html() -> str:
    """
    風險設定：使用者設的回撤預算、換算出來的出場寬度，以及**它守不住**。

    這一塊的重點不是炫耀可調整，而是揭露這個旋鈕實際上能做到什麼。
    把預算講成保證，比不提供這個旋鈕還糟。
    """
    d = exit_policy.describe(getattr(_cfg, "DRAWDOWN_BUDGET", 0.40))
    floor = ('<p class="risk-warn">已經是最保守的一檔，再往下調換不到更低的'
             '回撤，只會換到更低的報酬。</p>' if d["floor_reached"] else "")
    return f"""
    <details class="risk-box">
      <summary>風險設定：可接受回撤 {d['budget']:.0%} → 移動停損
        {d['trail_atr_mult']:g}×ATR</summary>
      <div class="risk-body">
        <div class="risk-grid">
          <div><span class="risk-label">出場寬度</span>
               <span class="risk-val">{d['trail_atr_mult']:g}×ATR</span></div>
          <div><span class="risk-label">校準期年化（中位）</span>
               <span class="risk-val">{d['expected_cagr']:+.1%}</span></div>
          <div><span class="risk-label">校準期回撤（中位）</span>
               <span class="risk-val">{d['expected_drawdown']:.1%}</span></div>
          <div><span class="risk-label">樣本外超出預算</span>
               <span class="risk-val warn">{d['oos_breach']} 檔</span></div>
        </div>
        <p class="risk-warn"><b>這個預算是傾向，不是保證。</b>
          上面的回撤是 22 檔的<b>中位數</b>——樣本外仍有
          <b>{d['oos_breach']}</b> 檔個別超過了 {d['budget']:.0%}。
          把它當成「大概會落在哪一區」，不要當成上限。</p>
        {floor}
        <p><b>能調的其實是報酬，不是回撤。</b>不論預算設多少，樣本外的實際
          回撤都落在 −31%~−34%；收緊停損只會少賺，不會少跌
          （1×ATR 年化 3.5% vs 5×ATR 年化 15.3%，兩者回撤相同）。</p>
        <p class="muted">寬度是所有標的共用的。逐檔挑最佳寬度實測無效——
          用前五年挑、後五年驗證，21 檔只有 1 檔挑對（5%），比隨機猜
          8 選 1（12.5%）還差。改 <code>config.DRAWDOWN_BUDGET</code> 可調整。</p>
      </div>
    </details>"""


def _positions_html(status: Optional[List[Dict]]) -> str:
    """
    持倉追蹤：移動停損每天都在動，所以每天都要給出當天的數字。

    只有 positions.json 裡有資料才會出現這個區塊——沒在持倉的人不需要看到
    一個空表格。
    """
    mult = getattr(_cfg, "TRADE_TRAIL_ATR_MULT", 2.0)
    if not status:
        return f"""
    <div class="section-title">持倉追蹤 Open Positions</div>
    <div class="empty-state">
        <p style="margin:0 0 8px;">目前沒有持倉紀錄。</p>
        <p class="muted" style="margin:0 0 10px;">實際成交後，把這段填進 repo 根目錄的
            <code>positions.json</code>，之後每天就會算出當天的移動停損價
            （{mult:g}×ATR，只升不降），跌破時也會單獨寄信通知。</p>
        <pre class="code-hint">[{{"symbol": "NVDA", "entry_date": "2026-09-28", "entry_price": 178.50}}]</pre>
    </div>"""
    rows = ""
    for t in status:
        cls = "pos-breach" if t["breached"] else ("pos-locked" if t["locked_in"] else "")
        note = ("⚠️ 已跌破，應出場" if t["breached"]
                else ("🔒 停損已高於成本" if t["locked_in"] else ""))
        up = t["unrealized_pct"]
        col = "#f0475d" if up >= 0 else "#16c784"
        rows += f"""
        <tr class="{cls}">
            <td><span class="sym-code">{t['symbol']}</span>
                <span class="sym-name">{t.get('name', '')}</span></td>
            <td class="num">{t['entry_price']:.2f}</td>
            <td class="num">{t['last']:.2f}</td>
            <td class="num" style="color:{col};">{up:+.1%}</td>
            <td class="num"><b>{t['stop']:.2f}</b></td>
            <td class="num">{t['stop_distance_pct']:.1%}</td>
            <td class="num col-sec">{t['entry_date']}</td>
            <td class="num col-sec">{t['bars_held']} 根</td>
            <td>{note}</td>
        </tr>"""
    return f"""
    <div class="section-title">持倉追蹤 Open Positions</div>
    <div class="table-wrap">
        <table>
            <thead><tr>
                <th>標的</th><th class="num">進場價</th><th class="num">現價</th>
                <th class="num">損益</th><th class="num">今日停損</th>
                <th class="num">距停損</th><th class="num col-sec">進場日</th>
                <th class="num col-sec">持有</th><th></th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    <p class="scroll-hint">← 左右滑動可看更多欄位 →</p>
    <p class="muted" style="margin-top:8px;">
        停損 = max(初始停損, 進場後最高價 − {mult:g}×ATR)，只升不降。
        資料來自 repo 根目錄的 positions.json，實際成交後自行填入。
    </p>"""


def _concentration_html(note: Optional[List[str]]) -> str:
    """
    集中度提醒：同時多個訊號時，它們實際上是幾個獨立風險。

    concentration.describe() 產生的是純文字（給郵件用），裡面的 **粗體**
    是 Markdown 語法，在 HTML 裡不會渲染，這裡轉成 <b>。
    """
    if not note:
        return ""
    import html as _html
    import re as _re

    def fmt(line: str) -> str:
        safe = _html.escape(line)
        return _re.sub(r"\*\*(.+?)\*\*", lambda m: "<b>" + m.group(1) + "</b>", safe)

    body = "".join(f"<div class='conc-line'>{fmt(line)}</div>" for line in note)
    return (f'<div class="section-title">集中度 Concentration</div>'
            f'<div class="conc-box">{body}</div>')


def _methodology_html() -> str:
    """
    方法論與局限的誠實揭露。

    放在儀表板上而不只是 README，是因為看板上的數字最容易被當成保證。
    這裡的每一句都對應 EXPERIMENTS.md 裡的實測結果。
    """
    return """
    <details class="method">
      <summary>這些數字代表什麼、不代表什麼（務必先讀）</summary>
      <div class="method-body">
        <p><b>回測怎麼做的：</b>Walk-forward 樣本外、逐根K棒只用當下已知資訊；
        扣除來回交易成本（台股 0.47%／ETF 0.27%／美股 0.023%）；
        不重疊進場；門檻用扣成本期望值的信賴下界挑選，並做 Šidák 多重比較修正。</p>
        <p><b>「可信」代表什麼：</b>只代表該標的在歷史資料上，這個門檻的
        扣成本期望值下界大於 0。<b>不代表未來會賺錢。</b></p>
        <p><b>已知的局限（實測結果）：</b></p>
        <ul>
          <li>訊號相對「隨機進場」的超額期望值接近零，且會隨參數變號。</li>
          <li>跨 22 檔、4 段滾動驗證：超額為正的比例 44%（擲硬幣是 50%）。</li>
          <li>ML 元件樣本外 AUC 僅 0.518，已預設關閉。</li>
          <li>獲利高度集中：前 10% 的交易貢獻全部報酬，其餘淨虧損。</li>
          <li>同期年化：本策略 15.3%、SPY 15.1%、QQQ 20.8%、SMH 半導體 ETF
              34.6%。<b>買一檔指數 ETF 抱著就贏過這套系統</b>，而那不需要
              任何選股或擇時。</li>
          <li>策略真正的優勢有兩個：在場時間只有 39%（每單位曝險的報酬
              39.6% 高於買進持有的 32.1%），以及 2022 熊市中位 −18.8%
              對買進持有的 −38.6%。</li>
          <li>「折價區」這個 15 分的評分元件，在 1,062 筆訊號裡只觸發 5 次
              ——實際進場中位落在近 60 日區間的第 88 百分位（買在高點）。
              但實測買高並沒有比較差，所以未更動進場邏輯。</li>
        </ul>
        <p class="muted">完整實驗紀錄與被推翻的假設見 repo 的 EXPERIMENTS.md。</p>
      </div>
    </details>"""


def build_index_html(results: List[Dict], output_path: str = "index.html",
                     concentration_note: Optional[List[str]] = None,
                     position_status: Optional[List[Dict]] = None) -> None:
    """
    results: 每個標的的分析結果字典，需包含：
        symbol, name, chart_html, bull_score, bear_score, zone, last_close,
        last_event(str), alert(bool), generated_at
        選填：prev_close（用來算漲跌%）、backtest（run_symbol_backtest 的回傳值）
    """
    generated_at = results[0]["generated_at"] if results else ""
    generated_iso = datetime.now(timezone.utc).isoformat()

    acts = [r for r in results if r.get("alert") and r.get("entry_plan")]
    pos = position_status or []
    breached = [t for t in pos if t["breached"]]
    unreal = sum(t["unrealized_pct"] for t in pos) / len(pos) if pos else None

    # KPI 回答的是「我今天要做什麼」，不是「系統跑了幾檔」。
    # 追蹤標的數之類的系統狀態放到頁尾就好。
    def kpi(label, value, cls="", sub=""):
        sub_html = f'<span class="kpi-sub">{sub}</span>' if sub else ""
        return (f'<div class="kpi-card {cls}"><span class="kpi-label">{label}</span>'
                f'<span class="kpi-value">{value}</span>{sub_html}</div>')

    kpi_html = f"""
    <div class="kpi-grid">
        {kpi("今日要出場", len(breached), "kpi-bear" if breached else "",
             "跌破移動停損" if breached else "持倉都在停損之上")}
        {kpi("今日可進場", len(acts), "kpi-alert" if acts else "",
             "已達警報門檻" if acts else "維持現金")}
        {kpi("持倉中", len(pos), "",
             f"平均 {unreal:+.1%}" if unreal is not None else "未填 positions.json")}
        {kpi("回測可信的標的", sum(1 for r in results
             if (r.get("backtest") or {}).get("bull", {}).get("confidence") == "ok"),
             "kpi-bull", f"共追蹤 {len(results)} 檔")}
    </div>"""

    # --- 清單表格 ---
    rows_html = ""
    for r in results:
        close_prev = r.get("prev_close")
        if close_prev:
            chg = (r["last_close"] - close_prev) / close_prev * 100
            # 台股慣例紅漲綠跌，但同一頁的多方分數又是綠色——顏色會互相矛盾。
            # 加上箭頭，讓漲跌不依賴顏色就讀得出來。
            up = chg >= 0
            chg_color = "#f0475d" if up else "#16c784"
            arrow = "▲" if up else "▼"
            chg_html = (f'<span class="chg" style="color:{chg_color};">'
                        f'{arrow} {abs(chg):.2f}%</span>')
        else:
            chg_html = '<span class="muted">—</span>'

        zone_cls = _zone_class(r["zone"])
        rows_html += f"""
        <tr onclick="document.getElementById('chart-{r['symbol']}').scrollIntoView({{behavior:'smooth', block:'start'}})">
            <td>
                <div class="sym-cell">
                    <span class="sym-code">{r['symbol']}</span>
                    <span class="sym-name">{r['name']}</span>
                </div>
            </td>
            <td class="num">{chg_html}</td>
            <td class="num">{r['last_close']:.2f}</td>
            <td><span class="zone-pill zone-{zone_cls}">{_zone_label(r['zone'])}</span></td>
            <td class="num"><div class="score-cell">{_score_bar(r['bull_score'], '--green')}</div></td>
            <td class="num"><div class="score-cell">{_score_bar(r['bear_score'], '--red')}</div></td>
            <td class="muted col-sec">{r.get('bull_threshold', '—')} / {r.get('bear_threshold', '—')}</td>
            <td class="muted col-sec">{r['last_event']}</td>
            <td>{_status_badge(r)}</td>
        </tr>"""

    # --- 個股導覽列（sticky sub-nav，方便跳轉）---
    nav_chips = "".join(
        f'<a href="#chart-{r["symbol"]}" class="nav-chip">{r["symbol"]}</a>' for r in results
    )

    # --- 各標的圖表卡片 ---
    charts_html = ""
    for r in results:
        zone_cls = _zone_class(r["zone"])
        ml_html = ""
        if r.get("ml_prob_up") is not None:
            p = r["ml_prob_up"]
            ml_html = f'<div class="mini-score"><span class="dot dot-ml"></span>ML 上漲機率 {p:.0%}</div>'
        charts_html += f"""
        <section class="chart-card" id="chart-{r['symbol']}">
            <div class="chart-card-header">
                <div>
                    <h2>{r['symbol']} <span class="chart-card-name">{r['name']}</span></h2>
                    <span class="zone-pill zone-{zone_cls}">{_zone_label(r['zone'])}</span>
                    <span class="muted" style="margin-left:8px;">最新事件：{r['last_event']}</span>
                </div>
                <div class="chart-card-scores">
                    <div class="mini-score"><span class="dot dot-bull"></span>多方 {r['bull_score']}</div>
                    <div class="mini-score"><span class="dot dot-bear"></span>空方 {r['bear_score']}</div>
                    {ml_html}
                </div>
            </div>
            {_backtest_summary_html(r.get('backtest'))}
            {r['chart_html']}
        </section>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SMC 股票分析儀表板</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+TC:wght@400;500;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
    :root {{
        --bg: #0a0e14; --bg-elevated: #10151d; --card: #131922; --card-border: #1f2733;
        --text: #eef1f6; --muted: #7c8698; --muted-2: #566073;
        --accent: #5b8def; --accent-2: #7c6cf0;
        --green: #16c784; --red: #f0475d; --amber: #f5a623;
        --radius: 14px;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
        margin: 0; font-family: 'Inter', 'Noto Sans TC', -apple-system, sans-serif;
        background:
            radial-gradient(1200px 500px at 15% -10%, rgba(91,141,239,0.10), transparent),
            radial-gradient(900px 400px at 100% 0%, rgba(124,108,240,0.08), transparent),
            var(--bg);
        color: var(--text); line-height: 1.6; -webkit-font-smoothing: antialiased;
    }}
    .num {{ font-variant-numeric: tabular-nums; }}
    a {{ color: inherit; }}

    header.hero {{
        padding: 44px 24px 28px; text-align: center; position: relative;
        border-bottom: 1px solid var(--card-border);
    }}
    .brand {{
        display: inline-flex; align-items: center; gap: 8px; font-size: 13px;
        color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase;
        font-weight: 600; margin-bottom: 14px;
    }}
    .brand-dot {{ width: 7px; height: 7px; border-radius: 50%; background: var(--green);
        box-shadow: 0 0 10px var(--green); }}
    header.hero h1 {{
        margin: 0 0 10px; font-size: 32px; font-weight: 800; letter-spacing: -0.02em;
        background: linear-gradient(90deg, #eef1f6, #a9b6cc);
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }}
    header.hero p {{ color: var(--muted); margin: 0; font-size: 13.5px; }}
    header.hero p .live {{ color: var(--green); font-weight: 600; }}

    .subnav {{
        position: sticky; top: 0; z-index: 20;
        background: rgba(10,14,20,0.85); backdrop-filter: blur(10px);
        border-bottom: 1px solid var(--card-border);
        padding: 10px 24px; display: flex; gap: 8px; overflow-x: auto;
    }}
    .nav-chip {{
        flex: 0 0 auto; padding: 6px 14px; border-radius: 999px; font-size: 12.5px;
        font-weight: 600; text-decoration: none; color: var(--muted);
        background: var(--card); border: 1px solid var(--card-border); white-space: nowrap;
        transition: all .15s ease;
    }}
    .nav-chip:hover {{ color: var(--text); border-color: var(--accent); }}

    .container {{ max-width: 1180px; margin: 0 auto; padding: 28px 20px 60px; }}

    .act-grid {{ display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); }}
    .act-card {{ background:var(--card); border:1px solid var(--card-border); border-left:3px solid var(--amber);
                 border-radius:10px; padding:14px 16px; max-width:520px; }}
    .act-note {{ font-size:11.5px; color:var(--muted-2); line-height:1.6;
                 margin:6px 0 2px 50px; }}
    .act-head {{ font-weight:700; margin-bottom:10px; display:flex; align-items:center; gap:8px; }}
    .act-score {{ margin-left:auto; font-size:12px; color:var(--amber); font-weight:700; }}
    .act-row {{ display:flex; align-items:baseline; gap:8px; font-size:13px; padding:3px 0;
                flex-wrap:wrap; }}
    .act-row > span:first-child {{ width:42px; color:var(--muted); flex:none; }}
    .act-row .muted {{ flex:1 1 180px; min-width:0; }}
    .act-row b {{ font-variant-numeric:tabular-nums; font-size:15px; }}
    .act-warn {{ margin-top:8px; font-size:12px; color:var(--amber); }}
    tr.pos-breach {{ background:rgba(240,71,93,0.10); }}
    .banner {{
        display:flex; gap:14px; align-items:center; padding:16px 18px; margin-bottom:22px;
        border-radius:var(--radius); border:1px solid var(--card-border);
        background:linear-gradient(160deg, var(--card), var(--bg-elevated));
    }}
    .banner-icon {{ font-size:24px; line-height:1; flex:none; }}
    .banner b {{ display:block; font-size:16px; letter-spacing:-0.01em; }}
    .banner-sub {{ display:block; color:var(--muted); font-size:12.5px; margin-top:3px; }}
    .banner-exit {{ border-color:rgba(240,71,93,0.45); box-shadow:0 0 0 1px rgba(240,71,93,0.12) inset; }}
    .banner-exit b {{ color:var(--red); }}
    .banner-entry {{ border-color:rgba(245,166,35,0.45); }}
    .banner-entry b {{ color:var(--amber); }}
    .banner-calm b {{ color:var(--muted); font-weight:600; }}
    .chg {{ font-variant-numeric:tabular-nums; font-weight:600; }}
    .scroll-hint {{ display:none; }}
    .kpi-sub {{ font-size:11.5px; color:var(--muted-2); margin-top:-2px; }}
    .code-hint {{
        background:var(--bg); border:1px solid var(--card-border); border-radius:8px;
        padding:10px 12px; font-size:11.5px; overflow-x:auto; text-align:left;
        color:var(--muted); margin:0; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
    }}
    code {{ background:var(--bg-elevated); padding:1px 5px; border-radius:4px; font-size:12px; }}
    .risk-box {{ background:var(--card); border:1px solid var(--card-border); border-radius:10px;
                 padding:12px 16px; margin-bottom:18px; }}
    .risk-box summary {{ cursor:pointer; font-size:13.5px; font-weight:600; }}
    .risk-body {{ font-size:13px; line-height:1.75; margin-top:10px; }}
    .risk-grid {{ display:grid; gap:10px; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
                  margin-bottom:12px; }}
    .risk-label {{ display:block; color:var(--muted); font-size:11.5px; }}
    .risk-val {{ font-size:17px; font-variant-numeric:tabular-nums; font-weight:600; }}
    .risk-val.warn {{ color:var(--amber); }}
    .risk-warn {{ border-left:3px solid var(--amber); padding-left:10px; }}
    tr.pos-locked {{ background:rgba(22,199,132,0.08); }}
    .empty-state {{ background:var(--card); border:1px dashed var(--card-border); border-radius:10px;
                    padding:22px; text-align:center; color:var(--muted); }}
    .conc-box {{ background:var(--card); border:1px solid var(--card-border); border-radius:10px; padding:14px 16px; }}
    .conc-line {{ font-size:13px; line-height:1.7; white-space:pre-wrap; }}
    .method {{ background:var(--card); border:1px solid var(--card-border); border-radius:10px;
               padding:12px 16px; margin-bottom:18px; }}
    .method summary {{ cursor:pointer; font-weight:600; font-size:13px; }}
    .method-body {{ font-size:13px; line-height:1.75; color:var(--muted); margin-top:10px; }}
    .method-body b {{ color:var(--text); }}
    .method-body ul {{ margin:6px 0 6px 18px; }}
    .kpi-grid {{
        display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 26px;
    }}
    .kpi-card {{
        background: linear-gradient(160deg, var(--card), var(--bg-elevated));
        border: 1px solid var(--card-border); border-radius: var(--radius);
        padding: 18px 20px; display: flex; flex-direction: column; gap: 6px;
    }}
    .kpi-label {{ font-size: 12px; color: var(--muted); font-weight: 500; }}
    .kpi-value {{ font-size: 28px; font-weight: 800; font-variant-numeric: tabular-nums; }}
    .kpi-alert .kpi-value {{ color: var(--amber); }}
    .kpi-bull .kpi-value {{ color: var(--green); }}
    .kpi-bear .kpi-value {{ color: var(--red); }}

    .table-wrap {{
        background: var(--card); border: 1px solid var(--card-border);
        border-radius: var(--radius); overflow: hidden; margin-bottom: 34px;
        overflow-x: auto;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 13px 16px; text-align: left; font-size: 13.5px; white-space: nowrap; }}
    th {{
        background: var(--bg-elevated); color: var(--muted); font-weight: 600;
        font-size: 11.5px; letter-spacing: .04em; text-transform: uppercase;
        border-bottom: 1px solid var(--card-border);
    }}
    tbody tr {{ border-top: 1px solid var(--card-border); cursor: pointer; transition: background .12s; }}
    tbody tr:hover {{ background: var(--bg-elevated); }}
    .sym-cell {{ display: flex; flex-direction: column; gap: 2px; }}
    .sym-code {{ font-weight: 700; font-size: 14px; }}
    .sym-name {{ font-size: 12px; color: var(--muted); }}
    .muted {{ color: var(--muted); font-size: 12.5px; }}

    .zone-pill {{
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 11.5px; font-weight: 600; white-space: nowrap;
    }}
    .zone-premium {{ background: rgba(240,71,93,0.14); color: #ff8095; }}
    .zone-discount {{ background: rgba(22,199,132,0.14); color: #4fe3ac; }}
    .zone-equilibrium {{ background: rgba(124,134,152,0.14); color: var(--muted); }}

    .score-cell {{ display: flex; align-items: center; gap: 8px; min-width: 96px; }}
    .score-bar-track {{
        width: 56px; height: 6px; border-radius: 4px; background: #1c232f; overflow: hidden;
    }}
    .score-bar-fill {{ height: 100%; border-radius: 4px; }}
    .score-bar-num {{ font-size: 12.5px; font-weight: 700; font-variant-numeric: tabular-nums; }}

    .badge {{ padding: 4px 11px; border-radius: 999px; font-size: 11.5px; font-weight: 700;
        white-space: nowrap; }}
    .badge-alert {{ background: rgba(245,166,35,0.16); color: var(--amber); }}
    .badge-bull {{ background: rgba(22,199,132,0.16); color: var(--green); }}
    .badge-bear {{ background: rgba(240,71,93,0.16); color: var(--red); }}
    .badge-neutral {{ background: rgba(124,134,152,0.16); color: var(--muted); }}

    .section-title {{
        font-size: 13px; font-weight: 700; color: var(--muted); text-transform: uppercase;
        letter-spacing: .06em; margin: 0 0 14px 4px;
    }}
    .chart-card {{
        background: var(--card); border: 1px solid var(--card-border);
        border-radius: var(--radius); padding: 18px 18px 8px; margin-bottom: 20px;
    }}
    .chart-card-header {{
        display: flex; justify-content: space-between; align-items: flex-start;
        flex-wrap: wrap; gap: 10px; padding: 2px 6px 14px;
    }}
    .chart-card h2 {{ font-size: 17px; margin: 0 0 6px; font-weight: 700; }}
    .chart-card-name {{ font-size: 13px; font-weight: 500; color: var(--muted); margin-left: 6px; }}
    .chart-card-scores {{ display: flex; gap: 14px; }}
    .mini-score {{ font-size: 12.5px; color: var(--muted); display: flex; align-items: center; gap: 6px; font-weight: 600; }}
    .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
    .dot-bull {{ background: var(--green); }}
    .dot-bear {{ background: var(--red); }}
    .dot-ml {{ background: var(--accent-2); }}

    .bt-summary {{
        background: var(--bg-elevated); border: 1px solid var(--card-border);
        border-radius: 10px; padding: 12px 14px; margin: 4px 6px 14px; font-size: 12.5px;
    }}
    .bt-title {{ font-weight: 700; margin-bottom: 8px; }}
    .bt-row {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; padding: 4px 0; }}
    .bt-side {{ font-weight: 700; min-width: 78px; }}
    .bt-metric {{ color: var(--muted); }}
    .bt-metric b {{ color: var(--text); }}
    .bt-conf {{ margin-left: auto; padding: 2px 9px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
    .conf-ok {{ background: rgba(22,199,132,0.16); color: var(--green); }}
    .conf-low {{ background: rgba(245,166,35,0.16); color: var(--amber); }}
    .conf-na {{ background: rgba(124,134,152,0.16); color: var(--muted); }}

    footer {{
        text-align: center; color: var(--muted-2); font-size: 12px;
        padding: 26px 20px; border-top: 1px solid var(--card-border);
    }}

    @media (max-width: 860px) {{
        header.hero {{ padding: 30px 18px 22px; }}
        header.hero h1 {{ font-size: 24px; }}
        .container {{ padding: 20px 14px 48px; }}
        .kpi-grid {{ grid-template-columns: repeat(2, 1fr); gap: 10px; }}
        .kpi-card {{ padding: 14px 15px; }}
        .kpi-value {{ font-size: 23px; }}
        /* 桌機的 320px 最小寬在窄螢幕上會撐破版面 */
        .act-grid {{ grid-template-columns: 1fr; }}
        .risk-grid {{ grid-template-columns: repeat(2, 1fr); }}
        table {{ font-size: 12px; }}
        .col-sec {{ display: none; }}   /* 次要欄位讓位給可讀性 */
        /* 表格能橫向捲動並不明顯，直接講出來 */
        .scroll-hint {{ display:block; font-size:11.5px; color:var(--muted-2);
                        text-align:center; margin:6px 0 0; }}
        th, td {{ padding: 10px 12px; }}
        .chart-card {{ padding: 14px 12px 6px; }}
        .chart-card-header {{ flex-direction: column; }}
        .table-wrap {{ -webkit-overflow-scrolling: touch; }}
    }}
</style>
</head>
<body>
<header class="hero">
    <div class="brand"><span class="brand-dot"></span>Smart Money Concepts · Auto Dashboard</div>
    <h1>SMC 股票分析儀表板</h1>
    <p>依 Order Block / FVG / 結構轉變 計算綜合分數，並給出可直接照做的掛單與停損價
       · <span class="live">每日自動更新</span>
       · 最後更新 {generated_at}<span id="stale"></span></p>
</header>

<nav class="subnav">{nav_chips}</nav>

<div class="container">
    {_action_banner_html(results, position_status)}

    {kpi_html}

    {_positions_html(position_status)}

    <div class="section-title">今日行動 Today’s Orders</div>
    {_entry_plan_html(results)}

    {_concentration_html(concentration_note)}

    <div class="section-title">設定與揭露 Method &amp; Limits</div>
    {_risk_policy_html()}
    {_methodology_html()}

    <div class="section-title">追蹤清單 Watchlist</div>
    <div class="table-wrap">
        <table>
            <thead>
                <tr>
                    <th>標的</th><th>漲跌%</th><th>收盤</th><th>位階</th>
                    <th>多方分數</th><th>空方分數</th><th class="col-sec">回測門檻(多/空)</th><th class="col-sec">最新結構事件</th><th>狀態</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    <p class="scroll-hint">← 左右滑動可看更多欄位 →</p>

    <div class="section-title">個股圖表 Charts</div>
    {charts_html}
</div>
<footer>
  <div>追蹤 {len(results)} 檔 · 出場 {getattr(_cfg, "TRADE_TRAIL_ATR_MULT", 2):g}×ATR 移動停損 · 回撤預算 {getattr(_cfg, "DRAWDOWN_BUDGET", 0.4):.0%}</div>
  <div style="margin-top:6px;">本頁面僅供技術分析研究使用，不構成投資建議，請自行審慎判斷並承擔交易風險。資料來源：Yahoo Finance (yfinance)。</div>
</footer>
<script>
// 排程壞掉時頁面會靜靜地顯示舊資料，這裡讓它自己說出來。
(function () {{
  var gen = new Date("{generated_iso}");
  var days = (Date.now() - gen.getTime()) / 86400000;
  if (days > 1.5) {{
    var el = document.getElementById("stale");
    if (el) {{
      el.textContent = "（已 " + Math.floor(days) + " 天未更新，排程可能失敗）";
      el.style.color = "#f5a623";
      el.style.fontWeight = "600";
    }}
  }}
}})();
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
