# -*- coding: utf-8 -*-
"""
plot_report.py
================
把 SMCAnalyzer 算出來的指標畫成互動式 K 線圖（Plotly），
並把所有股票的圖表 + 清單表格 + 警報，組合成單一 index.html。
"""

from __future__ import annotations
from typing import List, Dict
import plotly.graph_objects as go
from plotly.offline import plot as plotly_plot

from smc.analyzer import SMCAnalyzer


def build_chart_html(symbol: str, analyzer: SMCAnalyzer, lookback_bars: int = 180) -> str:
    """把單一標的的 K 線 + SMC 指標畫成一段 HTML（含 <div>），回傳字串。"""
    df = analyzer.df.tail(lookback_bars)
    start_ts = df.index[0]

    fig = go.Figure()

    # --- K 線本體 ---
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=symbol,
        increasing_line_color="#e74c3c", decreasing_line_color="#2ecc71",  # 台股習慣：紅漲綠跌
    ))

    # --- Order Blocks（半透明區塊）---
    for ob in analyzer.order_blocks:
        if ob.start_index < start_ts:
            continue
        color = "rgba(46, 204, 113, 0.18)" if ob.side == "bullish" else "rgba(231, 76, 60, 0.18)"
        line_color = "#27ae60" if ob.side == "bullish" else "#c0392b"
        x_end = ob.mitigated_index if ob.mitigated_index else df.index[-1]
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=ob.start_index, x1=x_end, y0=ob.bottom, y1=ob.top,
            fillcolor=color, line=dict(color=line_color, width=1),
            opacity=0.5 if not ob.mitigated else 0.15,
            layer="below",
        )

    # --- Fair Value Gaps ---
    for fvg in analyzer.fvgs:
        if fvg.start_index < start_ts:
            continue
        color = "rgba(52, 152, 219, 0.20)" if fvg.side == "bullish" else "rgba(241, 196, 15, 0.20)"
        x_end = fvg.filled_index if fvg.filled_index else df.index[-1]
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=fvg.start_index, x1=x_end, y0=fvg.bottom, y1=fvg.top,
            fillcolor=color, line=dict(width=0),
            opacity=0.6 if not fvg.filled else 0.2,
            layer="below",
        )

    # --- BOS / CHoCH 標記與虛線 ---
    for ev in analyzer.structure_events:
        if ev.index < start_ts:
            continue
        color = "#2980b9" if ev.side == "bullish" else "#e67e22"
        dash = "dot" if ev.type == "BOS" else "dash"
        fig.add_shape(
            type="line", xref="x", yref="y",
            x0=ev.broken_index, x1=ev.index, y0=ev.broken_level, y1=ev.broken_level,
            line=dict(color=color, width=1.5, dash=dash),
        )
        fig.add_annotation(
            x=ev.index, y=ev.broken_level, text=ev.type,
            showarrow=False, yshift=10 if ev.side == "bullish" else -10,
            font=dict(size=10, color=color),
        )

    # --- 溢價 / 折價 / 均衡線 ---
    if analyzer.current_zone:
        z = analyzer.current_zone
        for label, y, dash in [("Premium 高點", z["top"], "solid"),
                                ("均衡 50%", z["mid"], "dash"),
                                ("Discount 低點", z["bottom"], "solid")]:
            fig.add_shape(type="line", xref="paper", yref="y", x0=0, x1=1, y0=y, y1=y,
                          line=dict(color="#7f8c8d", width=1, dash=dash))
            fig.add_annotation(xref="paper", x=1.0, y=y, text=label, showarrow=False,
                               font=dict(size=9, color="#7f8c8d"), xanchor="left")

    fig.update_layout(
        title=f"{symbol}｜SMC 分析圖（近 {lookback_bars} 根K棒）",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=560,
        margin=dict(l=40, r=90, t=50, b=30),
        font=dict(family="'Noto Sans TC', sans-serif", size=12),
    )

    return plotly_plot(fig, include_plotlyjs=False, output_type="div")


def build_index_html(results: List[Dict], output_path: str = "index.html") -> None:
    """
    results: 每個標的的分析結果字典，需包含：
        symbol, name, chart_html, bull_score, bear_score, zone, last_close,
        last_event(str), alert(bool)
    """
    generated_at = results[0]["generated_at"] if results else ""

    rows_html = ""
    for r in results:
        badge = ""
        if r["alert"]:
            badge = '<span class="badge badge-alert">🔔 高分警報</span>'
        elif r["bull_score"] >= 50:
            badge = '<span class="badge badge-bull">偏多</span>'
        elif r["bear_score"] >= 50:
            badge = '<span class="badge badge-bear">偏空</span>'
        else:
            badge = '<span class="badge badge-neutral">中性</span>'

        rows_html += f"""
        <tr onclick="document.getElementById('chart-{r['symbol']}').scrollIntoView({{behavior:'smooth'}})">
            <td>{r['symbol']}</td>
            <td>{r['name']}</td>
            <td>{r['last_close']:.2f}</td>
            <td>{r['zone']}</td>
            <td>{r['bull_score']}</td>
            <td>{r['bear_score']}</td>
            <td>{r['last_event']}</td>
            <td>{badge}</td>
        </tr>"""

    charts_html = ""
    for r in results:
        charts_html += f"""
        <section class="chart-card" id="chart-{r['symbol']}">
            <h2>{r['symbol']}｜{r['name']}
                <span class="score">多方分數 {r['bull_score']} / 空方分數 {r['bear_score']}</span>
            </h2>
            {r['chart_html']}
        </section>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SMC 股票分析儀表板</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
    :root {{
        --bg: #0f1419; --card: #1a2029; --text: #e8eaed; --muted: #8a94a6;
        --accent: #3b82f6; --green: #22c55e; --red: #ef4444; --amber: #f59e0b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
        margin: 0; font-family: 'Noto Sans TC', -apple-system, sans-serif;
        background: var(--bg); color: var(--text); line-height: 1.6;
    }}
    header {{ padding: 32px 24px 16px; text-align: center; }}
    header h1 {{ margin: 0 0 6px; font-size: 26px; }}
    header p {{ color: var(--muted); margin: 0; font-size: 13px; }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 0 20px 60px; }}
    table {{
        width: 100%; border-collapse: collapse; margin: 24px 0;
        background: var(--card); border-radius: 12px; overflow: hidden;
    }}
    th, td {{ padding: 12px 14px; text-align: left; font-size: 13.5px; }}
    th {{ background: #232a35; color: var(--muted); font-weight: 500; }}
    tr {{ border-top: 1px solid #262d38; cursor: pointer; }}
    tr:hover {{ background: #212836; }}
    .badge {{ padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 600; }}
    .badge-alert {{ background: rgba(245,158,11,0.18); color: var(--amber); }}
    .badge-bull {{ background: rgba(34,197,94,0.18); color: var(--green); }}
    .badge-bear {{ background: rgba(239,68,68,0.18); color: var(--red); }}
    .badge-neutral {{ background: rgba(138,148,166,0.18); color: var(--muted); }}
    .chart-card {{
        background: var(--card); border-radius: 12px; padding: 16px;
        margin-bottom: 24px;
    }}
    .chart-card h2 {{ font-size: 16px; margin: 4px 8px 8px; display: flex;
        justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }}
    .score {{ font-size: 12px; color: var(--muted); font-weight: 400; }}
    footer {{ text-align: center; color: var(--muted); font-size: 12px; padding: 20px; }}
</style>
</head>
<body>
<header>
    <h1>📊 SMC 股票分析儀表板</h1>
    <p>Smart Money Concepts｜自動更新於 {generated_at}（GitHub Actions）</p>
</header>
<div class="container">
    <table>
        <thead>
            <tr>
                <th>代號</th><th>名稱</th><th>收盤</th><th>位階</th>
                <th>多方分數</th><th>空方分數</th><th>最新結構事件</th><th>狀態</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    {charts_html}
</div>
<footer>本頁面僅供技術分析研究使用，不構成投資建議。資料來源：Yahoo Finance (yfinance)。</footer>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
