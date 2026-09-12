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
from typing import List, Dict, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot as plotly_plot

from smc.analyzer import SMCAnalyzer


# ---------------------------------------------------------------------------
# 圖表本體
# ---------------------------------------------------------------------------
def build_chart_html(
    symbol: str,
    analyzer: SMCAnalyzer,
    ind_df: Optional[pd.DataFrame] = None,
    lookback_bars: int = 180,
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
    return {"ok": "可信", "low": "勝率未達標", "insufficient_data": "樣本不足"}.get(conf, "—")


def _confidence_class(conf: str) -> str:
    return {"ok": "conf-ok", "low": "conf-low", "insufficient_data": "conf-na"}.get(conf, "conf-na")


def _backtest_summary_html(bt: Optional[Dict]) -> str:
    """把某標的多空兩邊的回測結果，組成一小段摘要 HTML，用在圖表卡片內。"""
    if not bt:
        return '<div class="bt-summary muted">尚無回測資料</div>'

    def side_line(label: str, rec: Dict) -> str:
        wr = rec.get("win_rate")
        n = rec.get("n", 0)
        conf = rec.get("confidence", "insufficient_data")
        wr_txt = f"{wr:.0%}" if wr is not None else "—"
        return (f'<div class="bt-row">'
                f'<span class="bt-side">{label}</span>'
                f'<span class="bt-metric">建議門檻 <b>{rec["threshold"]}</b></span>'
                f'<span class="bt-metric">歷史勝率 <b>{wr_txt}</b>（{n} 筆訊號）</span>'
                f'<span class="bt-conf {_confidence_class(conf)}">{_confidence_label(conf)}</span>'
                f'</div>')

    return (f'<div class="bt-summary">'
            f'<div class="bt-title">📊 Walk-forward 回測（樣本外，未看未來資料）'
            f'<span class="muted">· 共評估 {bt.get("n_evaluated_bars", 0)} 根K棒</span></div>'
            + side_line("多方 Long", bt["bull"])
            + side_line("空方 Short", bt["bear"])
            + '</div>')


# ---------------------------------------------------------------------------
# 整頁組合
# ---------------------------------------------------------------------------
def build_index_html(results: List[Dict], output_path: str = "index.html") -> None:
    """
    results: 每個標的的分析結果字典，需包含：
        symbol, name, chart_html, bull_score, bear_score, zone, last_close,
        last_event(str), alert(bool), generated_at
        選填：prev_close（用來算漲跌%）、backtest（run_symbol_backtest 的回傳值）
    """
    generated_at = results[0]["generated_at"] if results else ""

    total = len(results)
    alert_count = sum(1 for r in results if r["alert"])
    bull_count = sum(1 for r in results if r["bull_score"] >= 50)
    bear_count = sum(1 for r in results if r["bear_score"] >= 50)

    # --- KPI 卡片列 ---
    kpi_html = f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <span class="kpi-label">追蹤標的</span>
            <span class="kpi-value">{total}</span>
        </div>
        <div class="kpi-card kpi-alert">
            <span class="kpi-label">🔔 觸發警報</span>
            <span class="kpi-value">{alert_count}</span>
        </div>
        <div class="kpi-card kpi-bull">
            <span class="kpi-label">偏多訊號</span>
            <span class="kpi-value">{bull_count}</span>
        </div>
        <div class="kpi-card kpi-bear">
            <span class="kpi-label">偏空訊號</span>
            <span class="kpi-value">{bear_count}</span>
        </div>
    </div>"""

    # --- 清單表格 ---
    rows_html = ""
    for r in results:
        close_prev = r.get("prev_close")
        if close_prev:
            chg = (r["last_close"] - close_prev) / close_prev * 100
            chg_color = "#f0475d" if chg >= 0 else "#16c784"
            chg_sign = "+" if chg >= 0 else ""
            chg_html = f'<span style="color:{chg_color};">{chg_sign}{chg:.2f}%</span>'
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
            <td class="muted">{r.get('bull_threshold', '—')} / {r.get('bear_threshold', '—')}</td>
            <td class="muted">{r['last_event']}</td>
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
    th, td {{ padding: 13px 16px; text-align: left; font-size: 13.5px; }}
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
        .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
        table {{ font-size: 12px; }}
        .chart-card-header {{ flex-direction: column; }}
    }}
</style>
</head>
<body>
<header class="hero">
    <div class="brand"><span class="brand-dot"></span>Smart Money Concepts · Auto Dashboard</div>
    <h1>SMC 股票分析儀表板</h1>
    <p>依 Order Block / FVG / 結構轉變 自動計算綜合勝率分數，<span class="live">每日由 GitHub Actions 自動更新</span> · 最後更新 {generated_at}</p>
</header>

<nav class="subnav">{nav_chips}</nav>

<div class="container">
    {kpi_html}

    <div class="section-title">追蹤清單 Watchlist</div>
    <div class="table-wrap">
        <table>
            <thead>
                <tr>
                    <th>標的</th><th>漲跌%</th><th>收盤</th><th>位階</th>
                    <th>多方分數</th><th>空方分數</th><th>回測門檻(多/空)</th><th>最新結構事件</th><th>狀態</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>

    <div class="section-title">個股圖表 Charts</div>
    {charts_html}
</div>
<footer>本頁面僅供技術分析研究使用，不構成投資建議，請自行審慎判斷並承擔交易風險。資料來源：Yahoo Finance (yfinance)。</footer>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
