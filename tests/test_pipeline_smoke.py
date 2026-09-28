# -*- coding: utf-8 -*-
"""
tests/test_pipeline_smoke.py
=============================
真正執行 main.run()，而不只是 import。

為什麼需要這個測試：
    先前把 "entry_plan" 加進 results.append 時，只確認了字串能匹配，
    沒有檢查那個 dict 在檔案裡的**位置**是否在變數定義之後。結果
    results.append（第 255 行）引用了到第 284 行才定義的 entry_plan，
    正式排程一跑就 UnboundLocalError。

    當時的驗證方式是 `import main`——但 import **不會執行函式主體**，
    所以「變數在使用之後才定義」這類錯誤完全看不出來。

    這個測試用假資料真的跑一次 run()，把整條流程的執行路徑走過。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def synthetic_ohlcv(n=420, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, n)))
    op = close * (1 + rng.normal(0, 0.004, n))
    hi = np.maximum(op, close) * (1 + abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(op, close) * (1 - abs(rng.normal(0, 0.006, n)))
    return pd.DataFrame(
        {"Open": op, "High": hi, "Low": lo, "Close": close,
         "Volume": rng.integers(1_000_000, 9_000_000, n)},
        index=pd.date_range("2021-01-01", periods=n, freq="B"))


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """把輸出導到暫存目錄，並用合成資料取代網路下載。"""
    import config
    import main

    monkeypatch.setattr(config, "OUTPUT_HTML", str(tmp_path / "index.html"))
    monkeypatch.setattr(config, "BACKTEST_RESULTS_JSON", str(tmp_path / "bt.json"))
    monkeypatch.setattr(config, "BACKTEST_CACHE_JSON", str(tmp_path / "cache.json"))
    monkeypatch.setattr(config, "ALERT_STATE_JSON", str(tmp_path / "alert.json"))
    monkeypatch.setattr(config, "POSITIONS_JSON", str(tmp_path / "positions.json"))
    # 縮小規模：冒煙測試的價值在「走過整條執行路徑」，不在資料量
    monkeypatch.setattr(config, "BACKTEST_WARMUP_BARS", 200)
    monkeypatch.setattr(config, "TRADE_TRAILING_MAX_HOLDING_BARS", 40)
    monkeypatch.setattr(config, "SIZING_SAMPLES", 40)

    data = {"AAA": synthetic_ohlcv(seed=1), "BBB": synthetic_ohlcv(seed=2),
            "TINY": synthetic_ohlcv(n=90, seed=3)}    # 資料太短，測 fallback
    monkeypatch.setattr(main, "fetch_data", lambda s, **kw: data[s])
    monkeypatch.setattr(config, "WATCHLIST", [
        {"symbol": "AAA", "name": "甲"},
        {"symbol": "BBB", "name": "乙"},
        {"symbol": "TINY", "name": "丙"},
    ])
    return main, config, tmp_path


def test_run_completes_and_writes_outputs(pipeline):
    """
    最重要的一個測試：整條流程真的跑得完。
    `import main` 不會執行 run()，抓不到未定義變數這類錯誤。
    """
    main, config, tmp = pipeline
    main.run(dry_run=True)
    assert (tmp / "index.html").exists()
    assert (tmp / "bt.json").exists()
    assert (tmp / "index.html").stat().st_size > 10_000


def test_symbol_with_too_little_data_does_not_break_the_run(pipeline):
    """上市太短的標的（SNDK/SPCX/SKHY 那類）應被容忍，不該中斷整場排程。"""
    main, config, tmp = pipeline
    main.run(dry_run=True)          # TINY 只有 90 根 < warmup 200
    assert (tmp / "index.html").exists()


def test_cache_is_written_and_reused(pipeline):
    """第二次執行應該命中快取，不再重算。"""
    import backtest_cache
    main, config, tmp = pipeline
    main.run(dry_run=True)
    cache = backtest_cache.load(str(tmp / "cache.json"))
    assert cache, "第一次執行後應該要有快取"
    fp = backtest_cache.config_fingerprint(config)
    assert all(v["fingerprint"] == fp for v in cache.values())
    main.run(dry_run=True)          # 再跑一次不該出錯


def test_dry_run_never_sends_mail(pipeline, monkeypatch):
    main, config, tmp = pipeline
    sent = []
    monkeypatch.setattr(main, "send_alert_email",
                        lambda *a, **k: sent.append(1))
    main.run(dry_run=True)
    assert sent == []


def test_dashboard_contains_the_action_section(pipeline):
    main, config, tmp = pipeline
    main.run(dry_run=True)
    html = (tmp / "index.html").read_text(encoding="utf-8")
    assert "今日行動" in html
    assert "這些數字代表什麼" in html      # 方法論揭露


def test_open_positions_appear_on_the_dashboard(pipeline, tmp_path, monkeypatch):
    """持倉追蹤要真的走過 run()：算停損、進 HTML。"""
    import json
    main, config, tmp = pipeline
    pos = tmp_path / "positions.json"
    pos.write_text(json.dumps(
        [{"symbol": "AAA", "entry_date": "2022-01-03", "entry_price": 100.0}]),
        encoding="utf-8")
    monkeypatch.setattr(config, "POSITIONS_JSON", str(pos))
    main.run(dry_run=True)
    html = (tmp / "index.html").read_text(encoding="utf-8")
    assert "持倉追蹤" in html
    assert "今日停損" in html


def test_no_positions_shows_how_to_add_one(pipeline, monkeypatch):
    """沒有持倉時要說明怎麼填，否則這個功能對沒用過的人是隱形的。"""
    main, config, tmp = pipeline
    monkeypatch.setattr(config, "POSITIONS_JSON", str(tmp / "nope.json"))
    main.run(dry_run=True)
    html = (tmp / "index.html").read_text(encoding="utf-8")
    assert "positions.json" in html
    assert "entry_price" in html          # 可直接複製的範例
    assert "目前沒有持倉紀錄" in html


def test_a_broken_positions_file_does_not_stop_the_run(pipeline, tmp_path, monkeypatch):
    main, config, tmp = pipeline
    bad = tmp_path / "positions.json"
    bad.write_text("{ 這不是 JSON", encoding="utf-8")
    monkeypatch.setattr(config, "POSITIONS_JSON", str(bad))
    main.run(dry_run=True)
    assert (tmp / "index.html").exists()


def test_a_position_for_an_untracked_symbol_is_reported(pipeline, tmp_path,
                                                        monkeypatch, capsys):
    """代號打錯就等於停損從此不更新，必須出聲。"""
    import json
    main, config, tmp = pipeline
    pos = tmp_path / "positions.json"
    pos.write_text(json.dumps(
        [{"symbol": "ZZZZ", "entry_date": "2022-01-03", "entry_price": 10.0}]),
        encoding="utf-8")
    monkeypatch.setattr(config, "POSITIONS_JSON", str(pos))
    main.run(dry_run=True)
    assert "ZZZZ" in capsys.readouterr().err
