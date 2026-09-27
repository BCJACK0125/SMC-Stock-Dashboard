# -*- coding: utf-8 -*-
"""
tests/test_backtest_cache.py
=============================
回測快取。最重要的性質是「改了設定就自動失效」——靠人記得清快取不可靠。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

import backtest_cache as BC


def cfg(**over):
    base = dict(SWING_LOOKBACK=3, EQ_TOLERANCE_PCT=0.0015, FVG_MIN_GAP_ATR=0.5,
                BACKTEST_CANDIDATE_THRESHOLDS=[35, 40, 45],
                TRADE_EXIT_MODE="trailing", TRADE_TRAIL_ATR_MULT=2.0,
                ENTRY_FALLBACK="market", ML_ENABLED=False)
    base.update(over)
    return SimpleNamespace(**base)


NOW = datetime(2026, 9, 27, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# 設定指紋
# --------------------------------------------------------------------------
def test_same_config_same_fingerprint():
    assert BC.config_fingerprint(cfg()) == BC.config_fingerprint(cfg())


def test_changing_any_relevant_setting_changes_the_fingerprint():
    """任何影響回測的參數變動都必須讓指紋改變，否則會沿用過時的結果。"""
    base = BC.config_fingerprint(cfg())
    for k, v in (("SWING_LOOKBACK", 5), ("FVG_MIN_GAP_ATR", 0.8),
                 ("TRADE_EXIT_MODE", "fixed"), ("TRADE_TRAIL_ATR_MULT", 3.0),
                 ("ENTRY_FALLBACK", "skip"), ("ML_ENABLED", True),
                 ("BACKTEST_CANDIDATE_THRESHOLDS", [35, 40])):
        assert BC.config_fingerprint(cfg(**{k: v})) != base, f"{k} 改了但指紋沒變"


def test_irrelevant_setting_does_not_invalidate():
    """不影響回測的設定（例如輸出檔名）不該白白讓快取失效。"""
    assert BC.config_fingerprint(cfg(OUTPUT_HTML="a.html")) == \
           BC.config_fingerprint(cfg(OUTPUT_HTML="b.html"))


# --------------------------------------------------------------------------
# 保鮮判斷
# --------------------------------------------------------------------------
def entry(fp="abc", days_ago=0):
    return {"fingerprint": fp,
            "computed_at": (NOW - timedelta(days=days_ago)).isoformat(),
            "result": {"ok": True}}


def test_fresh_within_window():
    assert BC.is_fresh(entry(days_ago=3), "abc", 7, now=NOW) is True


def test_stale_beyond_window():
    assert BC.is_fresh(entry(days_ago=10), "abc", 7, now=NOW) is False


def test_fingerprint_mismatch_beats_freshness():
    """設定變了就算是今天算的也不能用。"""
    assert BC.is_fresh(entry(fp="old", days_ago=0), "new", 7, now=NOW) is False


def test_missing_or_corrupt_entry_is_not_fresh():
    assert BC.is_fresh({}, "abc", 7, now=NOW) is False
    assert BC.is_fresh({"fingerprint": "abc", "computed_at": "壞掉的時間"},
                       "abc", 7, now=NOW) is False


# --------------------------------------------------------------------------
# get_or_compute
# --------------------------------------------------------------------------
def test_compute_is_skipped_on_a_cache_hit():
    """命中快取時不該呼叫 compute——那才是省時間的重點。"""
    calls = []
    cache = {"X": entry()}
    cache["X"]["fingerprint"] = BC.config_fingerprint(cfg())
    r, hit = BC.get_or_compute(cache, "X", cfg(),
                               lambda: calls.append(1) or {"fresh": True},
                               max_age_days=7, now=NOW)
    assert hit is True and calls == []
    assert r == {"ok": True}


def test_compute_runs_on_a_miss_and_is_stored():
    cache = {}
    r, hit = BC.get_or_compute(cache, "X", cfg(), lambda: {"v": 1},
                               max_age_days=7, now=NOW)
    assert hit is False and r == {"v": 1}
    assert cache["X"]["fingerprint"] == BC.config_fingerprint(cfg())


def test_config_change_forces_recompute():
    cache = {}
    BC.get_or_compute(cache, "X", cfg(), lambda: {"v": 1}, now=NOW)
    r, hit = BC.get_or_compute(cache, "X", cfg(TRADE_TRAIL_ATR_MULT=3.0),
                               lambda: {"v": 2}, now=NOW)
    assert hit is False and r == {"v": 2}


def test_zero_max_age_disables_the_cache():
    cache = {}
    BC.get_or_compute(cache, "X", cfg(), lambda: {"v": 1}, max_age_days=0, now=NOW)
    _, hit = BC.get_or_compute(cache, "X", cfg(), lambda: {"v": 2},
                               max_age_days=0, now=NOW)
    assert hit is False


# --------------------------------------------------------------------------
# 讀寫與型別還原
# --------------------------------------------------------------------------
def test_round_trip(tmp_path):
    p = str(tmp_path / "c.json")
    cache = {}
    BC.get_or_compute(cache, "X", cfg(), lambda: {"v": 1}, now=NOW)
    assert BC.save(p, cache) is True
    assert BC.load(p)["X"]["result"] == {"v": 1}


def test_missing_or_corrupt_file_loads_as_empty(tmp_path):
    assert BC.load(str(tmp_path / "nope.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{ 這不是 JSON", encoding="utf-8")
    assert BC.load(str(bad)) == {}, "壞掉的快取不該讓整個排程失敗"


def test_timestamps_are_restored_for_chart_markers():
    """
    JSON 會把 Timestamp 存成字串，不還原的話圖表標記的
    `ts in df.index` 比對會全部落空，歷史進出場點就畫不出來。
    """
    r = BC.restore_timestamps({"trades": [
        {"entry_ts": "2024-01-02", "exit_ts": "2024-01-10", "ret": 0.05}]})
    t = r["trades"][0]
    assert isinstance(t["entry_ts"], pd.Timestamp)
    assert isinstance(t["exit_ts"], pd.Timestamp)


def test_restore_tolerates_missing_or_bad_values():
    r = BC.restore_timestamps({"trades": [{"ret": 0.01}, {"entry_ts": "不是日期"}]})
    assert r["trades"][0]["ret"] == 0.01
    assert r["trades"][1]["entry_ts"] == "不是日期"


def test_restore_handles_absent_trades():
    assert BC.restore_timestamps({"bull": {}}) == {"bull": {}}
