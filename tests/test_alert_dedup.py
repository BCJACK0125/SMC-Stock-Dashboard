# -*- coding: utf-8 -*-
"""
tests/test_alert_dedup.py
==========================
警報去重：同一個訊號會連續好幾天成立，不該天天重寄同一封信。
"""
from __future__ import annotations

from datetime import timedelta

import pytest

import config
import main


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """每個測試用自己的狀態檔，不碰到專案目錄。"""
    monkeypatch.setattr(config, "ALERT_STATE_JSON",
                        str(tmp_path / "alert_state.json"), raising=False)
    monkeypatch.setattr(config, "ALERT_COOLDOWN_DAYS", 5, raising=False)
    monkeypatch.setattr(config, "ALERT_RENOTIFY_SCORE_JUMP", 15, raising=False)


def alert(symbol="2330.TW", side="bullish", score=60):
    return {"symbol": symbol, "side": side, "score": score,
            "threshold_used": 45, "name": "測試", "reasons": [], "last_close": 1.0}


NOW = main.datetime(2026, 1, 10, tzinfo=main.TAIPEI_TZ)


def test_first_alert_always_goes_out():
    fresh, state = main.filter_new_alerts([alert()], now=NOW)
    assert len(fresh) == 1
    assert "2330.TW|bullish" in state


def test_same_signal_is_suppressed_within_cooldown():
    _, state = main.filter_new_alerts([alert()], now=NOW)
    main.save_alert_state(state)
    fresh, _ = main.filter_new_alerts([alert()], now=NOW + timedelta(days=2))
    assert fresh == [], "冷卻期內不該重複通知"


def test_signal_fires_again_after_cooldown():
    _, state = main.filter_new_alerts([alert()], now=NOW)
    main.save_alert_state(state)
    fresh, _ = main.filter_new_alerts([alert()], now=NOW + timedelta(days=6))
    assert len(fresh) == 1


def test_big_score_jump_bypasses_cooldown():
    """分數從 60 跳到 80，代表訊號明顯增強，值得再通知一次。"""
    _, state = main.filter_new_alerts([alert(score=60)], now=NOW)
    main.save_alert_state(state)
    fresh, _ = main.filter_new_alerts([alert(score=80)],
                                      now=NOW + timedelta(days=1))
    assert len(fresh) == 1


def test_small_score_drift_stays_suppressed():
    _, state = main.filter_new_alerts([alert(score=60)], now=NOW)
    main.save_alert_state(state)
    fresh, _ = main.filter_new_alerts([alert(score=65)],
                                      now=NOW + timedelta(days=1))
    assert fresh == []


def test_opposite_side_is_tracked_separately():
    _, state = main.filter_new_alerts([alert(side="bullish")], now=NOW)
    main.save_alert_state(state)
    fresh, _ = main.filter_new_alerts([alert(side="bearish")],
                                      now=NOW + timedelta(days=1))
    assert len(fresh) == 1, "多空是不同訊號，不該互相壓抑"


def test_different_symbols_are_tracked_separately():
    _, state = main.filter_new_alerts([alert(symbol="2330.TW")], now=NOW)
    main.save_alert_state(state)
    fresh, _ = main.filter_new_alerts([alert(symbol="AAPL")],
                                      now=NOW + timedelta(days=1))
    assert len(fresh) == 1


def test_state_is_not_saved_until_caller_saves_it():
    """
    寄信失敗時不該記錄成「已通知」，否則這個訊號會被冷卻期吃掉而永遠
    不再發出。所以 filter 只回傳狀態，由呼叫端在寄信成功後才存檔。
    """
    main.filter_new_alerts([alert()], now=NOW)          # 不呼叫 save
    fresh, _ = main.filter_new_alerts([alert()], now=NOW + timedelta(days=1))
    assert len(fresh) == 1, "沒存檔就不該被視為已通知過"


def test_corrupt_state_file_does_not_block_alerts(tmp_path):
    with open(config.ALERT_STATE_JSON, "w", encoding="utf-8") as f:
        f.write("{ 這不是合法 JSON")
    fresh, _ = main.filter_new_alerts([alert()], now=NOW)
    assert len(fresh) == 1, "狀態檔壞掉時應該照常通知，而不是整個靜音"


def test_unparsable_timestamp_is_treated_as_stale():
    _, state = main.filter_new_alerts([alert()], now=NOW)
    state["2330.TW|bullish"]["at"] = "not-a-timestamp"
    main.save_alert_state(state)
    fresh, _ = main.filter_new_alerts([alert()], now=NOW)
    assert len(fresh) == 1
