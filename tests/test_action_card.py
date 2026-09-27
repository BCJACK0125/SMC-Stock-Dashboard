# -*- coding: utf-8 -*-
"""
tests/test_action_card.py
==========================
今日行動卡的「出場」欄。

原本這一欄是寫死的字串「移動停損 2×ATR」——改了 config 也不會變，而且
只給規則不給數字。每天排程都算得出當天的停損價，沒有理由不給。
"""
from __future__ import annotations

import pytest

import config
import plot_report
import trade_model
from trade_model import TradePlan


def result(atr=2.0, close=100.0, stop=92.0, alert=True):
    ep = trade_model.plan_entry(close, atr, offset_atr=0.5, valid_bars=1,
                                fallback="market")
    tp = TradePlan(side="bullish", entry=close, stop=stop, target=120.0,
                   stop_source="order_block", target_source="liquidity")
    return {"symbol": "X", "name": "甲", "bull_score": 60, "alert": alert,
            "entry_plan": ep, "trade_plan": tp, "atr": atr,
            "backtest": {"bull": {"confidence": "ok"}}}


def test_exit_row_shows_an_actual_price(monkeypatch):
    monkeypatch.setattr(config, "TRADE_TRAIL_ATR_MULT", 2.0)
    html = plot_report._entry_plan_html([result()])
    # 限價 100 − 0.5×2 = 99；99 − 2×2 = 95，比結構停損 92 高
    assert "95.00" in html
    assert "2×ATR" in html


def test_exit_row_follows_the_configured_multiplier(monkeypatch):
    monkeypatch.setattr(config, "TRADE_TRAIL_ATR_MULT", 4.0)
    html = plot_report._entry_plan_html([result(stop=85.0)])
    assert "4×ATR" in html
    assert "91.00" in html          # 99 − 4×2 = 91，比結構停損 85 高


def test_structural_stop_wins_when_it_is_the_looser_of_the_two(monkeypatch):
    """移動停損不能比結構停損更鬆——初始值取兩者較高者。"""
    monkeypatch.setattr(config, "TRADE_TRAIL_ATR_MULT", 4.0)
    html = plot_report._entry_plan_html([result(stop=96.0)])
    assert "96.00" in html


def test_falls_back_to_the_rule_text_without_atr(monkeypatch):
    monkeypatch.setattr(config, "TRADE_TRAIL_ATR_MULT", 3.0)
    r = result()
    r["atr"] = 0.0
    html = plot_report._entry_plan_html([r])
    assert "移動停損" in html and "3×ATR" in html


def test_no_alerts_gives_the_hold_cash_message():
    html = plot_report._entry_plan_html([result(alert=False)])
    assert "維持現金部位" in html


# ---------------------------------------------------------------------------
# 郵件內文
# ---------------------------------------------------------------------------
class _FakeSMTP:
    sent = []

    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def login(self, *a): pass
    def sendmail(self, sender, to, msg): _FakeSMTP.sent.append(msg)


@pytest.fixture
def capture_mail(monkeypatch):
    import smtplib
    _FakeSMTP.sent = []
    monkeypatch.setattr(smtplib, "SMTP_SSL", _FakeSMTP)
    return _FakeSMTP.sent


def alert(atr=2.0):
    r = result(atr=atr)
    return {"symbol": "X", "name": "甲", "side": "bullish", "score": 60,
            "reasons": ["測試"], "last_close": 100.0, "threshold_used": 50,
            "backtest_win_rate": 0.5, "backtest_n": 40,
            "backtest_confidence": "ok", "atr": atr,
            "entry_plan": r["entry_plan"], "trade_plan": r["trade_plan"]}


def body_of(msg: str) -> str:
    import base64
    from email import message_from_string
    return message_from_string(msg).get_payload(0).get_payload(decode=True).decode("utf-8")


def test_email_exit_line_gives_a_price_not_just_a_rule(capture_mail, monkeypatch):
    import notifier
    monkeypatch.setattr(notifier._cfg, "TRADE_TRAIL_ATR_MULT", 4.0)
    notifier.send_alert_email([alert()], "a@b.c", "pw", "d@e.f")
    b = body_of(capture_mail[0])
    assert "4×ATR" in b
    assert "91.00" in b or "92.00" in b       # max(結構停損, 99 − 4×2)


def test_a_breached_position_sends_mail_even_with_no_signals(capture_mail):
    import notifier
    st = [{"symbol": "X", "stop": 95.0, "last": 94.0, "unrealized_pct": -0.06,
           "stop_distance_pct": -0.01, "locked_in": False, "breached": True}]
    notifier.send_alert_email([], "a@b.c", "pw", "d@e.f", position_status=st)
    assert capture_mail, "持倉跌破停損必須寄信，不能因為當天沒訊號就沉默"
    assert "跌破" in body_of(capture_mail[0])


def test_nothing_at_all_sends_no_mail(capture_mail):
    import notifier
    notifier.send_alert_email([], "a@b.c", "pw", "d@e.f", position_status=[])
    assert capture_mail == []


def test_healthy_positions_ride_along_with_a_signal_mail(capture_mail):
    import notifier
    st = [{"symbol": "X", "stop": 95.0, "last": 110.0, "unrealized_pct": 0.10,
           "stop_distance_pct": 0.14, "locked_in": True, "breached": False}]
    notifier.send_alert_email([alert()], "a@b.c", "pw", "d@e.f",
                              position_status=st)
    b = body_of(capture_mail[0])
    assert "持倉今日停損" in b and "95.00" in b
