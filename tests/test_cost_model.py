# -*- coding: utf-8 -*-
"""
tests/test_cost_model.py
=========================
交易成本模型。成本拆成手續費／稅費／價差三塊，換券商時才知道該調哪一個。
"""
from __future__ import annotations

import pytest

import config


@pytest.fixture(autouse=True)
def restore_config(monkeypatch):
    """每個測試改動 config 後自動還原。"""
    yield


# --------------------------------------------------------------------------
# 台股
# --------------------------------------------------------------------------
def test_taiwan_stock_includes_fee_and_full_transaction_tax():
    cost = config.transaction_cost("2330.TW")
    expected = 0.001425 * config.BROKER_FEE_DISCOUNT * 2 + 0.003
    assert cost == pytest.approx(expected)


def test_taiwan_etf_pays_lower_transaction_tax():
    """ETF 證交稅 0.1%，一般股票 0.3%——這個差距不小，不能混為一談。"""
    stock = config.transaction_cost("2330.TW", is_etf=False)
    etf = config.transaction_cost("0050.TW", is_etf=True)
    assert etf < stock
    assert stock - etf == pytest.approx(0.002)


def test_otc_suffix_is_treated_as_taiwan(monkeypatch):
    assert config.transaction_cost("6488.TWO") == config.transaction_cost("2330.TW")


def test_broker_discount_lowers_only_the_commission(monkeypatch):
    """券商折扣打得到手續費，打不到證交稅。"""
    monkeypatch.setattr(config, "BROKER_FEE_DISCOUNT", 1.0)
    full = config.transaction_cost("2330.TW")
    monkeypatch.setattr(config, "BROKER_FEE_DISCOUNT", 0.6)
    disc = config.transaction_cost("2330.TW")
    assert disc < full
    assert full - disc == pytest.approx(0.001425 * 2 * 0.4)


def test_us_broker_settings_do_not_affect_taiwan(monkeypatch):
    before = config.transaction_cost("2330.TW")
    monkeypatch.setattr(config, "US_COMMISSION_FREE", False)
    monkeypatch.setattr(config, "US_COMMISSION_PCT", 0.005)
    assert config.transaction_cost("2330.TW") == pytest.approx(before)


# --------------------------------------------------------------------------
# 美股
# --------------------------------------------------------------------------
def test_commission_free_us_still_has_a_nonzero_cost():
    """
    零佣不等於零成本：法規費（SEC fee + FINRA TAF）加上買賣價差與滑價
    仍然存在。把成本設成 0 會讓回測高估績效。
    """
    cost = config.transaction_cost("AAPL")
    assert cost > 0
    assert cost == pytest.approx(config.US_REGULATORY + config.US_SPREAD_SLIPPAGE)


def test_us_cost_is_an_order_of_magnitude_below_taiwan():
    assert config.transaction_cost("AAPL") < config.transaction_cost("2330.TW") / 10


def test_non_commission_free_broker_adds_both_sides(monkeypatch):
    monkeypatch.setattr(config, "US_COMMISSION_FREE", False)
    monkeypatch.setattr(config, "US_COMMISSION_PCT", 0.001)
    cost = config.transaction_cost("AAPL")
    assert cost == pytest.approx(0.002 + config.US_REGULATORY + config.US_SPREAD_SLIPPAGE)


def test_illiquid_symbols_can_be_modelled_with_wider_spread(monkeypatch):
    """流動性差的標的價差更寬，這個參數就是留給那種情況調的。"""
    base = config.transaction_cost("AAPL")
    monkeypatch.setattr(config, "US_SPREAD_SLIPPAGE", 0.002)
    assert config.transaction_cost("AAPL") > base


def test_cost_is_deterministic():
    assert config.transaction_cost("NVDA") == config.transaction_cost("NVDA")
