# -*- coding: utf-8 -*-
"""
trade_model.py
===============
把「訊號」轉成「一筆有停損、有目標的交易」，並模擬它的實際結果。

為什麼需要這一層：
    原本的回測是「進場後固定持有 N 根K棒，看收盤價方向對不對」。這等於
    把 SMC 最有價值的部分丟掉——SMC 的核心賣點是「用結構位取得不對稱的
    風險報酬比」：停損放在訊號失效的地方，目標放在對向的流動性池。

    實測固定持有的賠率只有 0.54~1.65（0050 甚至是 0.54，代表平均獲利
    只有平均虧損的一半），這正是勝率 59% 卻每筆虧 0.75% 的原因。沒有
    出場邏輯，就等於在「猜方向」而不是「做交易」。

設計原則（對照業界 SMC 慣例）：
    - 停損放在「失效區」外緣：多單放在看漲 Order Block 下緣之下，
      因為價格跌破 OB 就代表這個訊號的前提不成立了。
    - 目標放在「對向流動性」：多單的目標是上方最近、尚未被掃蕩的 EQH
      流動性池——SMC 的邏輯是價格會往停損單堆積的地方走。
    - 兩者都有 fallback（ATR 倍數 / R 倍數），避免結構不存在時無法出手。
    - 緩衝距離用 ATR 倍數而非固定點數，才能跨標的、跨波動度環境通用。

⚠️ 模擬的保守假設：
    - 同一根K棒同時觸及停損與目標時，一律當作「先觸及停損」。用日內
      OHLC 無法知道真實先後順序，樂觀假設會系統性高估績效。
    - 成交價假設就是停損/目標價，沒有模擬跳空與滑價。真實交易若跳空
      穿越停損，實際虧損會比模擬更大。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

Side = Literal["bullish", "bearish"]
Outcome = Literal["target", "stop", "timeout"]


@dataclass
class TradePlan:
    side: Side
    entry: float
    stop: float
    target: float
    stop_source: str      # order_block / swing / atr
    target_source: str    # liquidity / r_multiple

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def reward(self) -> float:
        return abs(self.target - self.entry)

    @property
    def rr(self) -> float:
        return self.reward / self.risk if self.risk > 0 else 0.0


@dataclass
class TradeResult:
    plan: TradePlan
    outcome: Outcome
    exit_price: float
    bars_held: int
    return_pct: float     # 尚未扣交易成本（成本在回測彙總層統一扣）
    r_multiple: float     # 以初始風險為單位的損益


def plan_trade(
    view,
    side: Side,
    entry: float,
    atr: float,
    *,
    stop_buffer_atr: float = 0.25,
    atr_stop_mult: float = 1.5,
    fallback_target_r: float = 2.0,
    min_rr: float = 0.5,
    max_rr: float = 10.0,
    stop_risk_scale: float = 1.0,
) -> Optional[TradePlan]:
    """
    依「當下已知」的結構決定停損與目標。

    view 只需要提供 active_order_blocks() 與 liquidity_pools，所以
    SMCAnalyzer 與回測用的 AsOfView 都可以直接傳進來——這保證了正式
    評分與回測走的是同一套出場邏輯。
    """
    if atr <= 0 or entry <= 0:
        return None

    buffer = stop_buffer_atr * atr

    # ---------------------------------------------------------------- 停損 --
    stop, stop_source = None, "atr"
    if side == "bullish":
        # 多單：停損放在「進場價下方最近的看漲 OB」下緣之下。
        # 價格跌破這個 OB，代表當初判斷的需求區失守，訊號已經失效。
        obs = [ob for ob in view.active_order_blocks("bullish") if ob.bottom < entry]
        if obs:
            stop = max(ob.bottom for ob in obs) - buffer
            stop_source = "order_block"
    else:
        obs = [ob for ob in view.active_order_blocks("bearish") if ob.top > entry]
        if obs:
            stop = min(ob.top for ob in obs) + buffer
            stop_source = "order_block"

    # OB 不存在或位置不合理時，退回純波動度停損
    if stop is None or (side == "bullish" and stop >= entry) or (side == "bearish" and stop <= entry):
        dist = atr_stop_mult * atr
        stop = entry - dist if side == "bullish" else entry + dist
        stop_source = "atr"

    # 停損縮放：覆盤顯示獲利單的 MAE 中位數只有初始風險的 16%，
    # 代表贏家很少深套，結構停損給的空間遠超過實際需要。把停損拉近
    # 可以直接改善 R:R——但也會提高被正常波動洗掉的機率，所以是個
    # 需要實測的取捨，而不是無條件的改善。
    if stop_risk_scale != 1.0:
        stop = entry - stop_risk_scale * (entry - stop)

    risk = abs(entry - stop)
    if risk <= 0:
        return None

    # ---------------------------------------------------------------- 目標 --
    # 對向流動性：多單看上方尚未被掃蕩的 EQH，空單看下方的 EQL。
    # SMC 的邏輯是價格會朝停損單堆積處移動，所以那裡才是自然的獲利了結點。
    target, target_source = None, "r_multiple"
    if side == "bullish":
        pools = [p.price for p in view.liquidity_pools
                 if p.kind == "EQH" and not p.swept and p.price > entry]
        if pools:
            target, target_source = min(pools), "liquidity"
    else:
        pools = [p.price for p in view.liquidity_pools
                 if p.kind == "EQL" and not p.swept and p.price < entry]
        if pools:
            target, target_source = max(pools), "liquidity"

    if target is None:
        target = (entry + fallback_target_r * risk if side == "bullish"
                  else entry - fallback_target_r * risk)
        target_source = "r_multiple"

    plan = TradePlan(side=side, entry=entry, stop=stop, target=target,
                     stop_source=stop_source, target_source=target_source)

    # 流動性池太近（RR 過低）或太遠（RR 大到不切實際）時，改用 R 倍數目標
    if not (min_rr <= plan.rr <= max_rr):
        target = (entry + fallback_target_r * risk if side == "bullish"
                  else entry - fallback_target_r * risk)
        plan = TradePlan(side=side, entry=entry, stop=stop, target=target,
                         stop_source=stop_source, target_source="r_multiple")

    return plan


def simulate_trade(
    future: pd.DataFrame,
    plan: TradePlan,
    max_holding_bars: int = 20,
) -> TradeResult:
    """
    用進場之後的 K 棒，逐根檢查先碰到停損還是目標。

    future: 進場那根K棒「之後」的 OHLC（不含進場棒本身）。
    """
    window = future.iloc[:max_holding_bars]
    long_ = plan.side == "bullish"

    for i, (_, bar) in enumerate(window.iterrows(), start=1):
        hit_stop = bar["Low"] <= plan.stop if long_ else bar["High"] >= plan.stop
        hit_target = bar["High"] >= plan.target if long_ else bar["Low"] <= plan.target

        # 同一根同時觸及時一律算停損先到：用 OHLC 無法還原日內先後順序，
        # 反過來假設會系統性高估績效。
        if hit_stop:
            return _result(plan, "stop", plan.stop, i)
        if hit_target:
            return _result(plan, "target", plan.target, i)

    if len(window) == 0:
        return _result(plan, "timeout", plan.entry, 0)
    return _result(plan, "timeout", float(window["Close"].iloc[-1]), len(window))


def _result(plan: TradePlan, outcome: Outcome, exit_price: float, bars: int) -> TradeResult:
    direction = 1.0 if plan.side == "bullish" else -1.0
    ret = direction * (exit_price - plan.entry) / plan.entry
    r = (direction * (exit_price - plan.entry) / plan.risk) if plan.risk > 0 else 0.0
    return TradeResult(plan=plan, outcome=outcome, exit_price=exit_price,
                       bars_held=bars, return_pct=ret, r_multiple=r)


def simulate_trailing_trade(
    future: pd.DataFrame,
    plan: TradePlan,
    atr: float,
    *,
    trail_atr_mult: float = 3.0,
    max_holding_bars: int = 250,
    breakeven_at_r: Optional[float] = None,
) -> TradeResult:
    """
    移動停損出場：不設固定目標，讓獲利奔跑，只用移動停損保護。

    為什麼需要這個模式：
        固定 2R 目標的組合回測，年化只有等權買進持有的三分之一
        （+9~12.9% vs +29.2%），即使在場時間已經拉到 99%。原因是
        **固定目標在趨勢行情裡會砍掉贏家**——買進持有吃到完整的十年
        趨勢，固定目標每次只吃 2R 就出場，還要付一次來回成本。

        虧損那邊有停損保護（回撤 11~24% vs 39%），但獲利那邊也被截斷了，
        等於「限制下檔也限制上檔」。趨勢跟隨的標準解法是把目標拿掉，
        改用移動停損：虧損仍然有限，但獲利不設上限。

    初始停損沿用 plan.stop（結構位），之後隨著價格朝有利方向移動而
    上移（多單）或下移（空單），永不反向。
    """
    window = future.iloc[:max_holding_bars]
    if len(window) == 0:
        return _result(plan, "timeout", plan.entry, 0)

    long_ = plan.side == "bullish"
    trail_dist = trail_atr_mult * atr
    stop = plan.stop
    best = plan.entry
    risk = plan.risk
    be_armed = False

    for i, (_, bar) in enumerate(window.iterrows(), start=1):
        # 先檢查停損：同一根內先觸停損的保守假設跟固定出場模式一致
        if (bar["Low"] <= stop) if long_ else (bar["High"] >= stop):
            return _result(plan, "stop", stop, i)

        # 再用這根的極值把停損往有利方向推進（只進不退）
        if long_:
            best = max(best, float(bar["High"]))
            stop = max(stop, best - trail_dist)
        else:
            best = min(best, float(bar["Low"]))
            stop = min(stop, best + trail_dist)

        # 保本停損：浮盈達到 breakeven_at_r 倍風險之後，把停損移到成本價。
        # 覆盤顯示 72% 的虧損單過程中曾經浮盈 >1%，這個機制針對的就是
        # 那批單子。⚠️ 門檻設太低會在正常回檔時把未來的大贏家洗出去——
        # 而實測前 10% 的交易貢獻了全部獲利，洗掉一個的代價極高。
        if breakeven_at_r is not None and not be_armed and risk > 0:
            gain = (best - plan.entry) if long_ else (plan.entry - best)
            if gain >= breakeven_at_r * risk:
                be_armed = True
                stop = max(stop, plan.entry) if long_ else min(stop, plan.entry)

    return _result(plan, "timeout", float(window["Close"].iloc[-1]), len(window))


# ---------------------------------------------------------------------------
# 進場計畫
#
# 原本的回測「用訊號當根的收盤價進場」，但那個分數就是用同一個收盤價算的
# ——你看到訊號時市場已經收盤，根本下不了那個價。這不是前視偏差（沒用到
# 未來資料），但它假設了一個做不到的成交價，必須修正。
#
# 實測（661 筆配對比較）：
#   隔天開盤市價        平均 +2.400%
#   隔天限價 −0.5ATR    平均 +2.509%
#   配對差異 +0.108%，95% CI [−0.131%, +0.373%] → **不顯著**
#
# 兩者報酬上沒有差別，所以選限價的理由是**執行面**：收盤後掛好單、
# 隔天不必盯盤，而且有機會用較好的價格成交（實測 247/661 筆以限價成交）。
#
# ⚠️ 唯一不能省的設計是 fallback：限價沒成交就改市價進場。實測同樣的
# 限價距離，「未成交改市價」總報酬 +1553%、「未成交放棄」只有 +1137%
# ——差 27%，因為放棄掉的那批包含了最強的走勢。
# ---------------------------------------------------------------------------
@dataclass
class EntryPlan:
    reference_close: float   # 訊號當根的收盤價
    limit_price: float       # 隔天要掛的限價
    valid_bars: int          # 限價有效幾根K棒
    fallback: str            # "market" = 未成交就市價進場；"skip" = 放棄
    atr: float

    def describe(self) -> str:
        """給通知信用的人話說明。"""
        if self.fallback == "market":
            tail = f"；{self.valid_bars} 根內未成交則以市價進場"
        else:
            tail = f"；{self.valid_bars} 根內未成交則放棄（⚠️ 實測會少賺約 27%）"
        return f"建議掛限價 {self.limit_price:.2f}（訊號收盤 {self.reference_close:.2f}）{tail}"


def plan_entry(
    signal_close: float,
    atr: float,
    *,
    offset_atr: float = 0.5,
    valid_bars: int = 1,
    fallback: str = "market",
) -> Optional[EntryPlan]:
    """依訊號當根收盤價與 ATR，算出隔天要掛的限價。"""
    if signal_close <= 0 or atr <= 0:
        return None
    return EntryPlan(
        reference_close=float(signal_close),
        limit_price=float(signal_close - offset_atr * atr),
        valid_bars=int(valid_bars),
        fallback=fallback,
        atr=float(atr),
    )


def resolve_entry(future: pd.DataFrame, plan: EntryPlan):
    """
    用進場之後的 K 棒判斷限價單何時、以什麼價格成交。

    成交規則貼近實務：開盤價就低於限價時，會以**開盤價**成交（跳空開低
    你拿到的是更好的價格），否則要當根最低價觸及限價才成交。

    回傳 (成交價, 第幾根成交的位移) 或 None（未成交且 fallback="skip"）。
    """
    window = future.iloc[: plan.valid_bars]
    for i in range(len(window)):
        o = float(window["Open"].iloc[i])
        lo = float(window["Low"].iloc[i])
        if o <= plan.limit_price:
            return o, i
        if lo <= plan.limit_price:
            return plan.limit_price, i

    if plan.fallback != "market":
        return None
    idx = plan.valid_bars
    if idx >= len(future):
        return None
    return float(future["Open"].iloc[idx]), idx
