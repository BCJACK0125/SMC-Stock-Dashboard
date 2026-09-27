# -*- coding: utf-8 -*-
"""
smc/analyzer.py
================
Smart Money Concepts (SMC) 核心指標計算模組。

輸入：pandas DataFrame，欄位需包含 Open, High, Low, Close, Volume，
      index 為 DatetimeIndex（yfinance 下載後的原始格式即可直接使用）。

輸出：SMCAnalyzer 物件內含多個 DataFrame / list，分別對應：
    - swings              : 所有 swing high / swing low 點位
    - structure_events    : 所有 BOS / CHoCH 事件
    - order_blocks        : 所有 Order Block（含是否已被緩解 mitigated）
    - fvgs                : 所有 Fair Value Gap（含是否已被回補 filled）
    - liquidity_pools     : EQH / EQL（等高 / 等低）流動性池
    - zones               : 目前波段的 premium / discount / equilibrium 區間

設計原則：
    - 全部使用 pandas / numpy 做向量化或簡單迴圈運算，不依賴任何
      未封裝的第三方 SMC 套件，方便你自行檢查、調整邏輯。
    - 所有「事件」都記錄對應的 K 棒 index（時間戳）與價格，方便後續
      直接丟給 Plotly 畫圖或做評分。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Literal
import pandas as pd
import numpy as np

from indicators import atr as _atr   # 共用同一套 ATR 實作，避免公式漂移


Side = Literal["bullish", "bearish"]


@dataclass
class SwingPoint:
    index: pd.Timestamp
    price: float
    kind: Literal["high", "low"]
    label: Optional[str] = None  # HH / HL / LH / LL，在結構分析階段補上
    confirmed_index: Optional[pd.Timestamp] = None  # 這個 swing 要等到「幾根K棒之後」才能被確認


@dataclass
class StructureEvent:
    index: pd.Timestamp          # 突破發生（收盤確認）當下的時間
    price: float                 # 突破時的收盤價
    type: Literal["BOS", "CHoCH"]
    side: Side                   # bullish = 向上突破, bearish = 向下突破
    broken_level: float          # 被突破的結構價位（前一個關鍵 swing）
    broken_index: pd.Timestamp   # 被突破的那個 swing 發生的時間


@dataclass
class OrderBlock:
    start_index: pd.Timestamp
    end_index: pd.Timestamp      # 目前為止的有效區間右邊界（同 start，僅一根K棒）
    top: float
    bottom: float
    side: Side                   # bullish OB = 支撐 / bearish OB = 壓力
    caused_structure: Optional[StructureEvent] = None
    mitigated: bool = False
    mitigated_index: Optional[pd.Timestamp] = None


@dataclass
class FairValueGap:
    start_index: pd.Timestamp    # 缺口所在三根K棒中，第一根的時間
    end_index: pd.Timestamp      # 第三根的時間
    top: float
    bottom: float
    side: Side                   # bullish FVG（價格上漲造成的缺口，未來傾向回補支撐）
    filled: bool = False
    fill_ratio: float = 0.0      # 0~1，缺口被回補的比例
    filled_index: Optional[pd.Timestamp] = None


@dataclass
class LiquidityPool:
    index_a: pd.Timestamp
    index_b: pd.Timestamp
    price: float                 # 兩個高點/低點的平均價
    kind: Literal["EQH", "EQL"]
    confirmed_index: Optional[pd.Timestamp] = None  # 兩個 swing 都被確認的時間點（較晚的那個）
    swept: bool = False
    swept_index: Optional[pd.Timestamp] = None


class SMCAnalyzer:
    """
    使用方式：
        df = yf.download("2330.TW", period="2y", interval="1d")
        analyzer = SMCAnalyzer(df)
        analyzer.run_all()

        analyzer.swings
        analyzer.structure_events
        analyzer.order_blocks
        analyzer.fvgs
        analyzer.liquidity_pools
        analyzer.current_zone   # ("premium"/"discount"/"equilibrium", top, bottom, mid)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        swing_lookback: int = 2,
        eq_tolerance_pct: float = 0.0015,   # EQH/EQL 容許誤差（0.15%）
        fvg_min_gap_pct: float = 0.0,       # 缺口最小寬度（相對價格），0 = 不過濾
        fvg_min_gap_atr: float = 0.0,       # 缺口最小寬度（相對 ATR），0 = 不過濾
    ):
        required_cols = {"Open", "High", "Low", "Close"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame 缺少必要欄位: {missing}")

        self.df = df.copy()
        self.df = self.df.sort_index()
        self.swing_lookback = swing_lookback
        self.eq_tolerance_pct = eq_tolerance_pct
        self.fvg_min_gap_pct = fvg_min_gap_pct
        self.fvg_min_gap_atr = fvg_min_gap_atr

        self.swings: List[SwingPoint] = []
        self.structure_events: List[StructureEvent] = []
        self.order_blocks: List[OrderBlock] = []
        self.fvgs: List[FairValueGap] = []
        self.liquidity_pools: List[LiquidityPool] = []
        self.current_zone: Optional[dict] = None
        self.current_trend: Optional[Side] = None

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run_all(self) -> "SMCAnalyzer":
        self._find_swings()
        self._find_structure_events()
        self._find_order_blocks()
        self._find_fvgs()
        self._find_liquidity_pools()
        self._update_premium_discount_zone()
        self._update_mitigation_status()
        self._update_liquidity_sweeps()
        return self

    # ------------------------------------------------------------------
    # 1. Swing High / Swing Low（Fractal 法）
    # ------------------------------------------------------------------
    def _find_swings(self) -> None:
        n = self.swing_lookback
        highs = self.df["High"].values
        lows = self.df["Low"].values
        idx = self.df.index

        swings: List[SwingPoint] = []
        for i in range(n, len(self.df) - n):
            window_high = highs[i - n : i + n + 1]
            window_low = lows[i - n : i + n + 1]
            confirmed_at = idx[i + n]  # 要等右邊 n 根K棒都出現才能確認這是 swing

            # 嚴格大於左右兩側鄰居才算 swing high（避免平台整段都標記）
            if highs[i] > max(np.delete(window_high, n)):
                swings.append(SwingPoint(index=idx[i], price=float(highs[i]), kind="high",
                                          confirmed_index=confirmed_at))

            if lows[i] < min(np.delete(window_low, n)):
                swings.append(SwingPoint(index=idx[i], price=float(lows[i]), kind="low",
                                          confirmed_index=confirmed_at))

        swings.sort(key=lambda s: s.confirmed_index)

        # 相鄰同種類 swing 只保留較極端的一個（避免雜訊）
        cleaned: List[SwingPoint] = []
        for s in swings:
            if cleaned and cleaned[-1].kind == s.kind:
                if s.kind == "high" and s.price >= cleaned[-1].price:
                    cleaned[-1] = s
                elif s.kind == "low" and s.price <= cleaned[-1].price:
                    cleaned[-1] = s
                else:
                    continue
            else:
                cleaned.append(s)

        # 標記 HH / HL / LH / LL
        last_high: Optional[SwingPoint] = None
        last_low: Optional[SwingPoint] = None
        for s in cleaned:
            if s.kind == "high":
                if last_high is not None:
                    s.label = "HH" if s.price > last_high.price else "LH"
                last_high = s
            else:
                if last_low is not None:
                    s.label = "HL" if s.price > last_low.price else "LL"
                last_low = s

        self.swings = cleaned

    # ------------------------------------------------------------------
    # 2. BOS（結構突破，趨勢延續）/ CHoCH（結構轉變，可能反轉）
    #
    # 邏輯：
    #   維持一個「目前趨勢」狀態 (bullish / bearish / None)，以及目前
    #   有效的「結構高點 / 結構低點」（最近一次尚未被突破的關鍵 swing）。
    #
    #   - 若目前趨勢為 bullish：
    #       收盤價突破結構高點 → BOS（延續），更新結構高點
    #       收盤價跌破結構低點 → CHoCH（反轉為 bearish），更新結構低/高點
    #   - 若目前趨勢為 bearish：反向對稱處理
    #   - 若尚無趨勢：以第一次出現的突破決定初始方向
    # ------------------------------------------------------------------
    def _find_structure_events(self) -> None:
        if not self.swings:
            return

        closes = self.df["Close"]
        events: List[StructureEvent] = []

        trend: Optional[Side] = None
        struct_high: Optional[SwingPoint] = None  # 目前有效的結構高點
        struct_low: Optional[SwingPoint] = None   # 目前有效的結構低點

        # 已經被突破過的結構位，記下它的時間戳；之後只有「更新的」swing
        # 才能接手成為結構位，同一個 swing 不會被重複突破。
        broken_high_idx: Optional[pd.Timestamp] = None
        broken_low_idx: Optional[pd.Timestamp] = None

        swing_pointer = 0

        for ts, close in closes.items():
            # 每當有新的 swing 被「確認」，就讓它接手成為目前的結構位
            # （結構位永遠是最近一個尚未被突破的 swing）。
            while swing_pointer < len(self.swings) and self.swings[swing_pointer].confirmed_index <= ts:
                s = self.swings[swing_pointer]
                swing_pointer += 1
                if s.kind == "high":
                    if broken_high_idx is None or s.index > broken_high_idx:
                        struct_high = s
                else:
                    if broken_low_idx is None or s.index > broken_low_idx:
                        struct_low = s

            # 檢查是否突破結構高點（向上）
            if struct_high is not None and close > struct_high.price and struct_high.index < ts:
                ev_type = "BOS" if trend == "bullish" else "CHoCH" if trend is not None else "BOS"
                events.append(StructureEvent(
                    index=ts, price=float(close), type=ev_type, side="bullish",
                    broken_level=struct_high.price, broken_index=struct_high.index,
                ))
                trend = "bullish"
                # 舊的結構高點已失效，清空並等下一個新的 swing high 被確認。
                # 舊版在這裡把它設回同一個 pending_high，導致之後每一根收在
                # 其上的K棒都會再報一次同樣的 BOS。
                broken_high_idx = struct_high.index
                struct_high = None
                continue

            # 檢查是否跌破結構低點（向下）
            if struct_low is not None and close < struct_low.price and struct_low.index < ts:
                ev_type = "BOS" if trend == "bearish" else "CHoCH" if trend is not None else "BOS"
                events.append(StructureEvent(
                    index=ts, price=float(close), type=ev_type, side="bearish",
                    broken_level=struct_low.price, broken_index=struct_low.index,
                ))
                trend = "bearish"
                broken_low_idx = struct_low.index
                struct_low = None
                continue

        self.structure_events = events
        self.current_trend = trend

    # ------------------------------------------------------------------
    # 3. Order Blocks（訂單塊）
    #
    #   每次發生 BOS / CHoCH（代表一段強勢的推動段）時，往回找造成這段
    #   推動的「最後一根反向K棒」：
    #       - 上漲推動 → 最後一根收黑（Close < Open）的K棒 = 看漲 OB
    #       - 下跌推動 → 最後一根收紅（Close > Open）的K棒 = 看跌 OB
    #   區間定義為該K棒的 [Low, High]（含影線，較保守／寬鬆版本）。
    # ------------------------------------------------------------------
    def _find_order_blocks(self) -> None:
        if not self.structure_events:
            return

        df = self.df
        obs: List[OrderBlock] = []

        for ev in self.structure_events:
            # OB = 造成這段推動的「最後一根反向K棒」，所以只能在「被突破的
            # swing」到「突破確認」之間這段推動腿裡面找。
            # 舊版用 df.loc[:ev.broken_index]，那是「從資料最開頭到被突破的
            # swing」，抓到的是推動段之前、毫不相干的某根K棒。
            # 排除突破棒自己（.iloc[:-1]）：OB 是推動「之前」的最後一根反向棒，
            # 突破棒即使收黑也不該被當成 OB。
            search_window = df.loc[ev.broken_index:ev.index].iloc[:-1]
            if search_window.empty:
                continue

            if ev.side == "bullish":
                bearish_candles = search_window[search_window["Close"] < search_window["Open"]]
                if bearish_candles.empty:
                    continue
                ob_ts = bearish_candles.index[-1]
                row = df.loc[ob_ts]
                obs.append(OrderBlock(
                    start_index=ob_ts, end_index=ob_ts,
                    top=float(row["High"]), bottom=float(row["Low"]),
                    side="bullish", caused_structure=ev,
                ))
            else:
                bullish_candles = search_window[search_window["Close"] > search_window["Open"]]
                if bullish_candles.empty:
                    continue
                ob_ts = bullish_candles.index[-1]
                row = df.loc[ob_ts]
                obs.append(OrderBlock(
                    start_index=ob_ts, end_index=ob_ts,
                    top=float(row["High"]), bottom=float(row["Low"]),
                    side="bearish", caused_structure=ev,
                ))

        # 去重（同一根K棒可能被多次觸發，只留第一次）
        seen = set()
        unique_obs = []
        for ob in obs:
            key = (ob.start_index, ob.side)
            if key not in seen:
                seen.add(key)
                unique_obs.append(ob)

        self.order_blocks = unique_obs

    # ------------------------------------------------------------------
    # 4. Fair Value Gap（公允價值缺口 / 失衡區）
    #
    #   三根連續K棒 (K1, K2, K3)：
    #       - 看漲 FVG：K1.High < K3.Low → 缺口 = [K1.High, K3.Low]
    #       - 看跌 FVG：K1.Low  > K3.High → 缺口 = [K3.High, K1.Low]
    # ------------------------------------------------------------------
    def _find_fvgs(self) -> None:
        df = self.df
        n = len(df)
        fvgs: List[FairValueGap] = []

        # 用 ATR 當作「這根K棒的正常波動幅度」基準，濾掉雜訊級的小缺口。
        # 不用固定百分比，是因為不同標的、不同時期的波動度差很多：實測台股
        # 的 gap/ATR 系統性高於美股，固定百分比會在兩邊行為不一致。
        atr_series = _atr(df) if self.fvg_min_gap_atr > 0 else None

        def gap_is_significant(gap: float, ref_price: float, ts: pd.Timestamp) -> bool:
            if self.fvg_min_gap_pct > 0 and gap / ref_price < self.fvg_min_gap_pct:
                return False
            if atr_series is not None:
                a = float(atr_series.loc[ts])
                # ATR 尚未暖機完成（前幾根）時不做過濾，避免把開頭全砍掉
                if a > 0 and gap / a < self.fvg_min_gap_atr:
                    return False
            return True

        for i in range(1, n - 1):
            k1 = df.iloc[i - 1]
            k3 = df.iloc[i + 1]
            k1_ts, k3_ts = df.index[i - 1], df.index[i + 1]

            if k1["High"] < k3["Low"]:
                gap = k3["Low"] - k1["High"]
                if gap_is_significant(gap, k1["High"], k3_ts):
                    fvgs.append(FairValueGap(
                        start_index=k1_ts, end_index=k3_ts,
                        top=float(k3["Low"]), bottom=float(k1["High"]),
                        side="bullish",
                    ))
            elif k1["Low"] > k3["High"]:
                gap = k1["Low"] - k3["High"]
                if gap_is_significant(gap, k1["Low"], k3_ts):
                    fvgs.append(FairValueGap(
                        start_index=k1_ts, end_index=k3_ts,
                        top=float(k1["Low"]), bottom=float(k3["High"]),
                        side="bearish",
                    ))

        self.fvgs = fvgs

    # ------------------------------------------------------------------
    # 5. 流動性池（EQH / EQL：等高 / 等低點）
    # ------------------------------------------------------------------
    def _find_liquidity_pools(self) -> None:
        highs = [s for s in self.swings if s.kind == "high"]
        lows = [s for s in self.swings if s.kind == "low"]
        pools: List[LiquidityPool] = []

        def find_equal_pairs(points: List[SwingPoint], kind: str):
            for i in range(len(points) - 1):
                for j in range(i + 1, min(i + 4, len(points))):  # 只比對鄰近幾個，避免 O(n^2) 太大
                    a, b = points[i], points[j]
                    avg = (a.price + b.price) / 2
                    if abs(a.price - b.price) / avg <= self.eq_tolerance_pct:
                        confirmed = max(a.confirmed_index, b.confirmed_index)
                        pools.append(LiquidityPool(
                            index_a=a.index, index_b=b.index, price=avg, kind=kind,
                            confirmed_index=confirmed,
                        ))

        find_equal_pairs(highs, "EQH")
        find_equal_pairs(lows, "EQL")
        self.liquidity_pools = pools

    # ------------------------------------------------------------------
    # 6. 溢價 / 折價 / 均衡區（以最近一段波段的高低點，用 Fibonacci 0.5 劃分）
    # ------------------------------------------------------------------
    def _update_premium_discount_zone(self) -> None:
        if len(self.swings) < 2:
            self.current_zone = None
            return

        recent_high = next((s for s in reversed(self.swings) if s.kind == "high"), None)
        recent_low = next((s for s in reversed(self.swings) if s.kind == "low"), None)
        if recent_high is None or recent_low is None:
            self.current_zone = None
            return

        top, bottom = max(recent_high.price, recent_low.price), min(recent_high.price, recent_low.price)
        mid = (top + bottom) / 2
        last_close = float(self.df["Close"].iloc[-1])

        if last_close > mid:
            zone_name = "premium"
        elif last_close < mid:
            zone_name = "discount"
        else:
            zone_name = "equilibrium"

        self.current_zone = {
            "top": top, "bottom": bottom, "mid": mid,
            "zone": zone_name, "last_close": last_close,
        }

    # ------------------------------------------------------------------
    # 7. 更新 OB / FVG 的緩解（mitigation）狀態
    #    用「該區塊形成之後」的所有K棒價格是否重新進入該區間來判斷。
    # ------------------------------------------------------------------
    def _update_mitigation_status(self) -> None:
        df = self.df

        for ob in self.order_blocks:
            # OB 要等「造成它的那段推動」(BOS/CHoCH) 確認之後才算成立。
            # 從 start_index 起算是錯的：OB 本來就是推動段起點那根K棒，
            # 緊接著的K棒必然還在它附近，會被誤判成一形成就被緩解。
            since = ob.caused_structure.index if ob.caused_structure else ob.start_index
            after = df[df.index > since]
            if after.empty:
                continue
            # 緩解 = 價格「重新進入」[bottom, top] 區間（兩區間重疊）。
            # 舊版用單邊條件（Low <= top / High >= bottom），對任何一根位置
            # 正常的K棒幾乎恆真，等於所有 OB 都會立刻失效。
            touched = after[(after["Low"] <= ob.top) & (after["High"] >= ob.bottom)]
            if not touched.empty:
                ob.mitigated = True
                ob.mitigated_index = touched.index[0]

        for fvg in self.fvgs:
            after = df[df.index > fvg.end_index]
            if after.empty:
                continue
            if fvg.side == "bullish":
                touched = after[after["Low"] <= fvg.top]
            else:
                touched = after[after["High"] >= fvg.bottom]
            if not touched.empty:
                fvg.filled = True
                fvg.filled_index = touched.index[0]
                fvg.fill_ratio = 1.0  # 簡化：只要碰到就視為已回補，不做部分回補比例計算

    # ------------------------------------------------------------------
    # 8. 更新流動性池的掃蕩（sweep）狀態
    #    EQH/EQL 之所以是「流動性池」，是因為停損單堆在那裡；價格穿過去
    #    把單子掃掉（sweep）之後常常反轉，這才是可交易的訊號。
    #    LiquidityPool.swept / swept_index 兩個欄位原本宣告了卻從來沒有人
    #    賦值，所以永遠是 False——這裡把它補上。
    # ------------------------------------------------------------------
    def _update_liquidity_sweeps(self) -> None:
        df = self.df
        for pool in self.liquidity_pools:
            # 兩個 swing 都被確認之後，這個池子才算「存在」
            since = pool.confirmed_index or max(pool.index_a, pool.index_b)
            after = df[df.index > since]
            if after.empty:
                continue
            if pool.kind == "EQH":
                pierced = after[after["High"] > pool.price]
            else:
                pierced = after[after["Low"] < pool.price]
            if not pierced.empty:
                pool.swept = True
                pool.swept_index = pierced.index[0]

    # ------------------------------------------------------------------
    # 便利方法：取得目前仍「未被緩解」的 OB / FVG
    # ------------------------------------------------------------------
    def active_order_blocks(self, side: Optional[Side] = None) -> List[OrderBlock]:
        obs = [ob for ob in self.order_blocks if not ob.mitigated]
        return [ob for ob in obs if ob.side == side] if side else obs

    def active_fvgs(self, side: Optional[Side] = None) -> List[FairValueGap]:
        gaps = [g for g in self.fvgs if not g.filled]
        return [g for g in gaps if g.side == side] if side else gaps

    def last_structure_event(self) -> Optional[StructureEvent]:
        return self.structure_events[-1] if self.structure_events else None
