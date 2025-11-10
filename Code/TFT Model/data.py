# ----------------------------------------------------#
#
#   File       : data.py
#   Description: TX Futures data preparation for TFT
#
# ----------------------------------------------------#

from __future__ import annotations

import datetime as dt
import warnings
from bisect import bisect_left
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

from dateutil.relativedelta import relativedelta

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder

warnings.filterwarnings("ignore")

WEEKEND_RULE_START = dt.date(2001, 1, 1)


# 將不同格式的輸入統一轉成日期物件，方便後續切片
def _coerce_date(value: Optional[object]) -> Optional[dt.date]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.datetime.strptime(value, "%Y-%m-%d").date()
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().date()
    raise TypeError(f"Unsupported date type: {type(value)}")


def _is_market_rest_day(
    day: dt.date,
    trading_history: Set[dt.date],
    history_last_day: Optional[dt.date],
    future_rest_days: Set[dt.date],
) -> bool:
    if history_last_day and day <= history_last_day:
        return day not in trading_history
    if day in future_rest_days:
        return True
    if day >= WEEKEND_RULE_START and day.weekday() >= 5:
        return True
    return False


def _days_until_next_market_rest_day(
    day: dt.date,
    trading_history: Set[dt.date],
    history_last_day: Optional[dt.date],
    future_rest_days: Set[dt.date],
    max_lookahead: int = 31,
) -> int:
    for offset in range(0, max_lookahead + 1):
        candidate = day + dt.timedelta(days=offset)
        if _is_market_rest_day(candidate, trading_history, history_last_day, future_rest_days):
            return offset
    return max_lookahead


def _third_wednesday(year: int, month: int) -> dt.date:
    first_day = dt.date(year, month, 1)
    days_until_wed = (2 - first_day.weekday()) % 7
    first_wednesday = first_day + dt.timedelta(days=days_until_wed)
    return first_wednesday + dt.timedelta(days=14)


def _days_until_next_settlement(
    day: dt.date,
    settlement_dates: Optional[List[dt.date]] = None,
) -> float:
    """
    根據結算日行事曆計算距離下一個結算日的天數。
    若沒有提供行事曆，則使用「每月第三個星期三」的推算結果。
    """
    if settlement_dates:
        idx = bisect_left(settlement_dates, day)
        if idx < len(settlement_dates):
            return float((settlement_dates[idx] - day).days)
        return np.nan
    settlement = _third_wednesday(day.year, day.month)
    if day > settlement:
        next_month = day + relativedelta(months=1)
        settlement = _third_wednesday(next_month.year, next_month.month)
    return float((settlement - day).days)


def _compute_event_countdown(day: dt.date, event_dates: List[dt.date]) -> float:
    if not event_dates:
        return np.nan
    for event_day in event_dates:
        if event_day >= day:
            return float((event_day - day).days)
    return np.nan


class TXFuturesFeatureBuilder:
    """
    Prepare TX futures data with engineered features for the TFT.

    The resulting dataframe separates:
      - time_varying_unknown_reals (observed history)
      - time_varying_known_reals (calendar / event features)
    """

    def __init__(
        self,
        start_date: Optional[object] = "2010-01-01",
        end_date: Optional[object] = None,
        data_path: Optional[Path] = None,
        exdiv_cache_path: Optional[Path] = None,
        settlement_calendar_path: Optional[Path] = None,
        export_dir: Optional[Path] = None,
        preview_rows: int = 20,
        export_full: bool = False,
    ) -> None:
        # 預設使用專案根目錄的 Data/TX.csv 做為資料來源
        project_root = Path(__file__).resolve().parents[1]
        self.data_path = Path(data_path) if data_path else project_root / "Data" / "TX.csv"
        # 台積電除息日預設會從快取檔案讀取，若不存在會嘗試下載一次
        self.exdiv_cache_path = (
            Path(exdiv_cache_path)
            if exdiv_cache_path
            else project_root / "Data" / "TSMC_ex_dividend_dates.csv"
        )
        self.settlement_calendar_path = (
            Path(settlement_calendar_path)
            if settlement_calendar_path
            else project_root / "Data" / "TX_settlement_calendar.csv"
        )
        self.market_holiday_path = project_root / "Data" / "TX_market_holidays.csv"
        base_export_dir = (
            Path(export_dir)
            if export_dir
            else Path(__file__).resolve().parent / "outputs"
        )
        if not base_export_dir.is_absolute():
            base_export_dir = Path(__file__).resolve().parent / base_export_dir
        self.export_dir = base_export_dir
        self.preview_rows = preview_rows
        self.export_full = export_full

        # 起訖日期允許使用字串或 datetime 輸入
        self.start_date = _coerce_date(start_date) or dt.date(2010, 1, 1)
        self.end_date = _coerce_date(end_date)

        self.market_rest_days = self._load_market_holidays()
        self.market_holiday_set: Set[dt.date] = set(self.market_rest_days)

        # 台積電除息日名單僅在初始化時讀一次，後續直接使用
        self.tsmc_exdiv_dates = self._load_tsmc_exdiv_dates()
        self.settlement_dates = self._load_settlement_dates()
        self.settlement_set: Set[dt.date] = set(self.settlement_dates)

        # time_varying_unknown_reals: 模型訓練時只能看到歷史的數值
        self.observed_reals = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "return_close_pct",
        ]
        # time_varying_known_reals: 透過日曆或事件邏輯可事先知道的特徵
        self.known_future_reals = [
            "day_of_week",
            "day_of_month",
            "month",
            "week_of_year",
            "days_until_holiday",
            "is_taiwan_holiday",
            "calendar_days_until_settlement",
            "days_until_settlement",
            "is_settlement_day",
            "calendar_days_until_tsmc_exdiv",
            "days_until_tsmc_exdiv",
            "is_tsmc_exdiv",
        ]
        self.dataframe: Optional[pd.DataFrame] = None
        self.future_dataframe: Optional[pd.DataFrame] = None
        self.future_horizon: int = 0
        self.history_last_trading_day: Optional[dt.date] = None
        self.history_trading_day_set: Set[dt.date] = set()

    def _load_raw(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Cannot find TX data file: {self.data_path}")
        df = pd.read_csv(self.data_path)
        df.columns = [c.lower() for c in df.columns]
        if "dates" not in df.columns:
            raise KeyError("Expected a 'dates' column in TX.csv.")
        df["date"] = pd.to_datetime(df["dates"])
        df = df.sort_values("date").reset_index(drop=True)

        if "ticker" not in df.columns:
            df["ticker"] = "TX"
        df["ticker"] = df["ticker"].fillna("TX").astype(str).str.upper()
        df.rename(columns={"ticker": "symbol"}, inplace=True)

        numeric_cols = ["open", "high", "low", "close", "volume"]
        missing = [col for col in numeric_cols if col not in df.columns]
        if missing:
            raise KeyError(f"Columns missing from TX.csv: {missing}")
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=numeric_cols).reset_index(drop=True)
        return df

    def _filter_by_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.start_date:
            df = df[df["date"] >= pd.Timestamp(self.start_date)]
        if self.end_date:
            df = df[df["date"] <= pd.Timestamp(self.end_date)]
        return df.reset_index(drop=True)

    def _add_temporal_columns(
        self,
        df: pd.DataFrame,
        base_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        origin = pd.Timestamp(base_date) if base_date is not None else df["date"].min()
        df["time_idx"] = (df["date"] - origin).dt.days.astype(int)
        df["day_of_week"] = df["date"].dt.dayofweek.astype(float)
        df["day_of_month"] = df["date"].dt.day.astype(float)
        df["month"] = df["date"].dt.month.astype(float)
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int).astype(float)
        df["year"] = df["date"].dt.year
        return df

    def _add_return_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["return_close_pct"] = df["close"].pct_change().fillna(0.0)
        return df

    def _add_event_features(
        self,
        df: pd.DataFrame,
        trading_days: Optional[List[dt.date]] = None,
        history_last_day: Optional[dt.date] = None,
        historical_trading_set: Optional[Set[dt.date]] = None,
    ) -> pd.DataFrame:
        daily_dates = df["date"].dt.date
        if trading_days is None:
            trading_days = list(daily_dates)
        trading_days_sorted = sorted(set(trading_days))
        if trading_days_sorted:
            inferred_last_day = trading_days_sorted[-1]
            if history_last_day is None:
                history_last_day = inferred_last_day
            future_events = {
                evt
                for evt in (self.tsmc_exdiv_dates + self.settlement_dates)
                if evt > inferred_last_day
            }
            if future_events:
                max_event = max(future_events)
                day = inferred_last_day
                extra_days: List[dt.date] = []
                while day < max_event:
                    day += dt.timedelta(days=1)
                    if self._is_future_rest_day(day):
                        continue
                    extra_days.append(day)
                if extra_days:
                    # Extend trading index to cover upcoming event dates (settlement / ex-dividend)
                    trading_days_sorted.extend(extra_days)
                    trading_days_sorted = sorted(set(trading_days_sorted))
        trading_pos = {day: idx for idx, day in enumerate(trading_days_sorted)}

        trading_history_set: Set[dt.date]
        if historical_trading_set is not None:
            trading_history_set = set(historical_trading_set)
        else:
            trading_history_set = {
                day for day in trading_days_sorted if history_last_day is None or day <= history_last_day
            }
        future_rest_days = self.market_holiday_set

        # 倒數距離下一個台灣假日或周末還有幾天
        df["days_until_holiday"] = [
            float(
                _days_until_next_market_rest_day(
                    day,
                    trading_history_set,
                    history_last_day,
                    future_rest_days,
                )
            )
            for day in daily_dates
        ]
        df["is_taiwan_holiday"] = [
            int(
                _is_market_rest_day(
                    day,
                    trading_history_set,
                    history_last_day,
                    future_rest_days,
                )
            )
            for day in daily_dates
        ]

        # 台指期結算日倒數與指示（優先使用外部行事曆）
        calendar_settlement_countdowns: List[float] = []
        for day in daily_dates:
            val = _days_until_next_settlement(day, self.settlement_dates)
            if pd.isna(val):
                val = _days_until_next_settlement(day, None)
            calendar_settlement_countdowns.append(float(val))
        df["calendar_days_until_settlement"] = calendar_settlement_countdowns
        df["is_settlement_day"] = [1 if day in self.settlement_set else 0 for day in daily_dates]
        settlement_positions = [
            trading_pos[evt] for evt in self.settlement_dates if evt in trading_pos
        ]
        settlement_positions.sort()
        settlement_trading_countdowns: List[float] = []
        for day in daily_dates:
            current_pos = trading_pos.get(day)
            if current_pos is None:
                settlement_trading_countdowns.append(np.nan)
                continue
            idx = bisect_left(settlement_positions, current_pos)
            if idx < len(settlement_positions):
                settlement_trading_countdowns.append(float(settlement_positions[idx] - current_pos))
            else:
                settlement_trading_countdowns.append(np.nan)
        settlement_trading_series = pd.Series(settlement_trading_countdowns, dtype="float")
        if settlement_trading_series.notna().any():
            settlement_trading_series = settlement_trading_series.fillna(
                float(settlement_trading_series.dropna().max())
            )
        else:
            settlement_trading_series = settlement_trading_series.fillna(
                float(len(trading_days_sorted))
            )
        df["days_until_settlement"] = settlement_trading_series

        # 台積電除息日倒數與指示
        exdiv_countdowns = [
            _compute_event_countdown(day, self.tsmc_exdiv_dates) for day in daily_dates
        ]
        series = pd.Series(exdiv_countdowns, dtype="float")
        if series.isna().all():
            warnings.warn(
                "TSMC ex-dividend countdowns are NaN. Provide additional future ex-dividend dates to improve this feature."
            )
            fill_value = float(len(trading_days_sorted))
        else:
            fill_value = float(series.dropna().max())
        df["calendar_days_until_tsmc_exdiv"] = series.fillna(fill_value)

        event_positions = [
            trading_pos[evt] for evt in self.tsmc_exdiv_dates if evt in trading_pos
        ]
        event_positions.sort()
        trading_countdowns: List[float] = []
        for day in daily_dates:
            current_pos = trading_pos.get(day)
            if current_pos is None:
                trading_countdowns.append(np.nan)
                continue
            idx = bisect_left(event_positions, current_pos)
            if idx < len(event_positions):
                trading_countdowns.append(float(event_positions[idx] - current_pos))
            else:
                trading_countdowns.append(np.nan)
        trading_series = pd.Series(trading_countdowns, dtype="float")
        if trading_series.notna().any():
            trading_series = trading_series.fillna(float(trading_series.dropna().max()))
        else:
            trading_series = trading_series.fillna(float(len(trading_days_sorted)))
        df["days_until_tsmc_exdiv"] = trading_series
        df["is_tsmc_exdiv"] = trading_series.fillna(-1).eq(0).astype(int)

        return df

    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        # 目標欄位設定為隔日開盤與收盤價
        df["target_open_t1"] = df["open"].shift(-1)
        df["target_close_t1"] = df["close"].shift(-1)
        df = df.dropna(subset=["target_open_t1", "target_close_t1"])
        return df.reset_index(drop=True)

    def _load_market_holidays(self) -> List[dt.date]:
        records: Dict[dt.date, Dict[str, str]] = {}
        if self.market_holiday_path.exists():
            try:
                df = pd.read_csv(self.market_holiday_path, dtype={"description": str, "source": str})
                dates = pd.to_datetime(df["date"], errors="coerce").dt.date
                for entry_date, row in zip(dates, df.to_dict("records")):
                    if not isinstance(entry_date, dt.date):
                        continue
                    desc = (row.get("description") or "").strip()
                    source = (row.get("source") or "manual").strip() or "manual"
                    records[entry_date] = {"description": desc, "source": source}
                if records:
                    print(
                        f"[data] Loaded {len(records)} market holidays from {self.market_holiday_path}"
                    )
            except Exception as exc:
                warnings.warn(f"讀取市場休市日清單失敗：{exc}")

        today = dt.date.today()
        start_year = max(2000, min(self.start_date.year, today.year))
        end_basis = self.end_date or (today + dt.timedelta(days=365 * 2))
        end_year = max(end_basis.year, today.year)
        existing_years = {day.year for day in records}
        missing_years = [year for year in range(start_year, end_year + 1) if year not in existing_years]

        fetched_any = False
        for year in missing_years:
            fetched = self._fetch_twse_market_holidays(year)
            if not fetched:
                continue
            fetched_any = True
            for day, info in fetched.items():
                if day in records and records[day].get("source", "").lower() != "manual":
                    records[day] = info
                elif day not in records:
                    records[day] = info

        if fetched_any:
            try:
                rows = []
                for day in sorted(records):
                    info = records[day]
                    rows.append(
                        {
                            "date": day.isoformat(),
                            "description": info.get("description", ""),
                            "source": info.get("source", "") or "manual",
                        }
                    )
                df = pd.DataFrame(rows)
                self.market_holiday_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(self.market_holiday_path, index=False)
                print(
                    f"[data] Market holiday calendar updated with {len(rows)} entries -> {self.market_holiday_path}"
                )
            except Exception as exc:
                warnings.warn(f"寫入市場休市日清單失敗：{exc}")

        if not records:
            warnings.warn("沒有可用的市場休市日資料，僅使用周末與國定假日。")
            return []
        return sorted(records)

    def _is_future_rest_day(self, day: dt.date) -> bool:
        return _is_market_rest_day(
            day,
            self.history_trading_day_set,
            self.history_last_trading_day,
            self.market_holiday_set,
        )

    def _fetch_twse_market_holidays(self, year: int) -> Dict[dt.date, Dict[str, str]]:
        try:
            import requests
        except ImportError:
            warnings.warn("requests 未安裝，無法自動下載市場休市日。")
            return {}

        url = "https://www.twse.com.tw/holidaySchedule/holidaySchedule"
        params = {"response": "json", "queryYear": year}
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            warnings.warn(f"下載 {year} 年市場休市日資料失敗：{exc}")
            return {}

        entries: Dict[dt.date, Dict[str, str]] = {}
        for raw in payload.get("data", []):
            if not raw or len(raw) < 3:
                continue
            date_str, name, note = raw[0], raw[1], raw[2]
            if not date_str:
                continue
            merged_text = f"{name or ''}{note or ''}"
            if any(keyword in merged_text for keyword in ["開始交易", "最後交易", "恢復交易", "開始辦理交易"]):
                continue
            try:
                holiday_date = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            description_parts = []
            if name:
                description_parts.append(str(name).strip())
            if note:
                note_clean = str(note).strip()
                if note_clean and note_clean not in description_parts:
                    description_parts.append(note_clean)
            description = "；".join(description_parts)
            entries[holiday_date] = {"description": description, "source": "TWSE"}
        if entries:
            print(f"[data] Downloaded {len(entries)} TWSE market holidays for {year}.")
        return entries

    def _load_tsmc_exdiv_dates(self) -> List[dt.date]:
        manual_dates: Set[dt.date] = set()
        manual_loaded = False
        if self.exdiv_cache_path.exists():
            try:
                df = pd.read_csv(self.exdiv_cache_path, parse_dates=["date"])
                manual_dates = {d.date() for d in df["date"].dropna()}
                manual_loaded = True
                print(
                    f"[data] Loaded {len(manual_dates)} TSMC ex-dividend dates from {self.exdiv_cache_path}"
                )
            except Exception as exc:
                warnings.warn(f"讀取 TSMC 除息日快取失敗：{exc}")
        yf_dates = set(self._fetch_tsmc_exdiv_dates(include_future=False))
        today = dt.date.today()
        if manual_loaded and yf_dates:
            history_yf = {d for d in yf_dates if d <= today}
            if history_yf:
                yf_min = min(history_yf)
                history_manual = {d for d in manual_dates if yf_min <= d <= today}
                missing_in_manual = {d for d in history_yf if d not in history_manual}
                missing_in_yf = {d for d in history_manual if d not in history_yf}
                if missing_in_manual or missing_in_yf:
                    raise ValueError(
                        "手動除息日清單與 yfinance 歷史資料對不上，請檢查:\\n"
                        f"  僅出現在 yfinance 的日期: {sorted(missing_in_manual)}\\n"
                        f"  僅出現在本地檔案的日期: {sorted(missing_in_yf)}"
                    )
        combined = sorted(manual_dates | yf_dates)
        if not combined:
            warnings.warn(
                "無法取得任何除息日資料，將回傳空列表，除息倒數將為 NaN。"
            )
            return []
        # 若 yfinance 提供了新增的歷史資料，也一併寫回快取 (manual 空或仍補齊 union)
        if yf_dates - manual_dates or not manual_loaded:
            try:
                df = pd.DataFrame({"date": [d.isoformat() for d in combined]})
                self.exdiv_cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(self.exdiv_cache_path, index=False)
                print(
                    f"[data] Updated ex-dividend cache with {len(combined)} entries -> {self.exdiv_cache_path}"
                )
            except Exception as exc:
                warnings.warn(f"更新除息日快取失敗：{exc}")
        upcoming = [d for d in combined if d >= today]
        if not upcoming:
            warnings.warn("快取中沒有未來除息日，請補充最新官方資料。")
        return combined

    def _load_settlement_dates(self) -> List[dt.date]:
        """
        讀取台指期正式結算日清單；若檔案不存在則按第三個星期三規則產生。
        載回的日期會被排序並去重。
        """
        if self.settlement_calendar_path.exists():
            try:
                df = pd.read_csv(self.settlement_calendar_path)
                # 偵測欄位名稱：支援 settlement_date / date / 日期 等常見命名
                for candidate in ["settlement_date", "date", "日期"]:
                    if candidate in df.columns:
                        col = candidate
                        break
                else:
                    raise KeyError("找不到結算日欄位（settlement_date / date）。")
                dates = pd.to_datetime(df[col], errors="coerce").dropna().dt.date
                cleaned = sorted(set(dates))
                if cleaned:
                    print(
                        f"[data] Loaded {len(cleaned)} TX settlement dates from {self.settlement_calendar_path}"
                    )
                    return cleaned
                warnings.warn(
                    f"Settlement calendar {self.settlement_calendar_path} 無有效資料，改用預設規則。"
                )
            except Exception as exc:
                warnings.warn(f"讀取結算日清單失敗（{exc}），改用預設規則。")
        # fallback: 依據第三個星期三產生
        start_year = (self.start_date or dt.date(2000, 1, 1)).year - 1
        end_basis = self.end_date or dt.date.today() + dt.timedelta(days=365 * 3)
        end_year = end_basis.year + 2
        generated: List[dt.date] = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                generated.append(_third_wednesday(year, month))
        generated = sorted(set(generated))
        print(
            "[data] Generated "
            f"{len(generated)} TX settlement dates by rule-of-thumb (3rd Wednesday).\n"
            f"    -> 建議以官方列表覆蓋 {self.settlement_calendar_path} 後重新載入。"
        )
        try:
            df = pd.DataFrame({"settlement_date": [d.isoformat() for d in generated]})
            self.settlement_calendar_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.settlement_calendar_path, index=False)
            print(f"[data] Baseline settlement calendar cached to {self.settlement_calendar_path}")
        except Exception as exc:
            warnings.warn(f"無法寫入預設結算列表：{exc}")
        return generated

    def _fetch_tsmc_exdiv_dates(self, include_future: bool = False) -> List[dt.date]:
        try:
            import yfinance as yf
        except ImportError:
            warnings.warn(
                "yfinance is not installed; cannot auto-download TSMC ex-dividend dates."
            )
            return []
        try:
            ticker = yf.Ticker("2330.TW")
            dividends = ticker.dividends
            if dividends.empty:
                warnings.warn("yfinance returned an empty dividend series for 2330.TW.")
                return []
            idx = dividends.index
            if idx.tz is None:
                idx = idx.tz_localize("Asia/Taipei")
            else:
                idx = idx.tz_convert("Asia/Taipei")
            today = dt.date.today()
            dates = []
            for ts in idx:
                d = ts.date()
                if include_future or d <= today:
                    dates.append(d)
            dates = sorted(set(dates))
            print(
                f"[data] Downloaded {len(dates)} TSMC ex-dividend dates via yfinance ({'all' if include_future else 'history-only'})"
            )
            return dates
        except Exception as exc:
            warnings.warn(f"Failed to download TSMC dividend data: {exc}")
            return []

    def prepare_dataframe(self) -> pd.DataFrame:
        df = self._load_raw()
        df = self._filter_by_dates(df)
        df = self._add_temporal_columns(df)
        trading_days = df["date"].dt.date.tolist()
        df = self._add_event_features(df, trading_days=trading_days)
        df = self._add_return_columns(df)
        df = self._add_targets(df)
        df = df.sort_values("time_idx").reset_index(drop=True)

        columns_order = [
            "date",
            "symbol",
            "time_idx",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "return_close_pct",
            "day_of_week",
            "day_of_month",
            "month",
            "week_of_year",
            "days_until_holiday",
            "is_taiwan_holiday",
            "calendar_days_until_settlement",
            "days_until_settlement",
            "is_settlement_day",
            "calendar_days_until_tsmc_exdiv",
            "days_until_tsmc_exdiv",
            "is_tsmc_exdiv",
            "target_open_t1",
            "target_close_t1",
        ]
        columns_order = [col for col in columns_order if col in df.columns]
        self.dataframe = df[columns_order]
        if not self.dataframe.empty:
            self.history_last_trading_day = self.dataframe["date"].dt.date.max()
            self.history_trading_day_set = set(self.dataframe["date"].dt.date)
        else:
            self.history_last_trading_day = None
            self.history_trading_day_set = set()
        return self.dataframe

    def print_summary(self, rows: int = 5) -> None:
        if self.dataframe is None:
            raise ValueError("Dataframe not prepared. Call prepare_dataframe() first.")
        df = self.dataframe
        print(
            f"[data] TX futures feature set shape: {df.shape}, "
            f"date span: {df['date'].min().date()} -> {df['date'].max().date()}"
        )
        print("[data] Observed (unknown future) reals:", self.observed_reals)
        print("[data] Known future reals:", self.known_future_reals)
        preview_cols = [
            "date",
            "open",
            "close",
            "days_until_holiday",
            "is_taiwan_holiday",
            "calendar_days_until_settlement",
            "days_until_settlement",
            "is_settlement_day",
            "calendar_days_until_tsmc_exdiv",
            "days_until_tsmc_exdiv",
            "is_tsmc_exdiv",
            "target_open_t1",
            "target_close_t1",
        ]
        preview_cols = [c for c in preview_cols if c in df.columns]
        print(df[preview_cols].head(rows))

    def prepare_future_dataframe(
        self,
        horizon: int,
        skip_rest_days: bool = True,
    ) -> pd.DataFrame:
        """
        建立未來 horizon 個交易日的既知特徵（不含歷史觀測值），提供推論階段使用。
        """
        if self.dataframe is None:
            raise ValueError("Dataframe not prepared. Call prepare_dataframe() first.")
        if horizon <= 0:
            self.future_dataframe = None
            self.future_horizon = 0
            return pd.DataFrame()

        base_date = self.dataframe["date"].min()
        last_date = self.dataframe["date"].max().date()
        symbol = str(self.dataframe["symbol"].iloc[0])

        rows = []
        day = last_date
        added = 0
        while added < horizon:
            day += dt.timedelta(days=1)
            if skip_rest_days and self._is_future_rest_day(day):
                continue
            rows.append({"date": pd.Timestamp(day), "symbol": symbol})
            added += 1

        future_df = pd.DataFrame(rows)
        future_df = self._add_temporal_columns(future_df, base_date=base_date)
        combined_trading_days = self.dataframe["date"].dt.date.tolist() + future_df["date"].dt.date.tolist()
        future_df = self._add_event_features(
            future_df,
            trading_days=combined_trading_days,
            history_last_day=self.history_last_trading_day,
            historical_trading_set=self.history_trading_day_set,
        )

        # 未來視角下，觀測特徵與目標值仍未知，先以 NaN 佔位
        for col in self.observed_reals:
            future_df[col] = np.nan
        future_df["target_open_t1"] = np.nan
        future_df["target_close_t1"] = np.nan

        # 依據訓練資料欄位順序整理，缺少的欄位補 NaN
        for col in self.dataframe.columns:
            if col not in future_df.columns:
                future_df[col] = np.nan
        future_df = future_df[self.dataframe.columns]

        self.future_dataframe = future_df
        self.future_horizon = len(future_df)
        print(
            f"[data] Prepared known future features for {self.future_horizon} trading days "
            f"(from {future_df['date'].min().date()} to {future_df['date'].max().date()})."
        )
        return future_df

    def export(self) -> None:
        if self.dataframe is None:
            raise ValueError("Dataframe not prepared. Call prepare_dataframe() first.")
        self.export_dir.mkdir(parents=True, exist_ok=True)
        preview_path = self.export_dir / "tx_feature_preview.csv"
        self.dataframe.head(self.preview_rows).to_csv(preview_path, index=False)
        print(f"[data] Preview export saved to {preview_path}")
        if self.export_full:
            full_path = self.export_dir / "tx_features.csv"
            try:
                self.dataframe.to_csv(full_path, index=False)
                print(f"[data] Full dataset export saved to {full_path}")
            except PermissionError as exc:
                warnings.warn(f"無法寫入完整特徵檔案 {full_path}: {exc}")
                fallback = self.export_dir / f"tx_features_{dt.datetime.now():%Y%m%d%H%M%S}.csv"
                try:
                    self.dataframe.to_csv(fallback, index=False)
                    print(f"[data] Full dataset fallback saved to {fallback}")
                except Exception as fallback_exc:
                    warnings.warn(f"完整檔案備援寫入仍失敗：{fallback_exc}")
        if self.future_dataframe is not None and not self.future_dataframe.empty:
            future_path = (
                self.export_dir / f"tx_future_known_h{max(self.future_horizon, 0)}.csv"
            )
            try:
                self.future_dataframe.to_csv(future_path, index=False)
                print(
                    f"[data] Future known-feature export (h={self.future_horizon}) saved to {future_path}"
                )
            except PermissionError as exc:
                warnings.warn(f"無法寫入未來特徵檔案 {future_path}: {exc}")
                fallback = self.export_dir / (
                    f"tx_future_known_h{max(self.future_horizon, 0)}_{dt.datetime.now():%Y%m%d%H%M%S}.csv"
                )
                try:
                    self.future_dataframe.to_csv(fallback, index=False)
                    print(
                        f"[data] Future known-feature fallback (h={self.future_horizon}) saved to {fallback}"
                    )
                except Exception as fallback_exc:
                    warnings.warn(f"未來特徵備援寫入仍失敗：{fallback_exc}")

    def build_time_series_datasets(
        self,
        target: str = "target_close_t1",
        max_prediction_length: int = 5,
        max_encoder_length: int = 120,
        min_encoder_length: Optional[int] = None,
    ) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        if self.dataframe is None:
            raise ValueError("Dataframe not prepared. Call prepare_dataframe() first.")
        df = self.dataframe.copy()
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' not found in dataframe.")

        min_encoder_length = min_encoder_length or max_encoder_length // 2
        training_cutoff = int(df["time_idx"].max() - max_prediction_length)

        unknown_reals = [col for col in self.observed_reals if col != target]
        known_reals = [col for col in self.known_future_reals if col in df.columns]

        print(
            f"[TFT] N={len(df)}, encoder={min_encoder_length}-{max_encoder_length}, "
            f"prediction={max_prediction_length}, training_cutoff={training_cutoff}"
        )

        training = TimeSeriesDataSet(
            df.loc[lambda x: x["time_idx"] <= training_cutoff],
            time_idx="time_idx",
            target=target,
            group_ids=["symbol"],
            time_varying_unknown_reals=unknown_reals,
            time_varying_known_reals=known_reals,
            static_categoricals=[],
            static_reals=[],
            min_encoder_length=min_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            categorical_encoders={"symbol": NaNLabelEncoder().fit(df["symbol"])},
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )

        validation = TimeSeriesDataSet.from_dataset(
            training,
            df,
            predict=True,
            stop_randomization=True,
        )
        return training, validation


def load_data(
    start_date: Optional[object] = "2010-01-01",
    end_date: Optional[object] = None,
    target: str = "target_close_t1",
    prediction_length: int = 5,
    encoder_length: int = 120,
    export_preview: bool = True,
    export_full: bool = True,
    preview_rows: int = 20,
    future_horizon: int = 0,
    skip_future_rest_days: bool = True,
    return_builder: bool = False,
):
    # 封裝流程：建立特徵、輸出預覽，再回傳 TFT 所需的 train/val dataset 與原始資料框
    builder = TXFuturesFeatureBuilder(
        start_date=start_date,
        end_date=end_date,
        preview_rows=preview_rows,
        export_full=export_full,
    )
    df = builder.prepare_dataframe()
    builder.print_summary(rows=min(5, len(df)))
    future_df = None
    if future_horizon > 0:
        future_df = builder.prepare_future_dataframe(
            horizon=future_horizon,
            skip_rest_days=skip_future_rest_days,
        )
    if export_preview:
        builder.export()
    training, validation = builder.build_time_series_datasets(
        target=target,
        max_prediction_length=prediction_length,
        max_encoder_length=encoder_length,
    )
    if future_df is not None:
        df.attrs["future_known_inputs"] = future_df
    if return_builder:
        return training, validation, df, builder
    return training, validation, df


if __name__ == "__main__":
    builder = TXFuturesFeatureBuilder(start_date="1998-07-21", export_full=True)
    df = builder.prepare_dataframe()
    builder.print_summary()
    builder.prepare_future_dataframe(horizon=30)
    builder.export()
    train, val = builder.build_time_series_datasets()
    print("[data] Training/validation datasets prepared.")

# ----------------------------------------------------#
# Legacy implementation retained for reference (commented out)
# ----------------------------------------------------#
# ﻿# ---------------------------------------------------#
# #
# #   File       : data.py
# #   Description: Data loading and preprocessing for TFT
# #                (extracted from PytorchForecasting.py)
# #
# # ----------------------------------------------------#
#
# import datetime
# import warnings
# from pathlib import Path
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# from pytorch_forecasting import TimeSeriesDataSet
# from pytorch_forecasting.data.encoders import NaNLabelEncoder
#
#
# warnings.filterwarnings("ignore")
#
#
# class FTSEDataSet:
#     """
#     FTSE Dataset
#
#     Extracts the data from the CSV file
#     Runs through data loaders
#     Null values are removed
#     Dataset is split into training, validation and testing datasets
#     Converted into an appropriate format for the TFT
#
#     """
#
#     def __init__(self, start=datetime.datetime(2010, 1, 1), stop=datetime.datetime.now()):
#         root = Path(__file__).resolve().parents[1]
#         self.stocks_file_name = root / "Data" / "NEAFTSE2010-21.csv"
#
#     def load(self, binary: bool = True, visualize: bool = True, save_dir: str = "outputs"):
#         # 讀 CSV（注意 dayfirst）
#         df0 = (
#             pd.read_csv(self.stocks_file_name, index_col=0, parse_dates=True, dayfirst=True)
#             .sort_index()
#         )
#         print("[data] raw shape:", df0.shape)
#         print("[data] raw df head:\n", df0.head(3))
#
#         df0.dropna(axis=1, how="all", inplace=True)
#         df0.dropna(axis=0, how="all", inplace=True)
#         na_pct = (df0.isnull().sum() / len(df0.index)).sort_values(ascending=False)
#         print("[data] top-NA columns (%):\n", (na_pct * 100).head(10))
#         print("Dropping columns due to nans > 50%:",
#               df0.loc[:, list((100 * (df0.isnull().sum() / len(df0.index)) > 50))].columns)
#
#         cols_to_drop = df0.columns[df0.isnull().mean() > 0.5]
#         print("Dropping columns due to nans > 50%:", list(cols_to_drop))
#         df0 = df0.drop(columns=cols_to_drop)
#         df0 = df0.ffill().bfill()
#
#         print("Any columns still contain nans:", df0.isnull().values.any())
#         print("[data] cleaned df head:\n", df0.head(3))
#
#         df_returns = pd.DataFrame()
#
#         for name in df0.columns:
#             df_returns[name] = np.log(df0[name]).diff()
#         print("[data] returns df head:\n", df_returns.head(3))
#
#         # split into train and test
#         df_returns.dropna(axis=0, how="any", inplace=True)
#         if binary and "FTSE" in df_returns.columns:
#             df_returns.FTSE = [1 if ftse > 0 else 0 for ftse in df_returns.FTSE]
#         self.df_returns = df_returns
#
#         # --- simple visualizations ---
#         if visualize:
#             out_dir = (Path(__file__).resolve().parent / save_dir)
#             out_dir.mkdir(parents=True, exist_ok=True)
#
#             # 1) Price lines (if available)
#             try:
#                 cols = [c for c in ["Open", "High", "Low", "Close"] if c in df0.columns]
#                 if cols:
#                     plt.figure(figsize=(8, 3))
#                     df0[cols].plot(ax=plt.gca())
#                     plt.title("Raw prices (cleaned)")
#                     plt.tight_layout()
#                     p = out_dir / "raw_prices.png"
#                     plt.savefig(p, dpi=130, bbox_inches="tight")
#                     print("saved:", p)
#                     plt.close()
#             except Exception as e:
#                 print("[viz warn] raw price plot:", e)
#
#             # 2) Returns histograms
#             try:
#                 cols = [c for c in ["Open", "High", "Low", "Close"] if c in df_returns.columns]
#                 if cols:
#                     df_returns[cols].plot(kind="hist", bins=50, alpha=0.6, subplots=True, layout=(2, 2), figsize=(8, 4))
#                     plt.suptitle("Log-returns histograms")
#                     plt.tight_layout()
#                     p = out_dir / "returns_hist.png"
#                     plt.savefig(p, dpi=130, bbox_inches="tight")
#                     print("saved:", p)
#                     plt.close()
#             except Exception as e:
#                 print("[viz warn] returns hist:", e)
#
#             # 3) Correlation heatmap (returns)
#             try:
#                 corr = df_returns.corr(numeric_only=True)
#                 plt.figure(figsize=(5, 4))
#                 im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
#                 plt.colorbar(im, fraction=0.046, pad=0.04)
#                 plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
#                 plt.yticks(range(len(corr.index)), corr.index)
#                 plt.title("Returns correlation")
#                 plt.tight_layout()
#                 p = out_dir / "returns_corr.png"
#                 plt.savefig(p, dpi=130, bbox_inches="tight")
#                 print("saved:", p)
#                 plt.close()
#             except Exception as e:
#                 print("[viz warn] corr heatmap:", e)
#
#         return df_returns
#
#
# def load_data(visualize: bool = True):
#     """
#     Load data using the FTSEDataSet class
#     Set prediction and encoder lengths
#     Set up training/validation TimeSeriesDataSet
#
#     Returns
#     -------
#     training: TimeSeriesDataSet
#     validation: TimeSeriesDataSet
#     """
#
#     dataset = FTSEDataSet()
#     print("Dataset", dataset)
#     ftse_df = dataset.load(binary=False, visualize=visualize)
#     print(dataset)
#     time_index = "Date"
#     target = "Open"
#
#     # features: start with all, then remove target
#     features = ftse_df.columns.tolist()
#     print("Features", features)
#     if target in features:
#         features.remove(target)
#
#     # convert time index (days since min)
#     ftse_df[time_index] = pd.to_datetime(ftse_df.index)
#     min_date = ftse_df[time_index].min()
#     ftse_df[time_index] = (ftse_df[time_index] - min_date).dt.days.astype(int)
#
#     # group id
#     ftse_df["Open_Prediction"] = "Open"
#     ftse_df = ftse_df.sort_values(time_index)
#
#     # window lengths
#     N = len(ftse_df)
#     max_prediction_length = max(7, min(60, N // 20))
#     max_encoder_length = max(30, min(256, N // 4))
#     min_encoder_length = max_encoder_length // 2
#
#     training_cutoff = int(ftse_df[time_index].max() - max_prediction_length)
#
#     print(
#         f"[TFT] N={N}, max_encoder_length={max_encoder_length}, "
#         f"min_encoder_length={min_encoder_length}, "
#         f"max_prediction_length={max_prediction_length}, "
#         f"training_cutoff={training_cutoff}"
#     )
#     print("time_idx", time_index)
#     print(
#         "train slice preview:",
#         ftse_df.loc[ftse_df[time_index] <= training_cutoff].head(3),
#         sep="\n",
#     )
#
#     training = TimeSeriesDataSet(
#         ftse_df[lambda x: x[time_index] <= training_cutoff],
#         time_idx=time_index,
#         target=target,
#         categorical_encoders={"Open_Prediction": NaNLabelEncoder().fit(ftse_df.Open_Prediction)},
#         group_ids=["Open_Prediction"],
#         min_encoder_length=min_encoder_length,
#         max_encoder_length=max_encoder_length,
#         min_prediction_length=1,
#         max_prediction_length=max_prediction_length,
#         time_varying_unknown_reals=features,  # e.g. ['High','Low','Close']
#         add_relative_time_idx=True,
#         add_target_scales=True,
#         add_encoder_length=True,
#         allow_missing_timesteps=True,
#     )
#
#     print(training.get_parameters())
#
#     validation = TimeSeriesDataSet.from_dataset(
#         training, ftse_df, predict=True, stop_randomization=True
#     )
#
#     return training, validation
#
#
#
# if __name__ == "__main__":
#
#     # import matplotlib
#     # matplotlib.use("Agg")
#     import torch
#     print("cuda_is_available:", torch.cuda.is_available(),
#         "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
#
#     load_data()
