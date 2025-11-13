#!/usr/bin/env python3
import io
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

try:
    import ccxt
except Exception as e:
    st.error("Failed to import ccxt. Make sure it is installed in your environment.")
    raise


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------

@dataclass
class TradeResult:
    asset: str
    horizon_label: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    avg_days_in_trade: float
    final_equity: float
    ref_bars: int
    fills: int
    tp_hits: int


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------

def utc_now() -> datetime:
    """Current UTC time (aware)."""
    return datetime.now(timezone.utc)


def parse_trailing_tokens(tokens: List[str]) -> List[Tuple[str, relativedelta]]:
    """
    Parse tokens like "1y", "2y", "12m", "18m" into (label, relativedelta).
    """
    out: List[Tuple[str, relativedelta]] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok.endswith("y"):
            years = int(tok[:-1])
            out.append((tok, relativedelta(years=years)))
        elif tok.endswith("m"):
            months = int(tok[:-1])
            out.append((tok, relativedelta(months=months)))
        else:
            raise ValueError(f"Invalid trailing token '{tok}'. Use e.g. 1y, 2y, 12m, 18m.")
    return out


def fetch_ohlcv_hourly(exchange, symbol: str, since_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Fetch 1h OHLCV between since_ms and end_ms from an exchange.
    """
    all_rows = []
    timeframe = "1h"
    limit = 1000
    fetch_since = since_ms

    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        next_since = last_ts + 60 * 60 * 1000  # +1 hour
        if next_since >= end_ms:
            break
        if next_since <= fetch_since:
            break
        fetch_since = next_since
        time.sleep(0.15)  # be kind to API

    if not all_rows:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        ).set_index("timestamp")

    df = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df = df[(df["timestamp"] >= since_ms) & (df["timestamp"] <= end_ms)]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    return df


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return float(dd.min() * 100.0)


def get_clients():
    """Return Binance + Coinbase clients with rate limit enabled."""
    return ccxt.binance({"enableRateLimit": True}), ccxt.coinbase({"enableRateLimit": True})


def fetch_hourly_for_asset(
    symbol_binance: str,
    symbol_cb: str,
    earliest_start: datetime,
    end_dt_utc: datetime,
    priority: List[str],
) -> pd.DataFrame:
    """
    Try exchanges in 'priority' order until one returns data.
    """
    since_ms = int(earliest_start.timestamp() * 1000)
    end_ms = int(end_dt_utc.timestamp() * 1000)

    binance, coinbase = get_clients()
    seq = []
    for p in priority:
        if p.lower() == "binance":
            seq.append(("binance", binance, symbol_binance))
        elif p.lower() in ("coinbase", "coinbasepro", "coinbase-exchange"):
            seq.append(("coinbase", coinbase, symbol_cb))

    for name, ex, sym in seq:
        try:
            df = fetch_ohlcv_hourly(ex, sym, since_ms, end_ms)
            if not df.empty:
                return df
        except Exception as e:
            st.warning(f"{name} fetch failed for {sym}: {e}")
    return pd.DataFrame()


def build_horizon_specs(
    calendar_years: List[int],
    trailing_tokens: List[str],
    end_dt_utc: datetime,
) -> List[Dict]:
    """
    Build a list of horizon specifications:
    - calendar years: full Jan 1 – Jan 1 ranges
    - trailing tokens: ending at end_dt_utc
    Returns list of dicts: {label, start_utc, end_utc}
    """
    specs: List[Dict] = []

    # Calendar years (full years)
    for year in sorted(calendar_years):
        start_utc = datetime(year, 1, 1, tzinfo=timezone.utc)
        end_utc = datetime(year + 1, 1, 1, tzinfo=timezone.utc)  # exclusive
        label = f"Year {year}"
        specs.append({"label": label, "start_utc": start_utc, "end_utc": end_utc})

    # Trailing horizons
    trailing = parse_trailing_tokens(trailing_tokens)
    for token, rd in trailing:
        start_utc = end_dt_utc - rd
        end_utc = end_dt_utc  # exclusive in filtering (< end_utc)
        label = f"Trailing {token}"
        specs.append({"label": label, "start_utc": start_utc, "end_utc": end_utc})

    return specs


# --------------------------------------------------------------------------------------
# Backtest core
# --------------------------------------------------------------------------------------

def backtest_asset(
    df: pd.DataFrame,
    asset_label: str,
    start_utc: datetime,
    end_utc: datetime,
    horizon_label: str,
    buy_off: float,
    tp_up: float,
    fee_rate: float,
    tz_str: str,
    weekday: int,
    ref_hour: int,
    ref_minute: int,
    start_capital: float,
    make_plot: bool,
):
    """
    Backtest strategy for a single asset & horizon on hourly candles.

    Strategy:
    - Every Friday at ref_hour:ref_minute (local tz_str), take that bar's close as reference.
    - Place a limit buy at (1 - buy_off) * ref_price.
    - If intra-candle low <= limit <= high and not already in position -> buy with all capital, pay fee.
    - Set TP at (1 + tp_up) * buy_price.
    - If intra-candle low <= TP <= high while in position -> sell all, pay fee.
    - Only one position at a time.
    """

    # Restrict data to horizon window (end_utc is exclusive)
    df_h = df[(df.index >= start_utc) & (df.index < end_utc)].copy()
    if df_h.empty:
        return None, None

    capital = start_capital
    equity_curve = []
    equity_time = []

    in_position = False
    qty = 0.0
    buy_price = None
    tp_price = None
    active_buy_limit = None
    trade_entry_ts = None

    trade_count = 0
    wins = 0
    trade_durations = []
    ref_bars = 0
    fills = 0
    tp_hits = 0

    # MAIN LOOP THROUGH HOURLY DATA
    for ts, row in df_h.iterrows():
        # Mark-to-market equity
        if in_position and qty > 0:
            equity_curve.append(qty * row["close"])
        else:
            equity_curve.append(capital)
        equity_time.append(ts)

        # Convert candle time to local timezone
        ts_local = ts.tz_convert(tz_str)

        # Is this the reference bar for the week?
        is_reference_bar = (
            ts_local.weekday() == weekday
            and ts_local.hour == ref_hour
            and ts_local.minute == ref_minute
        )

        if is_reference_bar:
            ref_bars += 1

        # Place or refresh buy limit at reference bar (only if flat)
        if is_reference_bar and not in_position:
            ref_price = row["close"]
            active_buy_limit = ref_price * (1.0 - buy_off)

        # Check if buy limit is hit (only if flat and limit exists)
        if active_buy_limit is not None and not in_position:
            if row["low"] <= active_buy_limit <= row["high"]:
                # Buy fills
                buy_price = active_buy_limit
                qty = capital * (1.0 - fee_rate) / buy_price  # fee on notional
                in_position = True
                trade_entry_ts = ts
                tp_price = buy_price * (1.0 + tp_up)
                fills += 1
                active_buy_limit = None

        # If in position, check if TP is hit
        if in_position and qty > 0 and tp_price is not None:
            if row["low"] <= tp_price <= row["high"]:
                # Sell fills
                gross = qty * tp_price
                net = gross * (1.0 - fee_rate)
                capital = net
                in_position = False
                qty = 0.0
                trade_count += 1
                tp_hits += 1
                if tp_price > buy_price:
                    wins += 1
                if trade_entry_ts is not None:
                    trade_durations.append(
                        (ts - trade_entry_ts).total_seconds() / 86400.0
                    )
                trade_entry_ts = None
                tp_price = None

    # If still in position at end, mark to market but no TP hit
    if in_position and qty > 0:
        last_close = df_h.iloc[-1]["close"]
        capital = qty * last_close * (1.0 - fee_rate)  # assume fee to close at end
        trade_count += 1
        if trade_entry_ts is not None:
            trade_durations.append(
                (df_h.index[-1] - trade_entry_ts).total_seconds() / 86400.0
            )

    final_equity = capital
    total_return_pct = (final_equity / start_capital - 1.0) * 100.0 if start_capital > 0 else 0.0
    exact_years = (end_utc - start_utc).total_seconds() / (365.25 * 24 * 3600)
    cagr_pct = (
        ((final_equity / start_capital) ** (1.0 / max(exact_years, 1e-9)) - 1.0) * 100.0
        if final_equity > 0 and start_capital > 0
        else 0.0
    )

    eq_series = pd.Series(equity_curve, index=pd.DatetimeIndex(equity_time))
    max_dd = compute_max_drawdown(eq_series)
    avg_days = float(np.mean(trade_durations)) if trade_durations else 0.0

    tr = TradeResult(
        asset=asset_label,
        horizon_label=horizon_label,
        trades=trade_count,
        wins=wins,
        losses=trade_count - wins,
        win_rate=(wins / trade_count * 100.0) if trade_count > 0 else 0.0,
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        max_drawdown_pct=max_dd,
        avg_days_in_trade=avg_days,
        final_equity=final_equity,
        ref_bars=ref_bars,
        fills=fills,
        tp_hits=tp_hits,
    )

    fig = None
    if make_plot and not eq_series.empty:
        fig, ax = plt.subplots()
        ax.plot(eq_series.index, eq_series.values)
        ax.set_title(f"Equity Curve - {asset_label} - {horizon_label}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Equity (USD)")
        fig.autofmt_xdate()
        fig.tight_layout()

    return tr, fig


# --------------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------------

st.set_page_config(page_title="Crypto Strategy Tester", layout="wide")

st.title("📈 Crypto Strategy Tester")
st.caption("Friday dip-buy (limit) + take-profit, on hourly data (Binance/Coinbase via ccxt).")


with st.sidebar:
    st.header("Strategy parameters")

    buy_off_pct = st.number_input(
        "Buy-off (%) below reference price",
        min_value=0.0,
        max_value=95.0,
        value=20.0,
        step=0.5,
    )
    tp_up_pct = st.number_input(
        "Take-profit (%) above buy price",
        min_value=0.0,
        max_value=200.0,
        value=15.0,
        step=0.5,
    )
    fee_bps = st.number_input(
        "Fee per side (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.10,
        step=0.01,
    )
    start_capital = st.number_input(
        "Start capital (USD)",
        min_value=10.0,
        max_value=10_000_000.0,
        value=1000.0,
        step=100.0,
    )

    st.markdown("---")
    st.subheader("Time & reference bar")

    tz_str = st.text_input("Timezone (IANA)", value="Asia/Tbilisi")
    ref_hour = st.number_input(
        "Reference hour (local, 0–23)",
        min_value=0,
        max_value=23,
        value=12,
        step=1,
    )
    ref_min = st.number_input(
        "Reference minute (0–59)",
        min_value=0,
        max_value=59,
        value=0,
        step=1,
    )
    weekday = st.selectbox(
        "Reference weekday (Mon=0 ... Sun=6)",
        options=list(range(7)),
        index=4,  # 4 = Friday
        format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
    )

    st.markdown("---")
    st.subheader("Horizons")

    # Calendar years section
    now = utc_now()
    current_year = now.year

    max_years_back = st.slider(
        "Show calendar years back (full years only)",
        min_value=1,
        max_value=15,
        value=5,
        help="You can then pick which exact calendar years to backtest.",
    )

    years_available = list(range(current_year - 1, current_year - max_years_back - 1, -1))
    year_labels = [str(y) for y in years_available]

    calendar_selected_labels = st.multiselect(
        "Calendar years (Jan 1 – Dec 31)",
        options=year_labels,
        default=year_labels[:3],
    )
    calendar_selected = [int(y) for y in calendar_selected_labels]

    st.markdown("**Trailing periods (separate from calendar years)**")
    trailing_options = ["3m", "6m", "9m", "12m", "18m", "24m", "36m", "48m", "60m"]
    trailing_default = ["12m", "24m", "36m"]
    trailing_selected = st.multiselect(
        "Trailing horizons (months 'm' and years 'y')",
        options=trailing_options,
        default=trailing_default,
    )

    st.caption(
        "You can select multiple calendar years AND multiple trailing periods. "
        "Each will be backtested separately."
    )

    st.markdown("---")
    st.subheader("Exchanges")

    priority = st.multiselect(
        "Exchange priority (first tried → fallback)",
        options=["binance", "coinbase"],
        default=["binance", "coinbase"],
    )

    st.markdown("---")
    st.subheader("Assets")

    st.caption("Binance typically uses USDT, Coinbase uses USD.")

    c1, c2, c3 = st.columns(3)
    with c1:
        a1_label = st.text_input("Asset 1 label", value="BTC")
        a1_binance = st.text_input("Binance symbol 1", value="BTC/USDT")
        a1_cb = st.text_input("Coinbase symbol 1", value="BTC/USD")
    with c2:
        a2_label = st.text_input("Asset 2 label", value="ETH")
        a2_binance = st.text_input("Binance symbol 2", value="ETH/USDT")
        a2_cb = st.text_input("Coinbase symbol 2", value="ETH/USD")
    with c3:
        a3_label = st.text_input("Asset 3 label (optional)", value="")
        a3_binance = st.text_input("Binance symbol 3 (optional)", value="")
        a3_cb = st.text_input("Coinbase symbol 3 (optional)", value="")

    run_btn = st.button("🚀 Run Backtest")


# --------------------------------------------------------------------------------------
# Main execution
# --------------------------------------------------------------------------------------

if run_btn:
    try:
        if not calendar_selected and not trailing_selected:
            st.error("Please select at least one calendar year or one trailing period.")
            st.stop()

        end_dt_utc = utc_now()

        # Build detailed horizon specs
        horizon_specs = build_horizon_specs(
            calendar_years=calendar_selected,
            trailing_tokens=trailing_selected,
            end_dt_utc=end_dt_utc,
        )

        earliest_start = min(spec["start_utc"] for spec in horizon_specs)

        # Build asset list
        assets = []
        if a1_label and a1_binance and a1_cb:
            assets.append({"label": a1_label, "symbol_binance": a1_binance, "symbol_cb": a1_cb})
        if a2_label and a2_binance and a2_cb:
            assets.append({"label": a2_label, "symbol_binance": a2_binance, "symbol_cb": a2_cb})
        if a3_label and a3_binance and a3_cb:
            assets.append({"label": a3_label, "symbol_binance": a3_binance, "symbol_cb": a3_cb})

        if not assets:
            st.error("Please define at least one asset with valid symbols.")
            st.stop()

        st.info("Fetching hourly data (Binance first, Coinbase fallback)...")

        all_rows: List[Dict] = []
        tabs = st.tabs([a["label"] for a in assets])

        for idx, a in enumerate(assets):
            with tabs[idx]:
                df_asset = fetch_hourly_for_asset(
                    symbol_binance=a["symbol_binance"],
                    symbol_cb=a["symbol_cb"],
                    earliest_start=earliest_start,
                    end_dt_utc=end_dt_utc,
                    priority=priority,
                )

                if df_asset.empty:
                    st.error(
                        f"No data for {a['label']}. "
                        f"Check symbols or try switching exchange priority."
                    )
                    continue

                st.write(
                    f"Data range for {a['label']}: "
                    f"{df_asset.index[0]} → {df_asset.index[-1]} (UTC)"
                )

                res_rows = []
                for spec in horizon_specs:
                    hlabel = spec["label"]
                    start_utc = spec["start_utc"]
                    end_utc = spec["end_utc"]

                    tr, fig = backtest_asset(
                        df=df_asset,
                        asset_label=a["label"],
                        start_utc=start_utc,
                        end_utc=end_utc,
                        horizon_label=hlabel,
                        buy_off=buy_off_pct / 100.0,
                        tp_up=tp_up_pct / 100.0,
                        fee_rate=fee_bps / 100.0,
                        tz_str=tz_str,
                        weekday=weekday,
                        ref_hour=int(ref_hour),
                        ref_minute=int(ref_min),
                        start_capital=start_capital,
                        make_plot=True,
                    )

                    if tr is None:
                        st.warning(f"No data inside horizon {hlabel} for {a['label']}.")
                        continue

                    res_rows.append(tr.__dict__)
                    if fig is not None:
                        st.pyplot(fig)

                if res_rows:
                    df_res = pd.DataFrame(res_rows)
                    df_res = df_res[
                        [
                            "asset",
                            "horizon_label",
                            "trades",
                            "wins",
                            "losses",
                            "win_rate",
                            "total_return_pct",
                            "cagr_pct",
                            "max_drawdown_pct",
                            "avg_days_in_trade",
                            "final_equity",
                            "ref_bars",
                            "fills",
                            "tp_hits",
                        ]
                    ].rename(columns={"horizon_label": "horizon"})

                    st.subheader(f"Results — {a['label']}")
                    st.dataframe(df_res, use_container_width=True)

                    # Diagnostics: reference bars vs fills vs TP hits
                    st.markdown("**Diagnostics** (per horizon):")
                    st.dataframe(
                        df_res[["horizon", "ref_bars", "fills", "tp_hits"]],
                        use_container_width=True,
                    )

                    all_rows.extend(res_rows)

                    csv_buffer = io.StringIO()
                    df_res.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label=f"⬇️ Download CSV — {a['label']}",
                        data=csv_buffer.getvalue(),
                        file_name=f"backtest_results_{a['label']}.csv",
                        mime="text/csv",
                    )

        if all_rows:
            all_df = pd.DataFrame(all_rows).rename(columns={"horizon_label": "horizon"})
            st.subheader("All Results (Combined)")
            st.dataframe(all_df, use_container_width=True)
            csv_all = io.StringIO()
            all_df.to_csv(csv_all, index=False)
            st.download_button(
                "⬇️ Download CSV — All Assets",
                data=csv_all.getvalue(),
                file_name="backtest_results_all.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error during backtest: {e}")
else:
    st.markdown(
        """
Set your parameters in the sidebar and click **🚀 Run Backtest**.

**Horizons:**
- *Calendar years* → pick full years like 2020, 2021, 2022 (Jan 1 – Dec 31).
- *Trailing periods* → pick trailing 3m, 6m, 1y, 2y, 3y, 5y, etc.

Your Friday dip-buy strategy is then run **separately** on each selected horizon.
"""
    )
