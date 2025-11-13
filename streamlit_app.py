#!/usr/bin/env python3
import io
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytz
import streamlit as st
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

try:
    import ccxt
except Exception as e:
    st.error("Please install ccxt: pip install ccxt")
    raise

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

def utc_now():
    return datetime.now(timezone.utc)

def parse_horizons(tokens: List[str]) -> List[Tuple[str, relativedelta]]:
    out = []
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
            raise ValueError(f"Invalid horizon token '{tok}'. Use 1y, 24m, etc.")
    return out

def build_reference_index(start_utc: datetime, end_utc: datetime, tz_str: str, weekday: int, ref_hour: int, ref_minute: int) -> pd.DatetimeIndex:
    tz = pytz.timezone(tz_str)
    start_local = start_utc.astimezone(tz)
    end_local = end_utc.astimezone(tz)
    dates = pd.date_range(start=start_local.date(), end=end_local.date(), freq="D", tz=tz)
    moments = []
    for d in dates:
        if d.weekday() == weekday:
            dt_local = tz.localize(datetime(d.year, d.month, d.day, ref_hour, ref_minute))
            if dt_local >= start_local and dt_local <= end_local:
                moments.append(dt_local.astimezone(timezone.utc))
    return pd.DatetimeIndex(moments)

def fetch_ohlcv_hourly(exchange, symbol, since_ms, end_ms) -> pd.DataFrame:
    all_rows = []
    timeframe = '1h'
    limit = 1000
    fetch_since = since_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        next_since = last_ts + 60 * 60 * 1000
        if next_since >= end_ms:
            break
        if next_since <= fetch_since:
            break
        fetch_since = next_since
        time.sleep(0.15)

    if not all_rows:
        return pd.DataFrame(columns=['timestamp','open','high','low','close','volume']).set_index('timestamp')

    df = pd.DataFrame(all_rows, columns=['timestamp','open','high','low','close','volume'])
    df = df[(df['timestamp'] >= since_ms) & (df['timestamp'] <= end_ms)]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp').sort_index()
    return df

def compute_max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return float(dd.min() * 100.0)

def backtest_asset(df: pd.DataFrame,
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
                   make_plot: bool):
    df_h = df[(df.index >= start_utc) & (df.index <= end_utc)].copy()
    if df_h.empty:
        return None, None

    capital = 1000.0
    equity_curve = []
    equity_time = []

    in_position = False
    buy_price = None
    tp_price = None
    qty = 0.0
    active_buy_limit = None

    ref_idx = build_reference_index(start_utc, end_utc, tz_str, weekday, ref_hour, ref_minute)

    trade_count = 0
    wins = 0
    trade_durations = []

    for ts, row in df_h.iterrows():
    # Mark-to-market equity
    if in_position and qty > 0:
        equity_curve.append(qty * row['close'])
    else:
        equity_curve.append(capital)
    equity_time.append(ts)

    # Convert this candle's time to local Tbilisi time (or whatever tz_str is)
    ts_local = ts.tz_convert(tz_str)

    is_reference_bar = (
        ts_local.weekday() == weekday and
        ts_local.hour == ref_hour and
        ts_local.minute == ref_minute
    )

    # Place/refresh buy at reference bar
    if is_reference_bar and not in_position:
        ref_price = row['close']
        active_buy_limit = ref_price * (1.0 - buy_off)


        if active_buy_limit is not None and not in_position:
            if row['low'] <= active_buy_limit:
                price = active_buy_limit
                qty = capital * (1 - fee_rate) / price
                buy_price = price
                tp_price = buy_price * (1.0 + tp_up)
                in_position = True
                trade_entry_ts = ts
                active_buy_limit = None

        if in_position and qty > 0:
            if row['high'] >= tp_price:
                gross = qty * tp_price
                net = gross * (1 - fee_rate)
                capital = net
                in_position = False
                qty = 0.0
                trade_count += 1
                if tp_price > buy_price:
                    wins += 1
                trade_durations.append((ts - trade_entry_ts).total_seconds() / 86400.0)

    final_equity_base = capital if not in_position else qty * df_h.iloc[-1]['close']
    scale = start_capital / 1000.0 if start_capital > 0 else 1.0
    final_equity = final_equity_base * scale
    total_return_pct = (final_equity / start_capital - 1.0) * 100.0 if start_capital > 0 else 0.0
    exact_years = (end_utc - start_utc).total_seconds() / (365.25 * 24 * 3600)
    cagr_pct = ((final_equity / start_capital) ** (1.0 / max(exact_years, 1e-9)) - 1.0) * 100.0 if final_equity > 0 else 0.0

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
    )

    fig = None
    if make_plot and not eq_series.empty:
        fig, ax = plt.subplots()
        ax.plot(eq_series.index, eq_series.values)
        ax.set_title(f"Equity Curve - {asset_label} - {horizon_label}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Equity (USD)")
        fig.tight_layout()

    return tr, fig

def get_clients():
    return ccxt.binance({"enableRateLimit": True}), ccxt.coinbase({"enableRateLimit": True})

def fetch_hourly(symbol_binance: str, symbol_cb: str, earliest_start: datetime, end_dt_utc: datetime, priority: List[str]) -> pd.DataFrame:
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
            st.warning(f"{name} fetch failed: {e}")
    return pd.DataFrame()

# --------------------------- UI ---------------------------

st.set_page_config(page_title="Crypto Strategy Tester", layout="wide")
st.title("📈 Crypto Strategy Tester")
st.caption("Friday dip buy (limit) + take-profit strategy — hourly backtest")

with st.sidebar:
    st.header("Parameters")
    buy_off_pct = st.number_input("Buy-off (%) below ref price", min_value=0.0, max_value=95.0, value=20.0, step=0.5)
    tp_up_pct = st.number_input("Take-profit (%) above buy", min_value=0.0, max_value=200.0, value=15.0, step=0.5)
    fee_bps = st.number_input("Fee per side (%)", min_value=0.0, max_value=5.0, value=0.10, step=0.01)
    start_capital = st.number_input("Start capital (USD)", min_value=10.0, max_value=10_000_000.0, value=1000.0, step=100.0)

    tz_str = st.text_input("Timezone (IANA)", value="Asia/Tbilisi")
    ref_hour = st.number_input("Reference hour (local, 0-23)", min_value=0, max_value=23, value=12, step=1)
    ref_min = st.number_input("Reference minute (0-59)", min_value=0, max_value=59, value=0, step=1)
    weekday = st.selectbox("Weekday (Mon=0 ... Sun=6)", options=list(range(7)), index=4)

    default_horizons = ["1y", "2y", "3y", "5y"]
    horizons_selected = horizons_selected = st.multiselect(
    "Horizons (years 'y' or months 'm')",
    options=["6m","9m","12m","18m","24m","36m","48m","60m","1y","2y","3y","5y"],
    default=default_horizons,
)

    priority = st.multiselect("Exchange priority", options=["binance", "coinbase"], default=["binance","coinbase"])

    st.subheader("Assets")
    st.caption("Binance uses USDT, Coinbase uses USD.")
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

if run_btn:
    try:
        horizons = parse_horizons(horizons_selected)
        if not horizons:
            st.error("Please select at least one valid horizon (e.g., 12m, 1y).")
            st.stop()

        end_dt_utc = utc_now()
        max_rd = max((rd for (_, rd) in horizons), key=lambda r: (r.years, r.months, r.days, r.hours))
        earliest_start = end_dt_utc - max_rd

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

        st.info("Fetching hourly data... (Binance first, Coinbase fallback)")
        all_rows = []
        tabs = st.tabs([a["label"] for a in assets])

        for idx, a in enumerate(assets):
            with tabs[idx]:
                df = fetch_hourly(a["symbol_binance"], a["symbol_cb"], earliest_start, end_dt_utc, priority)
                if df.empty:
                    st.error(f"No data for {a['label']} — check symbols or try switching exchange priority.")
                    continue

                res_rows = []
                for (hlabel, rd) in horizons:
                    start_utc = end_dt_utc - rd
                    tr, fig = backtest_asset(
                        df=df,
                        asset_label=a["label"],
                        start_utc=start_utc,
                        end_utc=end_dt_utc,
                        horizon_label=hlabel,
                        buy_off=buy_off_pct/100.0,
                        tp_up=tp_up_pct/100.0,
                        fee_rate=fee_bps/100.0,
                        tz_str=tz_str,
                        weekday=weekday,
                        ref_hour=int(ref_hour),
                        ref_minute=int(ref_min),
                        start_capital=start_capital,
                        make_plot=True
                    )
                    if tr is None:
                        st.warning(f"No data inside horizon {hlabel}.")
                        continue
                    res_rows.append(tr.__dict__)
                    if fig is not None:
                        st.pyplot(fig)

                if res_rows:
                    df_res = pd.DataFrame(res_rows)
                    df_res = df_res[['asset','horizon_label','trades','wins','losses','win_rate','total_return_pct','cagr_pct','max_drawdown_pct','avg_days_in_trade','final_equity']]
                    df_res = df_res.rename(columns={"horizon_label": "horizon"})
                    st.subheader(f"Results — {a['label']}")
                    st.dataframe(df_res, use_container_width=True)

                    all_rows.extend(res_rows)
                    csv_buffer = io.StringIO()
                    df_res.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label=f"⬇️ Download CSV — {a['label']}",
                        data=csv_buffer.getvalue(),
                        file_name=f"backtest_results_{a['label']}.csv",
                        mime="text/csv"
                    )

        if all_rows:
            all_df = pd.DataFrame(all_rows)
            all_df = all_df.rename(columns={"horizon_label": "horizon"})
            st.subheader("All Results (Combined)")
            st.dataframe(all_df, use_container_width=True)
            csv_all = io.StringIO()
            all_df.to_csv(csv_all, index=False)
            st.download_button("⬇️ Download CSV — All Assets", data=csv_all.getvalue(), file_name="backtest_results_all.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
else:
    st.markdown("Set your parameters in the sidebar and click **Run Backtest** to begin.")
    st.markdown("Default config uses **Asia/Tbilisi Fridays at 12:00**, buy **20% below** ref price, TP **+15%**, and **hourly** data.")
