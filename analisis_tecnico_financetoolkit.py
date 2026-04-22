import inspect
import logging
import os
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from financetoolkit import Toolkit


logging.getLogger("financetoolkit").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

TICKERS = ["JPM", "KO", "BA"]
START_DATE = "2013-01-01"
END_DATE = "2024-01-01"
PERIOD = "daily"

api_key = os.getenv("FMP_API_KEY")
INVALID_API_KEY_VALUES = {None, "", "FINANCIAL_MODELING_PREP_KEY"}
if api_key in INVALID_API_KEY_VALUES:
    api_key = None

companies = Toolkit(
    tickers=TICKERS,
    api_key=api_key,
    start_date=START_DATE,
    end_date=END_DATE,
)

hist_data = companies.get_historical_data()
tech = companies.technicals


def get_tech_safe(method, **kwargs) -> pd.DataFrame:
    try:
        return method(**kwargs)
    except Exception as exc:
        logger.warning("No se pudo ejecutar %s: %s", method.__name__, exc)
        return pd.DataFrame()


def call_indicator(method, window: Optional[int] = None, period: str = PERIOD) -> pd.DataFrame:
    """Llama indicadores de FinanceToolkit mapeando firmas distintas de parámetros."""
    params = inspect.signature(method).parameters
    kwargs = {}

    if "period" in params:
        kwargs["period"] = period

    if window is not None:
        if "window" in params:
            kwargs["window"] = window
        elif "time_period" in params:
            kwargs["time_period"] = window
        elif "lookback_period" in params:
            kwargs["lookback_period"] = window

    return get_tech_safe(method, **kwargs)


def format_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.index, pd.PeriodIndex):
        df = df.copy()
        df.index = df.index.to_timestamp()
    return df


def extract_series(df: pd.DataFrame, ticker: str, sub_col: Optional[str] = None) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)

    try:
        if isinstance(df.columns, pd.MultiIndex):
            if sub_col is not None:
                if (sub_col, ticker) in df.columns:
                    return df[(sub_col, ticker)]
                if (ticker, sub_col) in df.columns:
                    return df[(ticker, sub_col)]

                lvl0 = set(df.columns.get_level_values(0))
                lvl1 = set(df.columns.get_level_values(1))
                if sub_col in lvl0 and ticker in lvl1:
                    return df.xs(sub_col, level=0, axis=1)[ticker]
                if ticker in lvl0 and sub_col in lvl1:
                    return df.xs(sub_col, level=1, axis=1)[ticker]
            else:
                lvl0 = set(df.columns.get_level_values(0))
                lvl1 = set(df.columns.get_level_values(1))
                if ticker in lvl0:
                    out = df[ticker]
                    return out.iloc[:, 0] if isinstance(out, pd.DataFrame) else out
                if ticker in lvl1:
                    out = df.xs(ticker, level=1, axis=1)
                    return out.iloc[:, 0] if isinstance(out, pd.DataFrame) else out

        if sub_col is not None and sub_col in df.columns:
            return df[sub_col]
        if ticker in df.columns:
            return df[ticker]
    except Exception as exc:
        logger.debug("No se pudo extraer serie para %s (%s): %s", ticker, sub_col, exc)

    return pd.Series(dtype=float)


def get_ohlc_for_ticker(price_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    open_ = extract_series(price_df, ticker, "Open")
    high_ = extract_series(price_df, ticker, "High")
    low_ = extract_series(price_df, ticker, "Low")
    close_ = extract_series(price_df, ticker, "Close")

    ohlc = pd.DataFrame(
        {
            "Open": open_,
            "High": high_,
            "Low": low_,
            "Close": close_,
        }
    ).dropna()
    return ohlc


def draw_candles(ax, ohlc: pd.DataFrame, width: float = 0.65) -> None:
    """Dibuja velas OHLC en un eje de Matplotlib usando un DataFrame con Open, High, Low y Close."""
    if ohlc.empty:
        return

    # Evita cuerpos de tamaño cero cuando Open == Close para que la vela siga siendo visible.
    min_body_height = 1e-6
    x_vals = mdates.date2num(ohlc.index.to_pydatetime())
    ohlc_values = ohlc[["Open", "High", "Low", "Close"]].to_numpy()

    for x, (open_, high_, low_, close_) in zip(x_vals, ohlc_values):
        color = "#1f9d55" if close_ >= open_ else "#d64545"
        ax.vlines(x, low_, high_, color=color, linewidth=1.2, alpha=0.9, zorder=1)

        body_low = min(open_, close_)
        body_height = abs(close_ - open_)
        if body_height == 0:
            body_height = min_body_height

        rect = Rectangle(
            (x - width / 2, body_low),
            width,
            body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=1.0,
            alpha=0.8,
            zorder=2,
        )
        ax.add_patch(rect)

    ax.xaxis_date()


ema = format_index(call_indicator(tech.get_exponential_moving_average, window=20))
dema = format_index(call_indicator(tech.get_double_exponential_moving_average, window=20))
keltner = format_index(call_indicator(tech.get_keltner_channels))
williams_r = format_index(call_indicator(tech.get_williams_percent_r, window=14))
aroon = format_index(call_indicator(tech.get_aroon_indicator, window=25))
atr = format_index(call_indicator(tech.get_average_true_range, window=14))
obv = format_index(call_indicator(tech.get_on_balance_volume))
chaikin = format_index(call_indicator(tech.get_chaikin_oscillator))
hist_data = format_index(hist_data)

for ticker in TICKERS:
    ohlc = get_ohlc_for_ticker(hist_data, ticker)
    if ohlc.empty:
        continue

    fig, axes = plt.subplots(8, 1, figsize=(16, 24), sharex=True)
    fig.suptitle(f"Análisis Técnico Individual - {ticker}", fontsize=18, fontweight="bold", y=0.99)

    # 1) EMA + Velas
    draw_candles(axes[0], ohlc)
    ema_data = extract_series(ema, ticker)
    if not ema_data.empty:
        axes[0].plot(ema.index, ema_data, color="#0052cc", linewidth=2.4, label="EMA (20)", zorder=3)
    axes[0].set_title("1. Exponential Moving Average (EMA)", fontsize=13)

    # 2) DEMA + Velas
    draw_candles(axes[1], ohlc)
    dema_data = extract_series(dema, ticker)
    if not dema_data.empty:
        axes[1].plot(dema.index, dema_data, color="#7b1fa2", linewidth=2.4, label="DEMA (20)", zorder=3)
    axes[1].set_title("2. Double EMA (DEMA)", fontsize=13)

    # 3) Keltner + Velas
    draw_candles(axes[2], ohlc)
    k_up = extract_series(keltner, ticker, "Upper")
    k_lo = extract_series(keltner, ticker, "Lower")
    if not k_up.empty and not k_lo.empty:
        axes[2].fill_between(
            keltner.index,
            k_lo,
            k_up,
            color="#ff9800",
            alpha=0.25,
            label="Keltner Channel",
            zorder=2.5,
        )
        axes[2].plot(keltner.index, k_up, color="#ff9800", linewidth=1.8, alpha=0.9, zorder=3)
        axes[2].plot(keltner.index, k_lo, color="#ff9800", linewidth=1.8, alpha=0.9, zorder=3)
    axes[2].set_title("3. Keltner Channels", fontsize=13)

    # 4) Williams %R
    wr_data = extract_series(williams_r, ticker)
    if not wr_data.empty:
        axes[3].plot(williams_r.index, wr_data, color="#00897b", linewidth=2.3, label="Williams %R")
    axes[3].axhline(-20, color="#e53935", linestyle="--", linewidth=1.4, alpha=0.8)
    axes[3].axhline(-80, color="#43a047", linestyle="--", linewidth=1.4, alpha=0.8)
    axes[3].set_ylim(-100, 0)
    axes[3].set_title("4. Williams %R", fontsize=13)

    # 5) Aroon
    a_up = extract_series(aroon, ticker, "Aroon Up")
    a_down = extract_series(aroon, ticker, "Aroon Down")
    if not a_up.empty and not a_down.empty:
        axes[4].plot(aroon.index, a_up, color="#2e7d32", linewidth=2.2, label="Aroon Up")
        axes[4].plot(aroon.index, a_down, color="#c62828", linewidth=2.2, label="Aroon Down")
    axes[4].set_ylim(0, 100)
    axes[4].set_title("5. Aroon Indicator", fontsize=13)

    # 6) ATR
    atr_data = extract_series(atr, ticker)
    if not atr_data.empty:
        axes[5].plot(atr.index, atr_data, color="#b71c1c", linewidth=2.3, label="ATR")
    axes[5].set_title("6. Average True Range (ATR)", fontsize=13)

    # 7) OBV
    obv_data = extract_series(obv, ticker)
    if not obv_data.empty:
        axes[6].plot(obv.index, obv_data, color="#0d47a1", linewidth=2.2, label="OBV")
    axes[6].set_title("7. On-Balance Volume (OBV)", fontsize=13)

    # 8) Chaikin
    chk = extract_series(chaikin, ticker)
    if not chk.empty:
        axes[7].fill_between(
            chaikin.index,
            0,
            chk,
            where=(chk >= 0),
            color="#43a047",
            alpha=0.30,
        )
        axes[7].fill_between(
            chaikin.index,
            0,
            chk,
            where=(chk < 0),
            color="#e53935",
            alpha=0.30,
        )
        axes[7].plot(chaikin.index, chk, color="#37474f", linewidth=1.8, label="Chaikin Oscillator")
    axes[7].axhline(0, color="black", linewidth=1.0)
    axes[7].set_title("8. Chaikin Oscillator", fontsize=13)
    axes[7].set_xlabel("Fecha", fontsize=11)

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="upper left", fontsize="small", framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.set_facecolor("#fafafa")

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0)

    plt.tight_layout()
    plt.show()
