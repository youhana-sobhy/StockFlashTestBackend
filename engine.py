from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from schemas import BacktestRequest, BacktestResponse, EquityPoint


SUPPORTED_INDICATORS = {"RSI", "SMA", "EMA"}


def fetch_historical_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
    )

    if data.empty:
        raise ValueError(f"No historical data found for ticker '{ticker}' in the given date range.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {sorted(missing)}")

    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    data.index = pd.to_datetime(data.index)
    return data.sort_index()


def calculate_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def calculate_sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(window=period, min_periods=period).mean()


def calculate_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, min_periods=period, adjust=False).mean()


def calculate_indicator(data: pd.DataFrame, name: str, period: int) -> pd.Series:
    name_upper = name.upper()
    close = data["Close"]

    if name_upper == "RSI":
        return calculate_rsi(close, period)
    if name_upper == "SMA":
        return calculate_sma(close, period)
    if name_upper == "EMA":
        return calculate_ema(close, period)

    raise ValueError(f"Unsupported indicator '{name}'. Supported: {sorted(SUPPORTED_INDICATORS)}")


def evaluate_entry_condition(indicator_value: float, operator: str, threshold: float) -> bool:
    if np.isnan(indicator_value):
        return False
    if operator == "<":
        return indicator_value < threshold
    if operator == ">":
        return indicator_value > threshold
    raise ValueError(f"Unsupported operator '{operator}'. Use '<' or '>'.")


def simulate_strategy(data: pd.DataFrame, request: BacktestRequest) -> tuple[list[dict], list[dict]]:
    """
    Walk through the dataframe day-by-day:
    - Enter when entry condition is met and flat.
    - Exit when take-profit or stop-loss is hit intraday using High/Low.
    """
    indicator = calculate_indicator(
        data,
        request.entry_condition.indicator.name,
        request.entry_condition.indicator.period,
    )

    cash = request.initial_capital
    shares = 0.0
    entry_price = 0.0
    in_position = False

    trades: list[dict] = []
    equity_records: list[dict] = []

    take_profit_multiplier = 1 + (request.exit_condition.take_profit_pct / 100.0)
    stop_loss_multiplier = 1 - (request.exit_condition.stop_loss_pct / 100.0)

    for date, row in data.iterrows():
        close_price = float(row["Close"])
        high_price = float(row["High"])
        low_price = float(row["Low"])
        indicator_value = float(indicator.loc[date])

        if in_position:
            take_profit_price = entry_price * take_profit_multiplier
            stop_loss_price = entry_price * stop_loss_multiplier

            exit_reason = None
            exit_price = close_price

            if high_price >= take_profit_price:
                exit_reason = "take_profit"
                exit_price = take_profit_price
            elif low_price <= stop_loss_price:
                exit_reason = "stop_loss"
                exit_price = stop_loss_price

            if exit_reason:
                proceeds = shares * exit_price
                pnl = proceeds - (shares * entry_price)
                return_pct = ((exit_price - entry_price) / entry_price) * 100.0

                trades.append(
                    {
                        "entry_date": entry_date.isoformat(),
                        "exit_date": date.date().isoformat(),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return_pct": return_pct,
                        "pnl": pnl,
                        "exit_reason": exit_reason,
                    }
                )

                cash += proceeds
                shares = 0.0
                entry_price = 0.0
                in_position = False

        if not in_position:
            if evaluate_entry_condition(
                indicator_value,
                request.entry_condition.operator,
                request.entry_condition.value,
            ):
                if close_price <= 0:
                    continue
                shares = cash / close_price
                entry_price = close_price
                entry_date = date
                cash = 0.0
                in_position = True

        strategy_equity = cash + (shares * close_price)
        equity_records.append(
            {
                "date": date.date().isoformat(),
                "strategyEquity": round(strategy_equity, 2),
            }
        )

    if in_position:
        final_date = data.index[-1]
        final_price = float(data.iloc[-1]["Close"])
        proceeds = shares * final_price
        pnl = proceeds - (shares * entry_price)
        return_pct = ((final_price - entry_price) / entry_price) * 100.0

        trades.append(
            {
                "entry_date": entry_date.isoformat(),
                "exit_date": final_date.date().isoformat(),
                "entry_price": entry_price,
                "exit_price": final_price,
                "return_pct": return_pct,
                "pnl": pnl,
                "exit_reason": "end_of_period",
            }
        )

        equity_records[-1]["strategyEquity"] = round(proceeds, 2)

    return trades, equity_records


def build_buy_hold_curve(data: pd.DataFrame, initial_capital: float) -> list[dict]:
    first_close = float(data.iloc[0]["Close"])
    if first_close <= 0:
        raise ValueError("Invalid starting close price for buy-and-hold benchmark.")

    shares = initial_capital / first_close
    curve: list[dict] = []

    for date, row in data.iterrows():
        close_price = float(row["Close"])
        curve.append(
            {
                "date": date.date().isoformat(),
                "buyHoldEquity": round(shares * close_price, 2),
            }
        )

    return curve


def calculate_max_drawdown(equity_series: pd.Series) -> float:
    if equity_series.empty:
        return 0.0

    rolling_peak = equity_series.cummax()
    drawdown = (equity_series - rolling_peak) / rolling_peak
    max_drawdown = drawdown.min()

    if pd.isna(max_drawdown):
        return 0.0

    return abs(float(max_drawdown)) * 100.0


def merge_equity_curves(strategy_curve: list[dict], buy_hold_curve: list[dict]) -> list[EquityPoint]:
    buy_hold_map = {point["date"]: point["buyHoldEquity"] for point in buy_hold_curve}
    merged: list[EquityPoint] = []

    for point in strategy_curve:
        date = point["date"]
        merged.append(
            EquityPoint(
                date=date,
                strategyEquity=point["strategyEquity"],
                buyHoldEquity=buy_hold_map.get(date, point["strategyEquity"]),
            )
        )

    return merged


def run_backtest(request: BacktestRequest) -> BacktestResponse:
    ticker = request.ticker.strip().upper()
    data = fetch_historical_data(ticker, request.start_date, request.end_date)

    trades, strategy_curve = simulate_strategy(data, request)
    buy_hold_curve = build_buy_hold_curve(data, request.initial_capital)
    equity_curve = merge_equity_curves(strategy_curve, buy_hold_curve)

    strategy_equities = pd.Series([point.strategyEquity for point in equity_curve])
    final_strategy_equity = float(strategy_equities.iloc[-1])
    final_buy_hold_equity = float(equity_curve[-1].buyHoldEquity)

    total_return_pct = ((final_strategy_equity - request.initial_capital) / request.initial_capital) * 100.0
    buy_hold_return_pct = ((final_buy_hold_equity - request.initial_capital) / request.initial_capital) * 100.0
    max_drawdown_pct = calculate_max_drawdown(strategy_equities)

    winning_trades = sum(1 for trade in trades if trade["pnl"] > 0)
    total_trades = len(trades)
    win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    strategy_blueprint = {
        "ticker": ticker,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "initial_capital": request.initial_capital,
        "entry_condition": {
            "indicator": {
                "name": request.entry_condition.indicator.name.upper(),
                "period": request.entry_condition.indicator.period,
            },
            "operator": request.entry_condition.operator,
            "value": request.entry_condition.value,
        },
        "exit_condition": {
            "take_profit_pct": request.exit_condition.take_profit_pct,
            "stop_loss_pct": request.exit_condition.stop_loss_pct,
        },
        "trades": trades,
    }

    return BacktestResponse(
        total_trades=total_trades,
        win_rate=round(win_rate, 2),
        total_return_pct=round(total_return_pct, 2),
        max_drawdown_pct=round(max_drawdown_pct, 2),
        buy_hold_return_pct=round(buy_hold_return_pct, 2),
        equity_curve=equity_curve,
        strategy_blueprint=strategy_blueprint,
    )
