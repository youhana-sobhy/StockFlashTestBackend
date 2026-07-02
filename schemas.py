from pydantic import BaseModel, Field


class IndicatorConfig(BaseModel):
    name: str = Field(..., description="Indicator name: RSI, SMA, or EMA")
    period: int = Field(..., gt=0, description="Lookback period for the indicator")


class EntryCondition(BaseModel):
    indicator: IndicatorConfig
    operator: str = Field(..., pattern=r"^(<|>)$", description="Comparison operator")
    value: float = Field(..., description="Threshold value to compare against")


class ExitCondition(BaseModel):
    take_profit_pct: float = Field(..., gt=0, description="Take profit percentage")
    stop_loss_pct: float = Field(..., gt=0, description="Stop loss percentage")


class BacktestRequest(BaseModel):
    ticker: str = Field(..., min_length=1, description="Stock ticker symbol")
    start_date: str = Field(..., description="Start date YYYY-MM-DD")
    end_date: str = Field(..., description="End date YYYY-MM-DD")
    entry_condition: EntryCondition
    exit_condition: ExitCondition
    initial_capital: float = Field(..., gt=0, description="Starting capital in USD")


class EquityPoint(BaseModel):
    date: str
    strategyEquity: float
    buyHoldEquity: float


class BacktestResponse(BaseModel):
    total_trades: int
    win_rate: float
    total_return_pct: float
    max_drawdown_pct: float
    buy_hold_return_pct: float
    equity_curve: list[EquityPoint]
    strategy_blueprint: dict
