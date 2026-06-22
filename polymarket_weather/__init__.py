"""PolyMarket Weather Trading Agent - Autonomous trading on weather prediction markets."""

__version__ = "0.1.0"

from .api_client import PolyMarketTrader, Market, OrderBook, get_weather_markets, get_order_book, get_midpoint_price
from .data_pipeline import run_data_pipeline, load_historical_weather, fetch_historical_weather, fetch_forecast_weather
from .model import WeatherForecastModel, train_all_models, engineer_features, EVENT_TYPES
from .agent import WeatherTradingAgent, Opportunity
from .risk import RiskManager
from .backtest import BacktestEngine, kelly_size, threshold_signal
from .synthetic import generate_synthetic_weather, generate_synthetic_markets, generate_full_demo_dataset
from .tools import run_full_cycle

__all__ = [
    "PolyMarketTrader",
    "Market",
    "OrderBook",
    "get_weather_markets",
    "get_order_book",
    "get_midpoint_price",
    "run_data_pipeline",
    "load_historical_weather",
    "fetch_historical_weather",
    "fetch_forecast_weather",
    "WeatherForecastModel",
    "train_all_models",
    "engineer_features",
    "EVENT_TYPES",
    "WeatherTradingAgent",
    "Opportunity",
    "RiskManager",
    "BacktestEngine",
    "kelly_size",
    "threshold_signal",
    "generate_synthetic_weather",
    "generate_synthetic_markets",
    "generate_full_demo_dataset",
    "run_full_cycle",
]
