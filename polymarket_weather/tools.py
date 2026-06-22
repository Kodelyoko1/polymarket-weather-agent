"""Orchestration engine - runs full trading cycle."""

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _needs_data_refresh() -> bool:
    """Check if data needs refreshing."""
    refresh_file = Path("data/pw_historical/last_refresh.txt")
    if not refresh_file.exists():
        return True
    
    try:
        last_refresh = datetime.fromisoformat(refresh_file.read_text())
        hours_ago = (datetime.now() - last_refresh).total_seconds() / 3600
        refresh_hours = int(os.getenv("PW_DATA_REFRESH_HOURS", "6"))
        return hours_ago > refresh_hours
    except Exception:
        return True


def _needs_model_retrain() -> bool:
    """Check if models need retraining."""
    trained_file = Path("data/pw_models/last_trained.txt")
    if not trained_file.exists():
        return True
    
    try:
        last_trained = datetime.fromisoformat(trained_file.read_text())
        days_ago = (datetime.now() - last_trained).days
        retrain_days = int(os.getenv("PW_RETRAIN_DAYS", "7"))
        return days_ago > retrain_days
    except Exception:
        return True


def refresh_data(force: bool = False) -> Dict:
    """Refresh weather data."""
    from .data_pipeline import run_data_pipeline, CITY_COORDS
    
    if not force and not _needs_data_refresh():
        logger.info("Data is fresh, skipping refresh")
        return {"skipped": True}
    
    logger.info("Refreshing weather data...")
    result = run_data_pipeline(list(CITY_COORDS.keys()))
    logger.info(f"Data refresh complete: {result}")
    return result


def retrain_models(force: bool = False) -> Dict:
    """Retrain all models."""
    from .model import train_all_models
    from .data_pipeline import load_historical_weather, CITY_COORDS
    
    if not force and not _needs_model_retrain():
        logger.info("Models are fresh, skipping retrain")
        return {"skipped": True}
    
    logger.info("Retraining models...")
    
    # Load all historical data
    weather_by_city = {}
    for city in CITY_COORDS.keys():
        records = load_historical_weather(city)
        if records:
            weather_by_city[city] = records
    
    if not weather_by_city:
        logger.warning("No historical data to train on")
        return {"error": "no_data"}
    
    results = train_all_models(weather_by_city)
    logger.info(f"Models retrained: {results}")
    return results


def run_backtest_quick() -> Dict:
    """Run quick backtest on recent data."""
    from .backtest import BacktestEngine
    from .model import WeatherForecastModel, EVENT_TYPES
    from .data_pipeline import load_historical_weather, CITY_COORDS
    
    logger.info("Running backtest...")
    
    # Load recent data (last 30 days)
    all_records = []
    for city in CITY_COORDS.keys():
        records = load_historical_weather(city)
        if records:
            # Last 30 days
            all_records.extend(records[-30:])
    
    if not all_records:
        logger.warning("No data for backtest")
        return {}
    
    engine = BacktestEngine(initial_bankroll=1000.0)
    
    # Add model predictions
    for record in all_records:
        # Estimate outcome (simplified)
        record["outcome"] = 1 if record.get("temperature_2m_max", 20) > 25 else 0
        record["market_price"] = 0.5 + (record["outcome"] - 0.5) * 0.2  # Slight bias
        record["model_prob"] = 0.5  # Placeholder
    
    results = engine.run(all_records)
    logger.info(f"Backtest results: {results}")
    return results


def _write_digest() -> None:
    """Write daily digest report."""
    report_dir = Path("data/pw_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"
    
    try:
        with open(report_path, "w") as f:
            f.write(f"# PolyMarket Weather Agent Digest\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(f"## Status\n\n")
            f.write(f"- System running\n")
            f.write(f"- Check back for trade updates\n")
    except Exception as e:
        logger.error(f"Failed to write digest: {e}")


def run_full_cycle() -> Dict:
    """Run complete trading cycle.
    
    Returns:
        Summary dict
    """
    logger.info("\n" + "="*50)
    logger.info("STARTING FULL CYCLE")
    logger.info("="*50 + "\n")
    
    cycle_result = {}
    
    # 1. Refresh data if needed
    refresh_result = refresh_data(force=False)
    cycle_result["data_refresh"] = refresh_result
    
    # 2. Retrain models if needed
    retrain_result = retrain_models(force=False)
    cycle_result["model_retrain"] = retrain_result
    
    # 3. Run trading agent
    from .agent import WeatherTradingAgent
    agent = WeatherTradingAgent()
    agent_result = agent.run_cycle()
    cycle_result["agent"] = agent_result
    
    # 4. Write digest
    _write_digest()
    
    logger.info("\n" + "="*50)
    logger.info(f"CYCLE COMPLETE: {agent_result}")
    logger.info("="*50 + "\n")
    
    return cycle_result
