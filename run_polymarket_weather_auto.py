#!/usr/bin/env python3
"""CLI entry point for local PolyMarket Weather Agent execution."""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

from polymarket_weather.tools import run_full_cycle, refresh_data, retrain_models, run_backtest_quick
from polymarket_weather.agent import WeatherTradingAgent
from polymarket_weather.risk import RiskManager
from polymarket_weather.synthetic import generate_full_demo_dataset


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PolyMarket Weather Trading Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 run_polymarket_weather_auto.py              # Run one full cycle
  python3 run_polymarket_weather_auto.py --train     # Force retrain models
  python3 run_polymarket_weather_auto.py --backtest  # Run backtest
  python3 run_polymarket_weather_auto.py --live      # Enable live trading
        """,
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Force retrain models",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest on historical data",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show risk manager status",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh weather data",
    )
    parser.add_argument(
        "--opportunities",
        action="store_true",
        help="Scan and show opportunities only (no trading)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (sets PW_LIVE_TRADING=1)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate full synthetic dataset",
    )

    args = parser.parse_args()

    # Handle --live flag
    if args.live:
        os.environ["PW_LIVE_TRADING"] = "1"
        logger.warning("\n" + "="*50)
        logger.warning("LIVE TRADING ENABLED!")
        logger.warning("="*50 + "\n")

    # Handle --demo flag
    if args.demo:
        logger.info("Generating full synthetic dataset...")
        generate_full_demo_dataset()
        return

    # Handle --status flag
    if args.status:
        risk = RiskManager()
        print("\nRisk Manager Status:")
        print("-" * 50)
        for key, value in risk.status_dict().items():
            print(f"{key:.<40} {value}")
        return

    # Handle --opportunities flag
    if args.opportunities:
        logger.info("Scanning opportunities...")
        agent = WeatherTradingAgent()
        opps = agent.scan_opportunities()
        print(f"\nFound {len(opps)} opportunities:\n")
        for i, opp in enumerate(opps, 1):
            print(
                f"{i}. {opp.market.question}\n"
                f"   City: {opp.city}, Event: {opp.event_type}\n"
                f"   Model: {opp.model_prob:.1%}, Market: {opp.market_price:.1%}\n"
                f"   Edge: {opp.edge:.1%}, Size: ${opp.size:.2f}\n"
            )
        return

    # Handle --refresh flag
    if args.refresh:
        logger.info("Forcing data refresh...")
        result = refresh_data(force=True)
        print(f"Data refresh result: {result}")
        return

    # Handle --train flag
    if args.train:
        logger.info("Forcing model retrain...")
        result = retrain_models(force=True)
        print(f"Model retrain result: {result}")
        return

    # Handle --backtest flag
    if args.backtest:
        logger.info("Running backtest...")
        result = run_backtest_quick()
        print("\nBacktest Results:")
        print("-" * 50)
        for key, value in result.items():
            if isinstance(value, float):
                print(f"{key:.<40} {value:.3f}")
            else:
                print(f"{key:.<40} {value}")
        return

    # Default: run full cycle
    logger.info("Running full trading cycle...")
    result = run_full_cycle()
    print("\nCycle Complete!")
    print("-" * 50)
    import json
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
