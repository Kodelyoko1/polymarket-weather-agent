"""Weather Trading Agent - core autonomous trading logic."""

import os
import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime, timedelta

from .api_client import get_weather_markets, get_order_book, get_midpoint_price, PolyMarketTrader
from .data_pipeline import fetch_forecast_weather, build_daily_weather_df, load_historical_weather
from .model import WeatherForecastModel, engineer_features
from .risk import RiskManager
from .backtest import kelly_size, threshold_signal

logger = logging.getLogger(__name__)

# City keyword mapping
CITY_KEYWORDS: Dict[str, str] = {
    "new york": "new_york", "nyc": "new_york", "manhattan": "new_york",
    "los angeles": "los_angeles", "la": "los_angeles",
    "chicago": "chicago",
    "houston": "houston",
    "phoenix": "phoenix",
    "philadelphia": "philadelphia", "philly": "philadelphia",
    "san antonio": "san_antonio",
    "dallas": "dallas",
    "miami": "miami",
    "atlanta": "atlanta",
}

# Event keyword mapping
EVENT_KEYWORDS: Dict[str, str] = {
    "above 90": "temp_above_90f", "exceed 90": "temp_above_90f",
    "above 32": "temp_above_32f", "freeze": "temp_above_32f", "frost": "temp_above_32f",
    "rain": "precip_any", "precipitation": "precip_any", "wet": "precip_any",
    "1 inch": "precip_1in", "one inch": "precip_1in",
    "25 mph": "wind_above_25mph", "wind": "wind_above_25mph",
}


def _extract_city(question: str) -> Optional[str]:
    """Extract city from market question."""
    question_lower = question.lower()
    
    # Try exact matches with word boundaries
    for keyword, city in sorted(CITY_KEYWORDS.items(), key=lambda x: -len(x[0])):
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, question_lower):
            return city
    
    return None


def _extract_event_type(question: str) -> str:
    """Extract event type from market question."""
    question_lower = question.lower()
    
    # Find best matching event type
    for keyword, event_type in sorted(EVENT_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if keyword in question_lower:
            return event_type
    
    # Default
    return "temp_above_90f"


@dataclass
class Opportunity:
    """A trading opportunity."""
    market: object  # Market dataclass
    city: str
    event_type: str
    model_prob: float
    market_price: float
    signal: Optional[str]  # "YES", "NO", or None
    edge: float
    size: float
    token_id: str


class WeatherTradingAgent:
    """Autonomous weather market trading agent."""

    def __init__(self):
        """Initialize agent with configuration from environment."""
        self.min_edge = float(os.getenv("PW_MIN_EDGE", "0.07"))
        self.min_liquidity = float(os.getenv("PW_MIN_LIQUIDITY", "500"))
        self.bankroll = float(os.getenv("PW_BANKROLL", "1000"))
        self.kelly_fraction = float(os.getenv("PW_KELLY_FRACTION", "0.25"))
        self.max_position_pct = float(os.getenv("PW_MAX_POSITION_PCT", "0.05"))
        
        self.risk_manager = RiskManager()
        self.trader = PolyMarketTrader()

    def scan_opportunities(self) -> List[Opportunity]:
        """Scan PolyMarket for trading opportunities.

        Returns:
            List of Opportunity objects sorted by edge descending
        """
        opportunities = []
        
        try:
            markets = get_weather_markets(limit=100, closed=False)
        except Exception as e:
            logger.warning(f"Failed to get markets: {e}")
            return []
        
        for market in markets:
            # Skip if insufficient liquidity
            if market.liquidity < self.min_liquidity:
                continue
            
            # Extract city and event
            city = _extract_city(market.question)
            if not city:
                continue
            
            event_type = _extract_event_type(market.question)
            
            # Get model probability
            model_prob = self._get_model_prob(city, event_type)
            
            # Get market price
            yes_token_id = market.yes_token_id()
            if not yes_token_id:
                continue
            
            try:
                market_price = get_midpoint_price(yes_token_id)
            except Exception as e:
                logger.warning(f"Failed to get price for {market.slug}: {e}")
                continue
            
            # Check edge
            signal = threshold_signal(model_prob, market_price, self.min_edge)
            if signal is None:
                continue
            
            edge = abs(model_prob - market_price)
            
            # Calculate size
            size = kelly_size(
                model_prob,
                market_price,
                signal,
                self.bankroll,
                self.max_position_pct,
                self.kelly_fraction,
            )
            
            if size < 1.0:
                continue
            
            # Get token ID based on signal
            if signal == "YES":
                token_id = yes_token_id
            else:
                no_token_id = market.no_token_id()
                if not no_token_id:
                    continue
                token_id = no_token_id
            
            opp = Opportunity(
                market=market,
                city=city,
                event_type=event_type,
                model_prob=model_prob,
                market_price=market_price,
                signal=signal,
                edge=edge,
                size=size,
                token_id=token_id,
            )
            opportunities.append(opp)
        
        # Sort by edge descending
        opportunities.sort(key=lambda o: o.edge, reverse=True)
        return opportunities

    def execute_opportunity(self, opp: Opportunity) -> Dict:
        """Execute a trading opportunity.

        Args:
            opp: Opportunity to execute

        Returns:
            Result dict
        """
        # Check risk
        approved, reason = self.risk_manager.check_trade(
            opp.model_prob,
            opp.market_price,
            opp.signal,
            opp.size,
        )
        
        if not approved:
            logger.info(f"Trade rejected: {reason}")
            return {
                "status": "rejected",
                "reason": reason,
                "market": opp.market.slug,
            }
        
        # Place order
        result = self.trader.place_limit_order(
            token_id=opp.token_id,
            side=opp.signal,
            price=opp.market_price,
            size=opp.size,
        )
        
        # Record in risk manager if live
        if result["status"] in ["live", "dry_run"]:
            order_id = result["order_id"]
            self.risk_manager.record_trade_open(
                order_id=order_id,
                token_id=opp.token_id,
                side=opp.signal,
                size=opp.size,
                entry_price=opp.market_price,
            )
            
            # Log to file
            self._log_trade(opp, result)
        
        return result

    def run_cycle(self) -> Dict:
        """Run one trading cycle: scan → execute.

        Returns:
            Summary dict
        """
        logger.info("Running trading cycle...")
        
        opportunities = self.scan_opportunities()
        logger.info(f"Found {len(opportunities)} opportunities")
        
        results = []
        for opp in opportunities:
            result = self.execute_opportunity(opp)
            results.append(result)
        
        # Summary
        executed = sum(1 for r in results if r["status"] in ["live", "dry_run"])
        
        return {
            "opportunities": len(opportunities),
            "trades_executed": executed,
            "trades_rejected": len(results) - executed,
            "risk_status": self.risk_manager.status_dict(),
        }

    def _get_model_prob(self, city: str, event_type: str) -> float:
        """Get model probability for city/event."""
        try:
            # Load model
            model = WeatherForecastModel.load(event_type)
            if model.model is None:
                logger.warning(f"Model not trained for {event_type}")
                return 0.5
            
            # Get forecast
            forecast = self._get_forecast(city)
            if not forecast:
                return 0.5
            
            # Predict
            probs = model.predict_proba(forecast)
            if probs:
                return probs[0]
            
            return 0.5
        except Exception as e:
            logger.warning(f"Failed to get model prob: {e}")
            return 0.5

    def _get_forecast(self, city: str) -> Optional[List[Dict]]:
        """Get weather forecast for city."""
        try:
            from .data_pipeline import CITY_COORDS
            from .synthetic import generate_synthetic_weather
            from datetime import datetime, timedelta
            
            if city not in CITY_COORDS:
                return None
            
            lat, lon, _ = CITY_COORDS[city]
            
            # Try real forecast
            raw = fetch_forecast_weather(lat, lon, days=14)
            if raw:
                return build_daily_weather_df(raw)
            
            # Fall back to synthetic
            logger.warning(f"Using synthetic forecast for {city}")
            today = datetime.now().date()
            end = (today + timedelta(days=14)).strftime("%Y-%m-%d")
            return generate_synthetic_weather(city, today.strftime("%Y-%m-%d"), end)
        except Exception as e:
            logger.warning(f"Failed to get forecast for {city}: {e}")
            return None

    def _log_trade(self, opp: Opportunity, result: Dict) -> None:
        """Log trade to JSONL file."""
        try:
            import json
            from pathlib import Path
            
            log_file = Path("data/pw_trades/trade_log.jsonl")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            entry = {
                "timestamp": datetime.now().isoformat(),
                "market": opp.market.slug,
                "city": opp.city,
                "event_type": opp.event_type,
                "model_prob": opp.model_prob,
                "market_price": opp.market_price,
                "signal": opp.signal,
                "edge": opp.edge,
                "size": opp.size,
                "result": result,
            }
            
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
