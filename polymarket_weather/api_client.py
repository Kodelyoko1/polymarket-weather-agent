"""PolyMarket Gamma API and CLOB API wrapper for market discovery and order placement."""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

# PolyMarket API endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


@dataclass
class Market:
    """Represents a PolyMarket condition (e.g., weather prediction)."""
    condition_id: str
    question: str
    slug: str
    end_date: str
    tokens: List[Dict]
    volume: float = 0.0
    liquidity: float = 0.0
    closed: bool = False
    tags: List[str] = field(default_factory=list)

    def yes_token_id(self) -> Optional[str]:
        """Get YES token ID from tokens list."""
        for token in self.tokens:
            if token.get("outcome") == "YES":
                return token.get("token_id")
        return None

    def no_token_id(self) -> Optional[str]:
        """Get NO token ID from tokens list."""
        for token in self.tokens:
            if token.get("outcome") == "NO":
                return token.get("token_id")
        return None


@dataclass
class OrderBook:
    """Represents the order book for a token (bids and asks)."""
    bids: List[Dict]
    asks: List[Dict]

    def mid_price(self) -> float:
        """Calculate mid-price as average of best bid and best ask."""
        best_bid = self.bids[0]["price"] if self.bids else 0.5
        best_ask = self.asks[0]["price"] if self.asks else 0.5
        return (best_bid + best_ask) / 2.0


def _load_cached_markets() -> List[Dict]:
    """Load cached markets from disk if available."""
    cache_path = Path("data/pw_cache/markets.json")
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached markets: {e}")
    return []


def _save_markets_cache(markets: List[Dict]) -> None:
    """Save markets to cache."""
    cache_dir = Path("data/pw_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "markets.json"
    try:
        with open(cache_path, "w") as f:
            json.dump(markets, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save markets cache: {e}")


def get_weather_markets(limit: int = 100, closed: bool = False) -> List[Market]:
    """Query Gamma API for weather prediction markets."""
    try:
        params = {"limit": limit, "closed": closed}
        resp = requests.get(f"{GAMMA_API}/markets", params=params, timeout=10)
        resp.raise_for_status()
        raw_markets = resp.json()

        # Save to cache on success
        _save_markets_cache(raw_markets)

        # Filter for weather-related markets
        weather_keywords = ["weather", "temperature", "rain", "wind", "precipitation", "forecast"]
        markets = []
        for m in raw_markets:
            question = (m.get("question") or "").lower()
            tags = [t.lower() for t in m.get("tags", [])]
            tags_str = " ".join(tags).lower()

            # Check if market is weather-related
            if any(kw in question or kw in tags_str for kw in weather_keywords):
                market = Market(
                    condition_id=m.get("condition_id", ""),
                    question=m.get("question", ""),
                    slug=m.get("slug", ""),
                    end_date=m.get("end_date", ""),
                    tokens=m.get("tokens", []),
                    volume=float(m.get("volume", 0)),
                    liquidity=float(m.get("liquidity", 0)),
                    closed=m.get("closed", False),
                    tags=m.get("tags", []),
                )
                markets.append(market)

        return markets
    except Exception as e:
        logger.warning(f"Failed to fetch from Gamma API: {e}. Using cached markets.")
        # Fallback to cached markets
        cached = _load_cached_markets()
        return [
            Market(
                condition_id=m.get("condition_id", ""),
                question=m.get("question", ""),
                slug=m.get("slug", ""),
                end_date=m.get("end_date", ""),
                tokens=m.get("tokens", []),
                volume=float(m.get("volume", 0)),
                liquidity=float(m.get("liquidity", 0)),
                closed=m.get("closed", False),
                tags=m.get("tags", []),
            )
            for m in cached
        ]


def get_order_book(token_id: str) -> OrderBook:
    """Fetch order book for a token from CLOB API."""
    try:
        resp = requests.get(
            f"{CLOB_API}/book",
            params={"token_id": token_id},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        return OrderBook(
            bids=data.get("bids", []),
            asks=data.get("asks", [])
        )
    except Exception as e:
        logger.warning(f"Failed to fetch order book for {token_id}: {e}. Using synthetic book.")
        # Fallback to synthetic book
        import random
        mid = 0.50 + random.uniform(-0.05, 0.05)
        return OrderBook(
            bids=[{"price": mid - 0.01, "size": 100}],
            asks=[{"price": mid + 0.01, "size": 100}]
        )


def get_midpoint_price(token_id: str) -> float:
    """Get midpoint price for a token."""
    book = get_order_book(token_id)
    return book.mid_price()


class PolyMarketTrader:
    """Handles authenticated order placement on PolyMarket CLOB API."""

    def __init__(self):
        """Initialize trader with credentials from environment."""
        self.live_trading = os.getenv("PW_LIVE_TRADING", "0") == "1"
        self.private_key = os.getenv("PW_PRIVATE_KEY", "")
        self.api_key = os.getenv("PW_API_KEY", "")
        self.api_secret = os.getenv("PW_API_SECRET", "")

        # Initialize py-clob-client if credentials available
        self.client = None
        if self.private_key or (self.api_key and self.api_secret):
            try:
                from py_clob_client.client import ClobClient
                from py_clob_client.constants import POLYGON

                if self.api_key and self.api_secret:
                    # L2 auth
                    self.client = ClobClient(chain_id=POLYGON)
                    self.client.set_api_credentials(
                        api_key=self.api_key,
                        api_secret=self.api_secret,
                        passphrase=""
                    )
                elif self.private_key:
                    # L1 auth
                    self.client = ClobClient(chain_id=POLYGON, private_key=self.private_key)
            except ImportError:
                logger.warning("py-clob-client not available. Operating in dry-run mode.")
            except Exception as e:
                logger.warning(f"Failed to initialize PolyMarket client: {e}. Operating in dry-run mode.")

    def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> Dict:
        """Place a limit order on PolyMarket CLOB.

        Args:
            token_id: Token ID (YES or NO outcome)
            side: "BUY" or "SELL"
            price: Price per share (0.01-0.99 USDC)
            size: USDC amount to trade

        Returns:
            Dict with status, order_id, and details
        """
        if not self.live_trading:
            logger.info(f"DRY RUN: {side} {size} USDC of {token_id} at {price}")
            return {
                "status": "dry_run",
                "order_id": f"dry_{token_id[:8]}",
                "side": side,
                "price": price,
                "size": size,
                "token_id": token_id,
            }

        if not self.client:
            logger.error("PolyMarket client not initialized. Cannot place order.")
            return {"status": "error", "reason": "client_not_initialized"}

        try:
            order = self.client.create_order(
                token_id=token_id,
                price=price,
                size=size,
                side=side,
                order_type="LIMIT",
            )
            logger.info(f"Order placed: {order}")
            return {
                "status": "live",
                "order_id": order.get("id", ""),
                "side": side,
                "price": price,
                "size": size,
                "token_id": token_id,
            }
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {"status": "error", "reason": str(e)}
