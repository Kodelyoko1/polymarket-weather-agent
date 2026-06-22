"""Offline synthetic data generator for development and fallback."""

import json
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict
from numpy import sin, cos, pi

logger = logging.getLogger(__name__)

# City coordinates and base temperatures
CITY_COORDS = {
    "new_york": (40.7128, -74.0060, 10),
    "los_angeles": (34.0522, -118.2437, 18),
    "chicago": (41.8781, -87.6298, 8),
    "houston": (29.7604, -95.3698, 20),
    "phoenix": (33.4484, -112.0740, 22),
    "philadelphia": (39.9526, -75.1652, 10),
    "san_antonio": (29.4241, -98.4936, 21),
    "dallas": (32.7767, -96.7970, 18),
    "miami": (25.7617, -80.1918, 25),
    "atlanta": (33.7490, -84.3880, 15),
}


def generate_synthetic_weather(
    city: str,
    start_date: str,
    end_date: str,
    seed: int = 42,
) -> List[Dict]:
    """Generate realistic synthetic weather data.

    Args:
        city: City key
        start_date: YYYY-MM-DD format
        end_date: YYYY-MM-DD format
        seed: Random seed

    Returns:
        List of daily weather records
    """
    random.seed(seed)
    records = []

    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Get base temperature for city
    _, _, base_temp = CITY_COORDS.get(city, (0, 0, 15))

    current_date = start
    current_temp = base_temp

    while current_date <= end:
        day_of_year = current_date.timetuple().tm_yday
        year_progress = (day_of_year - 1) / 365.0

        # Seasonal variation: +/- 15°C amplitude
        seasonal = 15 * sin(2 * pi * year_progress)

        # Day-to-day AR(1) noise
        noise = 3 * random.gauss(0, 1)
        current_temp = 0.7 * current_temp + 0.3 * (base_temp + seasonal) + noise

        # Temperature range
        temp_max = current_temp + random.uniform(2, 8)
        temp_min = current_temp - random.uniform(2, 8)
        temp_min = min(temp_min, temp_max - 3)  # Ensure reasonable range

        # Precipitation: Poisson-like events
        precip_sum = 0.0
        if random.random() < 0.3:  # 30% chance of rain
            # Gamma-distributed intensity (shape=2, scale=5mm)
            precip_sum = random.gammavariate(2, 5)

        # Wind: log-normal distribution
        wind_speed = max(0, random.lognormvariate(2, 0.5))
        wind_speed = min(wind_speed, 100)  # Cap at 100 kph

        # Radiation and ET
        radiation = max(0, 20 * sin(2 * pi * year_progress) + random.uniform(-5, 5))
        et = max(0, 3 * sin(2 * pi * year_progress) + random.uniform(-1, 1))

        records.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "temperature_2m_max": round(temp_max, 1),
            "temperature_2m_min": round(temp_min, 1),
            "precipitation_sum": round(precip_sum, 1),
            "wind_speed_10m_max": round(wind_speed, 1),
            "shortwave_radiation_sum": round(radiation, 1),
            "et0_fao_evapotranspiration": round(et, 2),
        })

        current_date += timedelta(days=1)

    return records


def generate_synthetic_markets(
    cities: Optional[List[str]] = None,
    n_per_city: int = 4,
) -> List[Dict]:
    """Generate synthetic PolyMarket weather markets.

    Args:
        cities: List of city keys. If None, uses all.
        n_per_city: Number of markets per city

    Returns:
        List of market dicts
    """
    if cities is None:
        cities = list(CITY_COORDS.keys())

    markets = []
    event_types = [
        ("temp_above_90f", "Will {city} exceed 90°F?"),
        ("temp_above_32f", "Will {city} stay above 32°F?"),
        ("precip_any", "Will {city} see precipitation?"),
        ("wind_above_25mph", "Will {city} have winds over 25 mph?"),
    ]

    for city in cities[:n_per_city]:
        for event_key, question_template in event_types:
            question = question_template.format(city=city.replace("_", " ").title())
            market = {
                "condition_id": f"syn_{city}_{event_key}",
                "question": question,
                "slug": f"syn-{city}-{event_key}",
                "end_date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                "tokens": [
                    {"token_id": f"syn_{city}_{event_key}_yes", "outcome": "YES"},
                    {"token_id": f"syn_{city}_{event_key}_no", "outcome": "NO"},
                ],
                "volume": round(random.uniform(100, 5000), 2),
                "liquidity": round(random.uniform(200, 2000), 2),
                "closed": False,
                "tags": ["weather", "synthetic"],
            }
            markets.append(market)

    return markets


def generate_full_demo_dataset() -> None:
    """Generate complete synthetic dataset for offline development."""
    logger.info("Generating full synthetic dataset...")

    # Historical weather (3 years)
    today = datetime.now().date()
    start_date = (today - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    data_dir = Path("data/pw_historical")
    data_dir.mkdir(parents=True, exist_ok=True)

    for city in CITY_COORDS.keys():
        records = generate_synthetic_weather(city, start_date, end_date, seed=hash(city) % 1000)
        path = data_dir / f"{city}.json"
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        logger.info(f"Generated {len(records)} synthetic records for {city}")

    # Synthetic markets
    markets = generate_synthetic_markets()
    cache_dir = Path("data/pw_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "markets.json", "w") as f:
        json.dump(markets, f, indent=2)
    logger.info(f"Generated {len(markets)} synthetic markets")
