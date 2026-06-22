"""Weather data fetching and storage pipeline using Open-Meteo API."""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import requests

logger = logging.getLogger(__name__)

# Open-Meteo API endpoints
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"

# City coordinates and UTC offsets
CITY_COORDS: Dict[str, tuple[float, float, int]] = {
    "new_york": (40.7128, -74.0060, -5),
    "los_angeles": (34.0522, -118.2437, -8),
    "chicago": (41.8781, -87.6298, -6),
    "houston": (29.7604, -95.3698, -6),
    "phoenix": (33.4484, -112.0740, -7),
    "philadelphia": (39.9526, -75.1652, -5),
    "san_antonio": (29.4241, -98.4936, -6),
    "dallas": (32.7767, -96.7970, -6),
    "miami": (25.7617, -80.1918, -5),
    "atlanta": (33.7490, -84.3880, -5),
}


def fetch_historical_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> Optional[Dict]:
    """Fetch historical weather from Open-Meteo Archive API.

    Args:
        lat: Latitude
        lon: Longitude
        start_date: YYYY-MM-DD format
        end_date: YYYY-MM-DD format

    Returns:
        Raw JSON response dict or None on error
    """
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "wind_speed_10m_max",
                "shortwave_radiation_sum",
                "et0_fao_evapotranspiration",
            ],
            "timezone": "auto",
        }
        resp = requests.get(ARCHIVE_API, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch historical weather from Open-Meteo: {e}")
        return None


def fetch_forecast_weather(lat: float, lon: float, days: int = 14) -> Optional[Dict]:
    """Fetch weather forecast from Open-Meteo Forecast API.

    Args:
        lat: Latitude
        lon: Longitude
        days: Number of forecast days

    Returns:
        Raw JSON response dict or None on error
    """
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "wind_speed_10m_max",
                "shortwave_radiation_sum",
                "et0_fao_evapotranspiration",
            ],
            "timezone": "auto",
            "forecast_days": days,
        }
        resp = requests.get(FORECAST_API, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch forecast from Open-Meteo: {e}")
        return None


def build_daily_weather_df(raw: Dict) -> List[Dict]:
    """Convert Open-Meteo column-oriented response to list of daily records.

    Args:
        raw: Raw JSON from Open-Meteo API

    Returns:
        List of dicts with date and weather fields
    """
    if not raw or "daily" not in raw:
        return []

    daily = raw["daily"]
    dates = daily.get("time", [])
    records = []

    for i, date_str in enumerate(dates):
        record = {"date": date_str}
        for key in [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
            "shortwave_radiation_sum",
            "et0_fao_evapotranspiration",
        ]:
            values = daily.get(key, [])
            record[key] = values[i] if i < len(values) else None
        records.append(record)

    return records


def load_historical_weather(city: str) -> List[Dict]:
    """Load stored historical weather for a city.

    Args:
        city: City key (e.g., 'new_york')

    Returns:
        List of daily weather records or empty list if not found
    """
    path = Path(f"data/pw_historical/{city}.json")
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load weather for {city}: {e}")
    return []


def _save_historical_weather(city: str, records: List[Dict]) -> None:
    """Save historical weather records for a city (append + deduplicate).

    Args:
        city: City key
        records: List of daily records
    """
    data_dir = Path("data/pw_historical")
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{city}.json"

    # Load existing records if any
    existing = load_historical_weather(city)
    existing_dates = {r["date"] for r in existing}

    # Append new records, avoiding duplicates
    for record in records:
        if record["date"] not in existing_dates:
            existing.append(record)
            existing_dates.add(record["date"])

    # Sort by date
    existing.sort(key=lambda r: r["date"])

    try:
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
        logger.info(f"Saved {len(existing)} records for {city}")
    except Exception as e:
        logger.error(f"Failed to save weather for {city}: {e}")


def run_data_pipeline(
    cities: Optional[List[str]] = None,
    lookback_years: int = 3,
) -> Dict:
    """Fetch and store historical weather for specified cities.

    Args:
        cities: List of city keys. If None, uses all available cities.
        lookback_years: Years of historical data to fetch

    Returns:
        Summary dict with cities updated and records added
    """
    from .synthetic import generate_full_demo_dataset

    if cities is None:
        cities = list(CITY_COORDS.keys())

    today = datetime.now().date()
    start_date = (today - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    cities_updated = []
    total_records = 0

    for city in cities:
        if city not in CITY_COORDS:
            logger.warning(f"Unknown city: {city}")
            continue

        lat, lon, _ = CITY_COORDS[city]
        logger.info(f"Fetching weather for {city} ({start_date} to {end_date})...")

        raw = fetch_historical_weather(lat, lon, start_date, end_date)
        if raw is None:
            logger.warning(f"Failed to fetch {city}. Using synthetic data as fallback.")
            generate_full_demo_dataset()
            continue

        records = build_daily_weather_df(raw)
        if records:
            _save_historical_weather(city, records)
            cities_updated.append(city)
            total_records += len(records)

    # Save last refresh timestamp
    refresh_file = Path("data/pw_historical/last_refresh.txt")
    refresh_file.parent.mkdir(parents=True, exist_ok=True)
    refresh_file.write_text(datetime.now().isoformat())

    return {
        "cities_updated": cities_updated,
        "records_added": total_records,
        "start_date": start_date,
        "end_date": end_date,
    }
