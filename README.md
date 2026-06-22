# PolyMarket Weather Trading Agent

An autonomous AI-powered trading agent that discovers pricing inefficiencies in PolyMarket weather prediction markets and executes trades with a data-driven edge.

## Features

- **Weather Data Pipeline**: Fetches 3+ years of historical weather from Open-Meteo (free, no API key)
- **XGBoost Models**: Trains 5 binary classifiers for weather outcomes (temp, precipitation, wind)
- **Market Scanning**: Continuously scans PolyMarket Gamma API for trading opportunities
- **Risk Management**: Kelly criterion sizing with portfolio kill switch and drawdown limits
- **Live Trading**: Places limit orders on PolyMarket CLOB API when edge detected
- **24/7 Operation**: Runs on Render free tier (Flask server + background daemon thread)
- **Credential Rotation**: GitHub Actions workflow rotates API keys without exposing secrets
- **Offline Mode**: Fully functional with synthetic data (no API keys required for development)

## Quick Start

### Local Development

```bash
git clone https://github.com/Kodelyoko1/polymarket-weather-agent.git
cd polymarket-weather-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-pw.txt
cp .env.example .env
python3 run_polymarket_weather_auto.py
```

### Deployment

1. Push to GitHub
2. Connect to Render (New → Blueprint)
3. Set environment variables in Render dashboard
4. Access dashboard at `https://<your-app>.onrender.com`

## License

MIT
