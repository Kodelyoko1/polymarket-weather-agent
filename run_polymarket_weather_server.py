#!/usr/bin/env python3
"""Flask web server for Render deployment.

Exposes:
  GET /health        - Health check (keeps Render dyno alive)
  GET /status        - Current agent status as JSON
  GET /              - HTML dashboard

Runs background trading loop in daemon thread.
"""

import os
import sys
import json
import threading
import logging
from pathlib import Path
from datetime import datetime
from time import sleep

from flask import Flask, jsonify, render_template_string

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

from polymarket_weather.tools import run_full_cycle, refresh_data, retrain_models
from polymarket_weather.risk import RiskManager

app = Flask(__name__)

# Global state
_state = {
    "status": "starting",
    "cycles": 0,
    "last_cycle_time": None,
    "last_cycle_result": {},
    "next_cycle_time": None,
    "is_trading": False,
}


@app.route("/health")
def health():
    """Health check endpoint - Render uses this to keep dyno alive."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "cycles": _state["cycles"],
    }), 200


@app.route("/status")
def status():
    """Return current agent status."""
    risk = RiskManager()
    return jsonify({
        "status": _state["status"],
        "cycles_completed": _state["cycles"],
        "last_cycle": _state["last_cycle_time"],
        "next_cycle": _state["next_cycle_time"],
        "is_trading": _state["is_trading"],
        "risk": risk.status_dict(),
        "result": _state["last_cycle_result"],
    }), 200


@app.route("/")
def index():
    """HTML dashboard."""
    risk = RiskManager()
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PolyMarket Weather Agent</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f0f; color: #eee; }}
            .container {{ max-width: 1000px; margin: 0 auto; padding: 20px; }}
            .header {{ border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }}
            h1 {{ font-size: 28px; margin-bottom: 5px; }}
            .status {{ font-size: 14px; color: #888; }}
            .status.ok {{ color: #4ade80; }}
            .status.error {{ color: #f87171; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
            .card {{ background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 20px; }}
            .card h3 {{ font-size: 12px; text-transform: uppercase; color: #888; margin-bottom: 10px; }}
            .card .value {{ font-size: 24px; font-weight: bold; margin-bottom: 5px; }}
            .card .unit {{ font-size: 12px; color: #888; }}
            .card.alert {{ border-color: #f87171; }}
            .card.alert h3 {{ color: #f87171; }}
            .log {{ background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 20px; margin-top: 20px; }}
            .log h3 {{ margin-bottom: 10px; }}
            .log code {{ display: block; font-family: 'Courier New', monospace; font-size: 12px; line-height: 1.5; white-space: pre-wrap; color: #888; max-height: 300px; overflow-y: auto; }}
            .refresh {{ text-align: center; margin-top: 20px; }}
            .refresh button {{ background: #2563eb; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-size: 14px; }}
            .refresh button:hover {{ background: #1d4ed8; }}
        </style>
        <script>
            function refresh() {{ location.reload(); }}
            setInterval(refresh, 30000);  // Auto-refresh every 30s
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌦️ PolyMarket Weather Trading Agent</h1>
                <div class="status ok">Status: {_state['status'].upper()}</div>
            </div>

            <div class="grid">
                <div class="card">
                    <h3>Cycles Completed</h3>
                    <div class="value">{_state['cycles']}</div>
                </div>
                <div class="card">
                    <h3>Bankroll</h3>
                    <div class="value">${risk.state.current_bankroll:.2f}</div>
                    <div class="unit">USDC</div>
                </div>
                <div class="card">
                    <h3>Daily P&L</h3>
                    <div class="value">${risk.state.daily_pnl:.2f}</div>
                    <div class="unit">({risk.state.daily_pnl/risk.state.starting_bankroll*100:.1f}%)</div>
                </div>
                <div class="card{'alert' if risk.state.is_halted else ''}">
                    <h3>Open Positions</h3>
                    <div class="value">{len(risk.state.open_positions)}</div>
                </div>
                <div class="card{'alert' if risk.state.consecutive_losses >= 3 else ''}">
                    <h3>Consecutive Losses</h3>
                    <div class="value">{risk.state.consecutive_losses}</div>
                </div>
                <div class="card">
                    <h3>Trading Status</h3>
                    <div class="value">{'🟢 LIVE' if _state['is_trading'] else '🔴 HALTED' if risk.state.is_halted else '⚫ IDLE'}</div>
                </div>
            </div>

            <div class="log">
                <h3>Last Cycle Result</h3>
                <code>{json.dumps(_state['last_cycle_result'], indent=2, default=str)}</code>
            </div>

            <div class="refresh">
                <button onclick="refresh()">Refresh Now</button>
            </div>
        </div>
    </body>
    </html>
    """
    return html


def _trading_loop():
    """Background daemon thread running trading cycles."""
    logger.info("Trading loop starting...")
    cycle_minutes = int(os.getenv("PW_CYCLE_MINUTES", "60"))
    
    # Force refresh and retrain on startup
    logger.info("Initial data refresh and model retrain on startup...")
    try:
        refresh_data(force=True)
        retrain_models(force=True)
    except Exception as e:
        logger.warning(f"Initial setup failed: {e}")
    
    _state["status"] = "running"
    
    while True:
        try:
            _state["is_trading"] = True
            logger.info(f"Running trading cycle (every {cycle_minutes} minutes)...")
            
            result = run_full_cycle()
            
            _state["cycles"] += 1
            _state["last_cycle_time"] = datetime.now().isoformat()
            _state["last_cycle_result"] = result
            _state["next_cycle_time"] = (
                datetime.now() + 
                __import__("datetime").timedelta(minutes=cycle_minutes)
            ).isoformat()
            
            logger.info(f"Cycle complete. Next in {cycle_minutes} minutes.")
        except Exception as e:
            logger.error(f"Cycle failed: {e}", exc_info=True)
            _state["status"] = "error"
        finally:
            _state["is_trading"] = False
            sleep(cycle_minutes * 60)


def main():
    """Start Flask server."""
    # Start background trading loop
    t = threading.Thread(target=_trading_loop, daemon=True)
    t.start()
    
    # Start Flask
    port = int(os.getenv("PORT", "10000"))
    logger.info(f"Starting Flask server on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
