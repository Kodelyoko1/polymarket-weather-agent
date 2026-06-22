"""Backtesting engine for validating trading strategies."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


def kelly_size(
    model_prob: float,
    market_price: float,
    side: str,
    bankroll: float,
    max_position_pct: float = 0.05,
    kelly_fraction: float = 0.25,
) -> float:
    """Calculate bet size using fractional Kelly criterion.

    Args:
        model_prob: Model probability (0.0-1.0)
        market_price: Market price (0.0-1.0)
        side: "YES" or "NO"
        bankroll: Current bankroll
        max_position_pct: Max position as % of bankroll
        kelly_fraction: Kelly multiplier (conservative)

    Returns:
        Bet size in USDC
    """
    if side == "YES":
        p = model_prob
        odds = 1 / market_price - 1 if market_price > 0 else 1
        q = 1 - model_prob
    else:  # "NO"
        p = 1 - model_prob
        odds = 1 / (1 - market_price) - 1 if market_price < 1 else 1
        q = model_prob

    # Kelly formula: f = (p * (1 + odds) - 1) / odds
    if odds <= 0:
        return 0.0

    kelly = (p * (1 + odds) - 1) / odds

    # Apply fractional Kelly
    kelly = kelly * kelly_fraction

    # Cap at max position
    max_size = bankroll * max_position_pct

    # Return size, but at least $1.00 if positive
    size = max(0, min(kelly * bankroll, max_size))
    return round(max(size, 1.0) if size > 0 else 0.0, 2)


def threshold_signal(
    model_prob: float,
    market_price: float,
    min_edge: float = 0.07,
) -> Optional[str]:
    """Determine trading signal based on edge.

    Args:
        model_prob: Model probability (0.0-1.0)
        market_price: Market price (0.0-1.0)
        min_edge: Minimum edge threshold

    Returns:
        "YES", "NO", or None
    """
    if model_prob - market_price > min_edge:
        return "YES"
    elif market_price - model_prob > min_edge:
        return "NO"
    return None


class BacktestEngine:
    """Simulates trading on historical data with known outcomes."""

    def __init__(self, initial_bankroll: float = 1000.0):
        self.initial_bankroll = initial_bankroll
        self.trades = []

    def run(self, records: List[Dict]) -> Dict:
        """Run backtest on historical records.

        Args:
            records: List of records with features + outcome + market_price

        Returns:
            Backtest metrics dict
        """
        bankroll = self.initial_bankroll
        self.trades = []

        for record in records:
            # Extract required fields
            model_prob = record.get("model_prob", 0.5)
            market_price = record.get("market_price", 0.5)
            outcome = record.get("outcome", 0)  # 0 or 1
            min_edge = record.get("min_edge", 0.07)

            # Determine signal
            signal = threshold_signal(model_prob, market_price, min_edge)
            if signal is None:
                continue

            # Calculate size
            size = kelly_size(model_prob, market_price, signal, bankroll)
            if size < 1.0:
                continue

            # Simulate trade result
            if signal == "YES":
                pnl = size if outcome == 1 else -size
                prob_correct = model_prob
            else:  # "NO"
                pnl = size if outcome == 0 else -size
                prob_correct = 1 - model_prob

            bankroll += pnl
            self.trades.append({
                "signal": signal,
                "size": size,
                "pnl": pnl,
                "outcome": outcome,
                "model_prob": model_prob,
                "market_price": market_price,
                "correct": pnl > 0,
            })

        # Calculate metrics
        if not self.trades:
            return {
                "trades": 0,
                "win_rate": 0.0,
                "roi_pct": 0.0,
                "sharpe": 0.0,
                "max_drawdown_pct": 0.0,
                "final_bankroll": bankroll,
            }

        pnls = [t["pnl"] for t in self.trades]
        correct = sum(1 for t in self.trades if t["correct"])
        win_rate = correct / len(self.trades)
        roi = (bankroll - self.initial_bankroll) / self.initial_bankroll
        roi_pct = roi * 100

        # Sharpe ratio (assuming daily returns)
        returns = np.array(pnls) / self.initial_bankroll
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) / self.initial_bankroll * 100 if len(drawdown) > 0 else 0

        # Brier score (probability calibration)
        y_true = np.array([t["outcome"] for t in self.trades])
        y_pred = np.array([t["model_prob"] if t["signal"] == "YES" else 1 - t["model_prob"] for t in self.trades])
        brier = np.mean((y_pred - y_true) ** 2)

        return {
            "trades": len(self.trades),
            "win_rate": float(win_rate),
            "roi_pct": float(roi_pct),
            "sharpe": float(sharpe),
            "max_drawdown_pct": float(max_drawdown),
            "brier_score": float(brier),
            "final_bankroll": float(bankroll),
        }

    def generate_report(self, name: str) -> Path:
        """Generate markdown backtest report."""
        report_dir = Path("data/pw_reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{name}.md"

        with open(report_path, "w") as f:
            f.write(f"# Backtest Report: {name}\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")

            if self.trades:
                metrics = self.run([])
                f.write(f"## Metrics\n\n")
                f.write(f"- Trades: {metrics['trades']}\n")
                f.write(f"- Win Rate: {metrics['win_rate']:.1%}\n")
                f.write(f"- ROI: {metrics['roi_pct']:.2f}%\n")
                f.write(f"- Sharpe Ratio: {metrics['sharpe']:.2f}\n")
                f.write(f"- Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n")
                f.write(f"- Final Bankroll: ${metrics['final_bankroll']:.2f}\n")
            else:
                f.write("No trades generated.\n")

        return report_path
