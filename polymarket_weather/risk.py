"""Risk management and position tracking."""

import json
import logging
from pathlib import Path
from typing import Tuple, Dict
from dataclasses import dataclass, asdict, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    """In-memory risk state."""
    is_halted: bool = False
    open_positions: Dict[str, Dict] = field(default_factory=dict)
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    starting_bankroll: float = 1000.0
    current_bankroll: float = 1000.0


class RiskManager:
    """Tracks risk metrics and enforces trading limits."""

    def __init__(self):
        """Initialize risk manager."""
        self.state_file = Path("data/pw_trades/risk_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Load or create state
        self.state = self._load_state()

        # Configuration
        import os
        self.min_edge = float(os.getenv("PW_MIN_EDGE", "0.07"))
        self.max_position_pct = float(os.getenv("PW_MAX_POSITION_PCT", "0.05"))
        self.max_open_positions = 10
        self.daily_loss_limit_pct = 0.1  # 10% of bankroll
        self.max_consecutive_losses = 5

    def check_trade(
        self,
        model_prob: float,
        market_price: float,
        side: str,
        proposed_size: float,
    ) -> Tuple[bool, str]:
        """Check if a trade is allowed.

        Args:
            model_prob: Model probability (0.0-1.0)
            market_price: Market price (0.0-1.0)
            side: "YES" or "NO"
            proposed_size: Size in USDC

        Returns:
            (approved: bool, reason: str)
        """
        # Check halt
        if self.state.is_halted:
            return False, "System is halted"

        # Check edge
        edge = abs(model_prob - market_price)
        if edge < self.min_edge:
            return False, f"Edge {edge:.3f} < min {self.min_edge}"

        # Check position size
        max_size = self.state.current_bankroll * self.max_position_pct
        if proposed_size > max_size:
            return False, f"Size {proposed_size} > max {max_size}"

        # Check open positions
        if len(self.state.open_positions) >= self.max_open_positions:
            return False, f"Max open positions ({self.max_open_positions}) reached"

        # Check daily loss limit
        starting_bankroll = self.state.starting_bankroll
        if self.state.daily_pnl < -starting_bankroll * self.daily_loss_limit_pct:
            return False, f"Daily loss limit exceeded: {self.state.daily_pnl}"

        return True, "Approved"

    def record_trade_open(
        self,
        order_id: str,
        token_id: str,
        side: str,
        size: float,
        entry_price: float = 0.5,
    ) -> None:
        """Record an open position."""
        self.state.open_positions[order_id] = {
            "token_id": token_id,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "opened_at": datetime.now().isoformat(),
        }
        self._save_state()
        logger.info(f"Position opened: {order_id}")

    def record_trade_close(self, order_id: str, pnl: float) -> None:
        """Record a closed position and update metrics."""
        if order_id in self.state.open_positions:
            del self.state.open_positions[order_id]

        # Update bankroll and daily P&L
        self.state.current_bankroll += pnl
        self.state.daily_pnl += pnl

        # Update consecutive losses
        if pnl < 0:
            self.state.consecutive_losses += 1
            if self.state.consecutive_losses >= self.max_consecutive_losses:
                self.halt(f"Consecutive losses: {self.state.consecutive_losses}")
        else:
            self.state.consecutive_losses = 0

        self._save_state()
        logger.info(f"Position closed: {order_id}, P&L: {pnl}")

    def halt(self, reason: str = "Manual halt") -> None:
        """Halt trading."""
        self.state.is_halted = True
        self._save_state()
        logger.error(f"Trading halted: {reason}")

    def resume(self) -> None:
        """Resume trading."""
        self.state.is_halted = False
        self.state.daily_pnl = 0.0
        self.state.consecutive_losses = 0
        self._save_state()
        logger.info("Trading resumed")

    def status_dict(self) -> Dict:
        """Get current risk state as dict."""
        return {
            "is_halted": self.state.is_halted,
            "current_bankroll": round(self.state.current_bankroll, 2),
            "daily_pnl": round(self.state.daily_pnl, 2),
            "consecutive_losses": self.state.consecutive_losses,
            "open_positions": len(self.state.open_positions),
            "n_open_positions": len(self.state.open_positions),
        }

    def _load_state(self) -> RiskState:
        """Load risk state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                    return RiskState(**data)
            except Exception as e:
                logger.warning(f"Failed to load risk state: {e}. Using defaults.")
        return RiskState()

    def _save_state(self) -> None:
        """Save risk state to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                data = asdict(self.state)
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save risk state: {e}")
