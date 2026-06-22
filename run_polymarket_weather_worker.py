#!/usr/bin/env python3
"""Optional: Simple worker loop for alternative deployment patterns.

Useful if you prefer a separate worker dyno instead of threading.
For Render: add as a separate service with `type: worker`.
"""

import os
import sys
import logging
from pathlib import Path
from time import sleep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from polymarket_weather.tools import run_full_cycle, refresh_data, retrain_models


def main():
    """Run trading loop as standalone worker."""
    logger.info("Worker starting...")
    cycle_minutes = int(os.getenv("PW_CYCLE_MINUTES", "60"))
    
    # Initial setup
    logger.info("Initial data refresh and model retrain...")
    try:
        refresh_data(force=True)
        retrain_models(force=True)
    except Exception as e:
        logger.warning(f"Initial setup failed: {e}")
    
    # Main loop
    while True:
        try:
            logger.info("Running trading cycle...")
            result = run_full_cycle()
            logger.info(f"Cycle complete. Result: {result}")
        except Exception as e:
            logger.error(f"Cycle failed: {e}", exc_info=True)
        
        logger.info(f"Sleeping for {cycle_minutes} minutes...")
        sleep(cycle_minutes * 60)


if __name__ == "__main__":
    main()
