"""
Experiment logging utilities for CSAF training runs.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logging(
    log_dir: str,
    run_name: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure root logger to write to both stdout and a log file.

    Args:
        log_dir:  Directory for log files.
        run_name: Optional experiment name (used in filename).
        level:    Logging level (default INFO).

    Returns:
        Configured root logger.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or "csaf"
    log_file = log_dir / f"{name}_{timestamp}.log"

    fmt = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ]
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    logger = logging.getLogger("csaf")
    logger.info(f"Logging to {log_file}")
    return logger


class ExperimentLogger:
    """
    Structured experiment result logger.

    Saves per-epoch metrics, final results, and configuration to JSON.
    """

    def __init__(self, output_dir: str, run_name: str = "csaf") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self._history: Dict[str, Any] = {
            "run_name": run_name,
            "start_time": datetime.now().isoformat(),
            "epochs": [],
            "final_metrics": {},
            "config": {},
        }

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        self._history["epochs"].append({"epoch": epoch, **metrics})

    def log_final(self, metrics: Dict[str, float]) -> None:
        self._history["final_metrics"] = metrics
        self._history["end_time"] = datetime.now().isoformat()

    def log_config(self, config: Dict) -> None:
        self._history["config"] = config

    def save(self) -> None:
        path = self.output_dir / f"{self.run_name}_results.json"
        with open(path, "w") as f:
            json.dump(self._history, f, indent=2)
        logging.getLogger("csaf").info(f"Saved experiment log to {path}")
