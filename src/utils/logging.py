"""Logging utilities for reproducibility."""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Configure logging
LOG_DIR = Path(__file__).parent.parent.parent / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Main logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("gender_bias_research")


class APICallLogger:
    """Logger for tracking all LLM API calls for reproducibility."""

    def __init__(self, log_file: Optional[str] = None):
        if log_file is None:
            log_file = LOG_DIR / f"api_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_call(
        self,
        endpoint: str,
        model: str,
        prompt: str,
        response: Any,
        metadata: Optional[dict] = None
    ):
        """Log an API call with full details."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "model": model,
            "prompt": prompt[:5000] if len(prompt) > 5000 else prompt,  # Truncate very long prompts
            "response": response,
            "metadata": metadata or {}
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_error(self, endpoint: str, error: str, metadata: Optional[dict] = None):
        """Log an API error."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "error": error,
            "metadata": metadata or {}
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")


# Global API logger instance
api_logger = APICallLogger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(f"gender_bias_research.{name}")
