import json
import logging
from datetime import datetime
from pathlib import Path


LOG_DIR = Path("data")
LOG_DIR.mkdir(exist_ok=True)
APP_LOG = LOG_DIR / "app.log"
FEEDBACK_LOG = LOG_DIR / "feedback_log.jsonl"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    file_handler = logging.FileHandler(APP_LOG, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def log_feedback(feature_used: str, language: str, helpful: str, comment: str) -> None:
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "feature_used": feature_used,
        "language": language,
        "helpful": helpful,
        "comment": comment,
    }
    with FEEDBACK_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
