"""Project-wide paths, constants and logging setup."""
import logging
import sys
from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT.parent / "sweettv-tv-program-recommender"  # read-only

VIEWS_FILE = DATA_DIR / "dataset11-30.csv"
SCHEDULE_FILES = {
    "11-20": DATA_DIR / "export_arh_11-20-final.csv",
    "21-30": DATA_DIR / "export_arh_21-30-final.csv",
    "31-42": DATA_DIR / "export_arh_31-42-final.csv",  # test weeks
}
SUBMISSION_TEMPLATE = DATA_DIR / "submission.csv"

SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
INTERIM_DIR = OUTPUTS_DIR / "interim"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
ISSUES_LOG = OUTPUTS_DIR / "issues_log.md"

# --- Constants -------------------------------------------------------------
SEED = 42
ENCODING = "utf-8"
K = 5  # MAP@K, fixed by the exam conditions
WATCH_THRESHOLD = 0.8  # overlap share needed to call a programme "watched"
TIME_SHIFT_HOURS = 0  # shift applied to view-log timestamps; 0 per data_audit.json["timezone"]

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def get_logger(step: str) -> logging.Logger:
    """Logger that writes to the console and to outputs/logs/<step>.log."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(step)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:  # already configured, e.g. re-run in a notebook
        return logger

    formatter = logging.Formatter(LOG_FORMAT)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(LOGS_DIR / f"{step}.log", encoding=ENCODING)
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
