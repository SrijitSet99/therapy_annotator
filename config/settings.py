from pathlib import Path
import os

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH: str = str(PACKAGE_ROOT / "data" / "raw" / "conversations.json")
OUTPUT_PATH: str = "data/processed/dataset_rows.json"
FAILED_PATH: str = "data/processed/failed_rows.json"
LOG_LEVEL: str = "INFO"
PARALLEL_WORKERS: int = 4
LOG_DIR: str = os.getenv("ANNOTATE_LOG_DIR", "annotate_logs")
PATIENTS_DIR: str = os.getenv("ANNOTATE_PATIENTS_DIR", "data/patients")

# Adaptive routing thresholds
CONFIDENCE_DEBATE_THRESHOLD: float = 0.8   # below → trigger debate
MAX_REVISION_ROUNDS: int = 3               # prevent confidence/revision loops from running forever
CONSENSUS_VOTERS: int = 3                  # number of independent consensus agents
DISAGREEMENT_ARBITRATION_THRESHOLD: float = 0.5  # vote agreement below → arbitrate
