import optuna
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "optuna_lr.db"

studies = optuna.get_all_study_summaries(
    storage=f"sqlite:///{DB_PATH}"
)

print("Studies found:")
for s in studies:
    print("-", s.study_name)
