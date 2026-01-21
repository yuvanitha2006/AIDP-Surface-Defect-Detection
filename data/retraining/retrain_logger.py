import json
from pathlib import Path
from datetime import datetime, timezone

def log_retraining_event(event: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        with open(path, "w") as f:
            json.dump([], f)

    with open(path, "r+") as f:
        data = json.load(f)
        data.append(event)
        f.seek(0)
        json.dump(data, f, indent=2)
