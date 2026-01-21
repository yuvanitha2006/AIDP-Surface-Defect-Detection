import json
from pathlib import Path

def log_retraining_event(event: dict, path: Path):
    if path.exists():
        try:
            with open(path, "r") as f:
                content = f.read().strip()
                data = json.loads(content) if content else []
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    data.append(event)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
