import json
from pathlib import Path
from datetime import datetime, timezone

TRIGGER_PATH = Path("triggers/retrain_trigger.json")

def retrain_model():
    print("🔁 Retraining model...")
    # PLACEHOLDER: load data, retrain autoencoder, save model
    print("✅ Model retrained successfully")

def main():
    if not TRIGGER_PATH.exists():
        print("No retraining trigger found.")
        return

    with open(TRIGGER_PATH) as f:
        trigger = json.load(f)

    if trigger.get("retrain") is True:
        print("🚨 Retraining triggered")
        print("Reason:", trigger["reason"])
        print("Timestamp:", trigger["timestamp"])

        retrain_model()

        # Reset trigger after retraining
        trigger["retrain"] = False
        trigger["completed_at"] = datetime.now(timezone.utc).isoformat()

        with open(TRIGGER_PATH, "w") as f:
            json.dump(trigger, f, indent=2)

        print("🔒 Trigger reset")

    else:
        print("No retraining required.")

if __name__ == "__main__":
    main()
