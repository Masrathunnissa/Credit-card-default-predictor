import json
import os
from datetime import datetime

def save_model_version_info(model_path, accuracy):
    version_info = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "accuracy": round(accuracy, 4)
    }

    version_file = "models/model_versions.json"
    if not os.path.exists(version_file):
        with open(version_file, 'w') as f:
            json.dump({}, f)

    with open(version_file, 'r') as f:
        versions = json.load(f)

    version_key = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    versions[version_key] = version_info

    with open(version_file, 'w') as f:
        json.dump(versions, f, indent=4)
