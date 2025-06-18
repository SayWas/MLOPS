"""Script to run the Titanic Classification API."""

import os
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).parent))

os.environ["PYTHONPATH"] = str(Path(__file__).parent)


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info")

    print(f"Starting Titanic Classification API on {host}:{port}")
    print(f"Model type: {os.getenv('MODEL_TYPE', 'random_forest')}")
    print(f"Documentation available at: http://{host}:{port}/docs")

    uvicorn.run(
        "service.api.app:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=True,
        workers=1,
    )
