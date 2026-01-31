# gateway/tests/conftest.py
import os
import pathlib

HERE = pathlib.Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]  # .../llm-internal-assistant
CONFIG_FILE = REPO_ROOT / "configs" / "server.example.yaml"

# Force override to avoid stale shell env
os.environ["CONFIG_PATH"] = str(CONFIG_FILE)
os.environ.setdefault("VLLM_BASE_URL", "http://localhost:8001")
os.environ.setdefault("DISABLE_EMBEDDINGS", "1")
