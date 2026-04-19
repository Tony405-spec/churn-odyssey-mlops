import json
from pathlib import Path


def main():
    summary = json.loads(Path("artifacts/train_summary.json").read_text())
    Path("artifacts/evaluation.json").write_text(json.dumps({"status": "ok", "summary": summary}))


if __name__ == "__main__":
    main()
