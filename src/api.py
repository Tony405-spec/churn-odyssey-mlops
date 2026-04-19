from __future__ import annotations

from churn_odyssey.api import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000)
