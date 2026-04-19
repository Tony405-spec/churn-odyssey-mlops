FROM nvidia/cuda:12.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.10 python3-pip curl build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY pyproject.toml* poetry.lock* /app/
RUN poetry config virtualenvs.create false && poetry install --no-dev

COPY src/ /app/src/
COPY churn_odyssey/ /app/churn_odyssey/
COPY requirements.txt /app/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
