FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ config/
COPY src/ src/
COPY scripts/ scripts/

# So we can run python -m src.ingestion.pipeline and python -m src.cli
ENV PYTHONPATH=/app

# Use CMD so docker-compose can override: ingestion runs pipeline, agent runs cli, and
# init_db works: docker compose run --rm ingestion python scripts/init_db.py
CMD ["python", "-m", "src.cli"]
