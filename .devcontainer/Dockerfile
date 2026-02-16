# Dev container: same Python and deps as prod, plus dev tools.
# Build context is repo root (see docker-compose.yml).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/kb_engine_playground

# Install app dependencies (source is mounted over this dir)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/workspaces/kb_engine_playground

# Keep container running; workspace is mounted by docker-compose
CMD ["sleep", "infinity"]
