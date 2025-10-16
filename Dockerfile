FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system packages required for poetry / build backends.
RUN apt-get update && \
    apt-get install --yes --no-install-recommends curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install project dependencies.
COPY pyproject.toml poetry.lock* /app/
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    /root/.local/bin/poetry config virtualenvs.create false && \
    /root/.local/bin/poetry install --no-root --only main

# Copy the rest of the source.
COPY . /app

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py"]