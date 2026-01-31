FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY requirements-embeddings.txt /app/requirements-embeddings.txt

RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir -r /app/requirements-embeddings.txt

COPY src /app/src
ENV PYTHONPATH=/app
CMD ["python", "-m", "src.app"]

