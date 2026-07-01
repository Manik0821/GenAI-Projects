FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VERCEL=0

COPY requirements-cloudrun.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY fastMCP_server /app/fastMCP_server
COPY structured_restaurant_data.json /app/structured_restaurant_data.json

COPY processing-data/augmented_user_review.json /app/processing-data/augmented_user_review.json

WORKDIR /app/fastMCP_server

CMD ["uvicorn", "cloud_api:app", "--host", "0.0.0.0", "--port", "8080"]