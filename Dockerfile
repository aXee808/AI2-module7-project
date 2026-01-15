# ===== Stage 1 : build =====
FROM python:3.11-slim AS builder


# create virtual environment
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH

WORKDIR /app
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ===== Stage 2 : runtime =====
FROM python:3.11-slim

WORKDIR /app

# Copier uniquement ce qui est nécessaire
COPY --from=builder /venv /venv
ENV PATH=/venv/bin:$PATH

COPY ./app/app.py .
# comment for docker-compose  
COPY ./app/data/titanic.csv ./data/
# uncomment for docker-compose
#COPY .env .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]