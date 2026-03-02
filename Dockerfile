FROM python:3.10-slim

WORKDIR /app

COPY model.joblib .
COPY app.py .
COPY pyproject.toml .

RUN pip install --no-cache-dir fastapi[standard] joblib scikit-learn pandas

EXPOSE 8080

CMD ["fastapi", "run", "app.py", "--port", "8080"]