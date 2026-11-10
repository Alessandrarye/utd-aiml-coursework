FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir pandas numpy scikit-learn xgboost joblib pyyaml fastapi uvicorn pydantic
COPY params.yaml .
COPY src ./src
COPY api ./api
COPY models ./models
# Run as a non-root user. API_KEY must be supplied at runtime, never baked in.
RUN useradd --create-home appuser && mkdir -p monitoring && chown -R appuser /app
USER appuser
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
