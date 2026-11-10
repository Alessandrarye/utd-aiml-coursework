# Churn prediction API. Build from the project root after training a model:
#   docker build -t churn-api .
#   docker run --rm -p 8000:8000 churn-api
# PYTHON_VERSION should match the Python the model was trained with
# (see "python_version" in models/metadata.json).
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# serving-requirements.txt is written by src/train.py and pins the exact
# scikit-learn / pandas / numpy versions the saved model was trained with.
COPY models/serving-requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/churn_model.joblib models/metadata.json ./models/

RUN useradd --create-home appuser && mkdir -p /app/logs && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
