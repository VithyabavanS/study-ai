FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ANONYMIZED_TELEMETRY=False \
    HF_HOME=/home/app/.cache/huggingface \
    PORT=8501

# Run as a normal user, not root
RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/study-ai

# CPU-only PyTorch first: the default build pulls several GB of GPU libraries
RUN pip install --upgrade pip \
 && pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app src ./src
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app .streamlit ./.streamlit

USER app
# Bake the models into the image so the first question doesn't wait for a download
RUN PYTHONPATH=src python scripts/download_models.py

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:%s/_stcore/health' % os.environ['PORT'])"
CMD ["sh", "-c", "streamlit run src/app.py --server.address=0.0.0.0 --server.port=${PORT}"]