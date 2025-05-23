FROM python:3.9

WORKDIR /app

# 1) System-Dependencies installieren
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1-mesa-glx \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libjpeg-dev \
      libpng-dev \
      libgeos-dev \
      gcc \
      python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2) Python-Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3) gesamten Code kopieren
COPY src/ src/
COPY scripts/ scripts/
COPY config/ config/

# 4) Damit Python-Module aus 'src' importierbar sind
ENV PYTHONPATH=/app/src

# 5) Default-Entrypoint
CMD ["bash"]