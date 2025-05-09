# Dockerfile
FROM python:3.9

WORKDIR /app

# 1) System-Dependencies installieren
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libgl1-mesa-glx \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      gcc \
      python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2) Python-Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3) gesamten Code kopieren
COPY . .

# 4) Damit Python-Module aus 'src' importierbar sind
ENV PYTHONPATH=/app/src

# 5) Default-Entrypoint: nutzt jetzt den neuen Package-Pfad
CMD ["python", "-m", "oft.examples.visualize"]

