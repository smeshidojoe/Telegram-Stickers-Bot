FROM python:3.13-slim

# ffmpeg тянет и ffprobe — нужны оба.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN useradd --create-home --uid 10001 bot

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Бот пишет рабочие файлы в /app/tmp, а работает не от root.
RUN mkdir -p tmp && chown -R bot:bot /app

USER bot

# У long polling нет эндпоинта, поэтому смотрим на свежесть heartbeat:
# он перестаёт обновляться, если цикл событий встал. Живёт в Dockerfile,
# а не в compose, чтобы работало и при обычном docker run.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os,sys,time;p='/app/tmp/heartbeat';sys.exit(0 if os.path.exists(p) and time.time()-os.path.getmtime(p)<120 else 1)"]

# Токен приходит переменной окружения: --env-file .env или -e BOT_TOKEN=...
# В образ .env не кладём, иначе секрет останется в слоях.
CMD ["python", "main.py"]
