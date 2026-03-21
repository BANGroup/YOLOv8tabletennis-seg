# YOLOv8 Table Tennis Segmentation Docker Image
# GPU-enabled для CUDA inference

FROM python:3.11-slim

LABEL maintainer="Alexey"
LABEL description="YOLOv8 Table Tennis Ball Tracking & Segmentation"
LABEL version="1.0"

# Переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Рабочая директория
WORKDIR /app

# Системные зависимости для OpenCV и видео
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements первым для кэширования
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

# Создаём директории
RUN mkdir -p /app/sources /app/results/experimental /app/scripts/config

# Точка входа
ENTRYPOINT ["python"]
CMD ["scripts/main_processor.py", "--help"]

