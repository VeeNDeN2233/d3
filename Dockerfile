# Используем официальный образ с CUDA поддержкой
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Устанавливаем Python 3.11 и системные зависимости (объединено для оптимизации слоев)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && ln -s /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости (объединено для оптимизации)
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.7.1+cu118 torchvision==0.22.1+cu118

# Копируем весь проект
COPY . .

# Создаем необходимые директории
RUN mkdir -p uploads results checkpoints logs

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=run_flask.py
ENV FLASK_ENV=production
ENV PORT=5000

# Открываем порт
EXPOSE 5000

# Команда запуска
CMD ["python", "run_flask.py"]
