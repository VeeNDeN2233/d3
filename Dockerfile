# Используем официальный образ Python с CUDA поддержкой
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Устанавливаем Python 3.11
# Повторные попытки для обработки временных проблем с зеркалами Ubuntu
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update --fix-missing || apt-get update --fix-missing || apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Создаем символическую ссылку для python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем PyTorch с CUDA поддержкой
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

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
