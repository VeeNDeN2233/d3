# Используем официальный образ с CUDA поддержкой
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Устанавливаем переменные окружения для оптимизации apt
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=run_flask.py \
    FLASK_ENV=production \
    PORT=5000 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Устанавливаем Python 3.11 и системные зависимости одной командой (минимизация слоев)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        ffmpeg \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Обновляем pip (кэшируемый слой - меняется редко)
RUN python -m pip install --upgrade pip setuptools wheel

# Копируем только requirements.txt для лучшего кэширования слоев
COPY requirements.txt .

# Устанавливаем Python зависимости (используем BuildKit cache mount для ускорения)
# Сначала устанавливаем зависимости из requirements.txt, затем PyTorch
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
        torch==2.7.1+cu118 \
        torchvision==0.22.1+cu118

# Копируем остальные файлы проекта (меняются чаще всего - отдельный слой)
COPY . .

# Создаем необходимые директории одной командой
RUN mkdir -p uploads results checkpoints logs && \
    chmod 755 uploads results checkpoints logs

# Открываем порт
EXPOSE 5000

# Команда запуска
CMD ["python", "run_flask.py"]
