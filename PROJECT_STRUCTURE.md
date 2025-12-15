# Структура проекта

## Основные директории

```
d3/
├── models/              # Модели нейросетей
│   ├── autoencoder_advanced.py    # Улучшенная модель (Bidirectional LSTM + Attention)
│   ├── autoencoder_gpu.py         # Базовая модель
│   └── anomaly_detector.py        # Детектор аномалий
│
├── utils/               # Утилиты
│   ├── data_loader.py            # Загрузка данных MINI-RGBD
│   ├── pose_processor.py         # Обработка поз
│   └── data_augmentation.py      # Аугментация данных
│
├── checkpoints/         # Сохраненные модели (игнорируется в git)
│   └── .gitkeep                  # Сохраняет структуру папки
│
├── docs/                # Документация
│   ├── USAGE.md                  # Инструкция по использованию
│   ├── GIT_SETUP.md              # Настройка Git
│   └── archive/                   # Архивная документация
│
├── results/             # Результаты анализа (игнорируется в git)
├── test_videos/         # Тестовые видео (игнорируется в git)
├── uploads/            # Загруженные файлы (игнорируется в git)
│
├── medical_interface.py  # Веб-интерфейс (Gradio)
├── inference_advanced.py # Инференс улучшенной модели
├── train_advanced.py     # Обучение улучшенной модели
├── train_gpu.py         # Обучение базовой модели
├── video_processor.py   # Обработка видео (MediaPipe)
│
├── config.yaml          # Конфигурация
├── requirements.txt     # Зависимости
└── README.md            # Основная документация
```

## Игнорируемые элементы (см. .gitignore)

- `MINI-RGBD_web/` - датасет
- `checkpoints/*.pt` - модели
- `results/` - результаты анализа
- `test_videos/` - тестовые видео
- `uploads/` - загруженные файлы
- `__pycache__/` - кэш Python
- `.venv/`, `venv/` - виртуальные окружения

