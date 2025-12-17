# Архитектура системы детекции аномалий движений младенцев

## Диаграмма архитектуры системы

```mermaid
graph TB
    %% Пользовательский интерфейс
    subgraph UI["🌐 Пользовательский интерфейс (Gradio)"]
        WEB["medical_interface.py<br/>Веб-интерфейс"]
        LOGIN["Страница входа"]
        MAIN["Главная страница<br/>4 шага анализа"]
    end

    %% Аутентификация
    subgraph AUTH["🔐 Система аутентификации"]
        AUTH_MGR["auth_manager.py<br/>Менеджер аутентификации"]
        AUTH_HANDLER["auth_handler.py<br/>Обработчик аутентификации"]
        DB["users.db<br/>База данных пользователей"]
    end

    %% Ядро системы
    subgraph CORE["⚙️ Ядро системы (Core)"]
        STATE_MGR["state_manager.py<br/>Управление состоянием"]
        FILE_PROC["file_processor.py<br/>Обработка файлов"]
        ANALYSIS_CTRL["analysis_controller.py<br/>Контроллер анализа"]
        STEP_MGR["StepManager<br/>Управление шагами"]
        PIPELINE["AnalysisPipeline<br/>Пайплайн анализа"]
    end

    %% Обработка видео
    subgraph VIDEO["🎥 Обработка видео"]
        VIDEO_PROC["video_processor.py<br/>MediaPipe Pose"]
        POSE_PROC["pose_processor.py<br/>Обработка поз"]
        KEYPOINTS["Извлечение ключевых точек<br/>33 точки → 25 суставов"]
    end

    %% Модели машинного обучения
    subgraph ML["🤖 Модели машинного обучения"]
        AUTOENCODER["autoencoder_advanced.py<br/>Bidirectional LSTM + Attention"]
        ANOMALY_DET["anomaly_detector.py<br/>Детектор аномалий"]
        MODEL_CACHE["model_cache.py<br/>Кэш моделей"]
    end

    %% Утилиты
    subgraph UTILS["🛠️ Утилиты"]
        DATA_LOADER["data_loader.py<br/>Загрузка данных MINI-RGBD"]
        DATA_AUG["data_augmentation.py<br/>Аугментация данных"]
        NORMAL_STATS["normal_statistics.py<br/>Статистика нормальных движений"]
        ANOMALY_ANALYZER["anomaly_analyzer.py<br/>Анализ аномалий"]
        VIDEO_VIZ["video_visualizer.py<br/>Визуализация видео"]
        ANALYSIS_CACHE["analysis_cache.py<br/>Кэш результатов"]
        PERF_OPT["performance_optimizer.py<br/>Оптимизация производительности"]
    end

    %% Инференс и отчеты
    subgraph INFERENCE["📊 Инференс и отчеты"]
        INF_ADV["inference_advanced.py<br/>Инференс модели"]
        REPORT["Генерация медицинского отчета<br/>GMA оценка"]
        VISUALIZATION["Визуализация результатов<br/>Графики и видео"]
    end

    %% Обучение
    subgraph TRAINING["🎓 Обучение моделей"]
        TRAIN_ADV["train_advanced.py<br/>Обучение улучшенной модели"]
        TRAIN_GPU["train_gpu.py<br/>Обучение базовой модели"]
        CHECKPOINTS["checkpoints/<br/>Сохраненные модели"]
    end

    %% Конфигурация
    CONFIG["config.yaml<br/>Конфигурация системы"]

    %% Потоки данных
    WEB --> LOGIN
    LOGIN --> AUTH_MGR
    AUTH_MGR --> DB
    AUTH_MGR --> AUTH_HANDLER
    AUTH_HANDLER --> STATE_MGR

    WEB --> MAIN
    MAIN --> STATE_MGR
    STATE_MGR --> STEP_MGR
    STEP_MGR --> ANALYSIS_CTRL
    ANALYSIS_CTRL --> PIPELINE

    MAIN --> FILE_PROC
    FILE_PROC --> VIDEO_PROC
    VIDEO_PROC --> KEYPOINTS
    KEYPOINTS --> POSE_PROC
    POSE_PROC --> PIPELINE

    PIPELINE --> AUTOENCODER
    AUTOENCODER --> ANOMALY_DET
    MODEL_CACHE --> AUTOENCODER
    MODEL_CACHE --> ANOMALY_DET

    ANOMALY_DET --> INF_ADV
    INF_ADV --> ANOMALY_ANALYZER
    ANOMALY_ANALYZER --> NORMAL_STATS
    INF_ADV --> REPORT
    INF_ADV --> VISUALIZATION
    VISUALIZATION --> VIDEO_VIZ

    TRAIN_ADV --> DATA_LOADER
    DATA_LOADER --> DATA_AUG
    DATA_AUG --> AUTOENCODER
    AUTOENCODER --> CHECKPOINTS
    ANOMALY_DET --> CHECKPOINTS

    CONFIG --> TRAIN_ADV
    CONFIG --> INF_ADV
    CONFIG --> VIDEO_PROC
    CONFIG --> POSE_PROC

    ANALYSIS_CACHE --> INF_ADV
    PERF_OPT --> PIPELINE

    %% Стили
    classDef uiClass fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef authClass fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef coreClass fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef videoClass fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef mlClass fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef utilsClass fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef infClass fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    classDef trainClass fill:#f1f8e9,stroke:#33691e,stroke-width:2px

    class WEB,LOGIN,MAIN uiClass
    class AUTH_MGR,AUTH_HANDLER,DB authClass
    class STATE_MGR,FILE_PROC,ANALYSIS_CTRL,STEP_MGR,PIPELINE coreClass
    class VIDEO_PROC,POSE_PROC,KEYPOINTS videoClass
    class AUTOENCODER,ANOMALY_DET,MODEL_CACHE mlClass
    class DATA_LOADER,DATA_AUG,NORMAL_STATS,ANOMALY_ANALYZER,VIDEO_VIZ,ANALYSIS_CACHE,PERF_OPT utilsClass
    class INF_ADV,REPORT,VISUALIZATION infClass
    class TRAIN_ADV,TRAIN_GPU,CHECKPOINTS trainClass
```

## Поток обработки видео

```mermaid
sequenceDiagram
    participant User as 👤 Пользователь
    participant UI as 🌐 Gradio Interface
    participant Auth as 🔐 Auth Handler
    participant State as 📊 State Manager
    participant Video as 🎥 Video Processor
    participant MediaPipe as MediaPipe Pose
    participant Pose as Pose Processor
    participant Model as 🤖 Autoencoder
    participant Detector as 🔍 Anomaly Detector
    participant Inference as 📊 Inference Engine
    participant Report as 📄 Report Generator

    User->>UI: Загрузка видео
    UI->>Auth: Проверка аутентификации
    Auth-->>UI: Сессия валидна
    
    UI->>State: Обновление состояния (шаг 1)
    State-->>UI: Состояние обновлено
    
    User->>UI: Указание параметров (возраст)
    UI->>State: Обновление параметров (шаг 2)
    
    User->>UI: Запуск анализа
    UI->>State: Переход к шагу анализа (шаг 3)
    
    State->>Video: Обработка видео
    Video->>MediaPipe: Извлечение позы (33 точки)
    MediaPipe-->>Video: Ключевые точки
    
    Video->>Pose: Преобразование в MINI-RGBD (25 суставов)
    Pose->>Pose: Нормализация и создание последовательностей
    Pose-->>Video: Последовательности движений
    
    Video->>Model: Подача последовательностей
    Model->>Model: Реконструкция через Bidirectional LSTM
    Model-->>Video: Реконструированные последовательности
    
    Video->>Detector: Вычисление ошибок реконструкции
    Detector->>Detector: Сравнение с порогом аномалии
    Detector-->>Video: Флаги аномалий
    
    Video->>Inference: Результаты анализа
    Inference->>Report: Генерация медицинского отчета
    Report->>Report: GMA оценка и рекомендации
    Report-->>Inference: Готовый отчет
    
    Inference->>State: Сохранение результатов (шаг 4)
    State-->>UI: Обновление UI с результатами
    UI-->>User: Отображение результатов
```

## Архитектура модели

```mermaid
graph LR
    subgraph INPUT["Входные данные"]
        VIDEO["RGB Видео<br/>Младенец на спине"]
    end

    subgraph EXTRACTION["Извлечение признаков"]
        MP["MediaPipe Pose<br/>33 ключевые точки"]
        CONV["Преобразование<br/>33 → 25 суставов<br/>MINI-RGBD формат"]
        NORM["Нормализация<br/>- Относительно торса<br/>- Bounding box<br/>- Каноническая ориентация"]
        SEQ["Создание последовательностей<br/>Длина: 30 кадров<br/>Шаг: 1 кадр"]
    end

    subgraph ENCODER["Энкодер"]
        BI_LSTM1["Bidirectional LSTM<br/>256 hidden units"]
        BI_LSTM2["Bidirectional LSTM<br/>128 hidden units"]
        BI_LSTM3["Bidirectional LSTM<br/>64 hidden units"]
        ATT["Multi-head Attention<br/>4 головы"]
        LATENT["Латентное представление<br/>64 размерности"]
    end

    subgraph DECODER["Декодер"]
        LSTM1["LSTM<br/>64 → 128"]
        LSTM2["LSTM<br/>128 → 256"]
        LSTM3["LSTM<br/>256 → 75"]
        OUTPUT["Реконструированная<br/>последовательность<br/>75 размерностей"]
    end

    subgraph DETECTION["Детекция аномалий"]
        MSE["MSE Loss<br/>Ошибка реконструкции"]
        THRESHOLD["Порог аномалии<br/>95-й перцентиль<br/>на validation данных"]
        ANOMALY["Флаг аномалии<br/>True/False"]
    end

    VIDEO --> MP
    MP --> CONV
    CONV --> NORM
    NORM --> SEQ
    
    SEQ --> BI_LSTM1
    BI_LSTM1 --> BI_LSTM2
    BI_LSTM2 --> BI_LSTM3
    BI_LSTM3 --> ATT
    ATT --> LATENT
    
    LATENT --> LSTM1
    LSTM1 --> LSTM2
    LSTM2 --> LSTM3
    LSTM3 --> OUTPUT
    
    SEQ --> MSE
    OUTPUT --> MSE
    MSE --> THRESHOLD
    THRESHOLD --> ANOMALY

    style INPUT fill:#e3f2fd
    style EXTRACTION fill:#f1f8e9
    style ENCODER fill:#fff3e0
    style DECODER fill:#fce4ec
    style DETECTION fill:#ffebee
```

## Структура данных

```mermaid
graph TB
    subgraph VIDEO_DATA["Видео данные"]
        RGB["RGB Видео<br/>MP4/AVI/MOV"]
    end

    subgraph KEYPOINTS["Ключевые точки"]
        MP_33["MediaPipe<br/>33 точки<br/>x, y, z, visibility"]
        MINI_25["MINI-RGBD<br/>25 суставов<br/>x, y, z"]
    end

    subgraph SEQUENCES["Последовательности"]
        SEQ_ARRAY["Массив последовательностей<br/>Shape: N × 30 × 75<br/>N - количество последовательностей<br/>30 - длина последовательности<br/>75 - 25 суставов × 3 координаты"]
    end

    subgraph MODEL_OUT["Выход модели"]
        RECONSTR["Реконструкция<br/>Shape: N × 30 × 75"]
        ERROR["Ошибка реконструкции<br/>Shape: N<br/>MSE per sequence"]
        ANOMALY_FLAG["Флаг аномалии<br/>Shape: N<br/>Boolean array"]
    end

    subgraph RESULTS["Результаты анализа"]
        STATS["Статистика<br/>- Mean error<br/>- Anomaly rate<br/>- Risk level"]
        REPORT_DATA["Медицинский отчет<br/>- GMA оценка<br/>- Рекомендации<br/>- Детальный анализ"]
        VISUAL["Визуализация<br/>- График ошибок<br/>- Видео с скелетом<br/>- Heatmap аномалий"]
    end

    RGB --> MP_33
    MP_33 --> MINI_25
    MINI_25 --> SEQ_ARRAY
    SEQ_ARRAY --> RECONSTR
    SEQ_ARRAY --> ERROR
    RECONSTR --> ERROR
    ERROR --> ANOMALY_FLAG
    ERROR --> STATS
    ANOMALY_FLAG --> STATS
    STATS --> REPORT_DATA
    ERROR --> VISUAL
    ANOMALY_FLAG --> VISUAL

    style VIDEO_DATA fill:#e1f5ff
    style KEYPOINTS fill:#f3e5f5
    style SEQUENCES fill:#e8f5e9
    style MODEL_OUT fill:#fff3e0
    style RESULTS fill:#fce4ec
```

## Компоненты системы

### 1. Пользовательский интерфейс (UI)
- **medical_interface.py**: Gradio веб-интерфейс
- **4 шага анализа**: Загрузка → Параметры → Анализ → Результаты
- **Аутентификация**: Вход/регистрация пользователей

### 2. Система аутентификации
- **auth_manager.py**: Управление сессиями и пользователями
- **auth_handler.py**: Обработка запросов аутентификации
- **users.db**: SQLite база данных пользователей

### 3. Ядро системы (Core)
- **state_manager.py**: Централизованное управление состоянием приложения
- **file_processor.py**: Универсальная обработка файлов из Gradio
- **analysis_controller.py**: Контроллер процесса анализа
- **StepManager**: Управление шагами интерфейса
- **AnalysisPipeline**: Пайплайн анализа с поддержкой отмены

### 4. Обработка видео
- **video_processor.py**: Обработка видео через MediaPipe Pose
- **pose_processor.py**: Преобразование поз в формат MINI-RGBD
- Извлечение 33 ключевых точек → преобразование в 25 суставов
- Нормализация и создание последовательностей

### 5. Модели машинного обучения
- **autoencoder_advanced.py**: Bidirectional LSTM + Attention автоэнкодер
- **anomaly_detector.py**: Детектор аномалий на основе ошибки реконструкции
- **model_cache.py**: Кэширование загруженных моделей

### 6. Утилиты
- **data_loader.py**: Загрузка данных из датасета MINI-RGBD
- **data_augmentation.py**: Аугментация данных для обучения
- **normal_statistics.py**: Статистика нормальных движений
- **anomaly_analyzer.py**: Детальный анализ аномалий
- **video_visualizer.py**: Визуализация результатов на видео
- **analysis_cache.py**: Кэширование результатов анализа
- **performance_optimizer.py**: Оптимизация производительности

### 7. Инференс и отчеты
- **inference_advanced.py**: Инференс улучшенной модели
- Генерация медицинских отчетов в формате GMA
- Визуализация результатов (графики, видео с скелетом)

### 8. Обучение моделей
- **train_advanced.py**: Обучение Bidirectional LSTM + Attention модели
- **train_gpu.py**: Обучение базовой модели
- **checkpoints/**: Сохраненные модели и детекторы

## Технологический стек

- **Python 3.8+**
- **PyTorch**: Глубокое обучение
- **MediaPipe**: Извлечение позы
- **Gradio**: Веб-интерфейс
- **SQLite**: База данных пользователей
- **OpenCV**: Обработка видео
- **NumPy**: Вычисления
- **Matplotlib**: Визуализация

## Поток данных

1. **Загрузка видео** → Валидация → Сохранение во временную директорию
2. **Извлечение позы** → MediaPipe Pose → 33 ключевые точки
3. **Преобразование** → MINI-RGBD формат → 25 суставов
4. **Нормализация** → Относительно торса, bounding box, каноническая ориентация
5. **Создание последовательностей** → Длина 30 кадров, шаг 1 кадр
6. **Реконструкция** → Bidirectional LSTM автоэнкодер
7. **Детекция аномалий** → Сравнение ошибки реконструкции с порогом
8. **Генерация отчета** → GMA оценка, статистика, рекомендации
9. **Визуализация** → Графики, видео с наложенным скелетом

## Особенности архитектуры

- ✅ **Модульность**: Четкое разделение компонентов
- ✅ **Управление состоянием**: Централизованное состояние через StateManager
- ✅ **Кэширование**: Кэш моделей и результатов анализа
- ✅ **Оптимизация**: Оптимизация памяти и производительности
- ✅ **Отмена операций**: Поддержка отмены длительных операций
- ✅ **Аутентификация**: Система пользователей и сессий
- ✅ **Валидация**: Проверка входных данных на всех этапах
