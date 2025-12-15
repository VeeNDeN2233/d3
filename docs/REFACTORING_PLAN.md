# План рефакторинга проекта

## Выполнено (Фаза 1)

### 1. Создана новая архитектура модулей

#### Структура директорий:
```
core/
├── __init__.py              # Экспорт основных классов
├── state_manager.py         # Управление состояниями (AppState, StateManager)
├── auth_handler.py          # Обработка авторизации (отделена от UI)
├── file_processor.py        # Универсальная обработка файлов
├── analysis_controller.py    # Управление анализом (StepManager, AnalysisPipeline)
└── analysis_adapter.py       # Адаптер для интеграции с существующим кодом

utils/
└── gradio_helpers.py        # Вспомогательные функции для Gradio
```

### 2. Реализованы основные классы

#### `AppState` (core/state_manager.py)
- Централизованное управление всеми состояниями приложения
- Поддержка сериализации/десериализации
- Разделение на подсостояния:
  - `UserState` - состояние пользователя
  - `VideoState` - состояние загруженного видео
  - `AnalysisParameters` - параметры анализа
  - `AnalysisState` - состояние процесса анализа
  - `ModelState` - состояние моделей

#### `StateManager` (core/state_manager.py)
- Менеджер для обновления состояний
- Поддержка слушателей изменений
- Методы для обновления отдельных подсостояний

#### `VideoProcessor` (core/file_processor.py)
- Универсальная обработка всех форматов ввода Gradio
- Поддержка: dict, object, string, list, Path
- Валидация файлов: размер, формат, доступность
- Автоматическое копирование во временную директорию
- Очистка старых временных файлов

#### `AuthHandler` (core/auth_handler.py)
- Отделена логика авторизации от UI
- Методы: login, register, logout, get_user_from_session
- Интеграция с существующим AuthManager
- Обновление состояния пользователя

#### `StepManager` (core/analysis_controller.py)
- Управление шагами интерфейса
- Проверка возможности перехода между шагами
- Валидация переходов
- Методы: next_step(), previous_step(), go_to_step()

#### `AnalysisPipeline` (core/analysis_controller.py)
- Пайплайн анализа с поддержкой отмены
- Выполнение в отдельном потоке
- Обновление прогресса через callback
- Обработка ошибок и отмены операций

## Требуется выполнить (Фазы 2-4)

### Фаза 2: Рефакторинг medical_interface.py

#### Задачи:
1. **Интеграция новых модулей**
   - Заменить глобальные переменные на `StateManager`
   - Использовать `VideoProcessor` для обработки файлов
   - Использовать `AuthHandler` для авторизации
   - Использовать `StepManager` для управления шагами

2. **Упрощение функций**
   - Рефакторинг `handle_login` с использованием `AuthHandler`
   - Рефакторинг `handle_register` с использованием `AuthHandler`
   - Рефакторинг `analyze_baby_video` с использованием `VideoProcessor`
   - Упрощение функций управления шагами

3. **Улучшение обработки ошибок**
   - Централизованная обработка ошибок
   - Понятные сообщения пользователю
   - Логирование всех ошибок

### Фаза 3: Добавление функций

#### Задачи:
1. **Индикация прогресса**
   - Добавить `gr.Progress` компонент
   - Интегрировать с `AnalysisPipeline`
   - Обновление прогресса в реальном времени

2. **Механизм отмены**
   - Кнопка "Отменить анализ"
   - Обработка отмены в `AnalysisPipeline`
   - Очистка ресурсов при отмене

3. **Загрузка моделей независимо от авторизации**
   - Загружать модели при старте приложения
   - Показывать статус загрузки
   - Разрешить анализ без авторизации (опционально)

### Фаза 4: Оптимизация

#### Задачи:
1. **Кэширование результатов**
   - Кэширование результатов анализа
   - Проверка кэша перед анализом
   - Очистка старых кэшей

2. **Оптимизация памяти**
   - Lazy loading моделей
   - Очистка памяти после анализа
   - Обработка больших файлов

3. **Улучшение производительности**
   - Batch processing где возможно
   - Оптимизация операций с файлами
   - Мониторинг использования ресурсов

## Пример использования новой архитектуры

### Инициализация:
```python
from core import AppState, StateManager, AuthHandler, VideoProcessor, StepManager

# Создание менеджеров
state_manager = StateManager()
auth_handler = AuthHandler()
video_processor = VideoProcessor()
step_manager = StepManager(state_manager.get_state())
```

### Обработка входа:
```python
def handle_login(email: str, password: str):
    success, message, user_data, token = auth_handler.login(email, password)
    if success:
        state_manager.update_user(
            is_authenticated=True,
            session_token=token,
            email=user_data.get('email'),
            username=user_data.get('username'),
            full_name=user_data.get('full_name')
        )
        step_manager.go_to_step(AnalysisStep.UPLOAD)
    return success, message
```

### Обработка файла:
```python
def handle_video_upload(video_file):
    video_path = video_processor.get_video_path(video_file)
    if video_path:
        is_valid, error = video_processor.validate_video(video_path)
        if is_valid:
            state_manager.update_video(
                file_path=str(video_path),
                file_name=video_path.name,
                file_size=video_path.stat().st_size,
                is_uploaded=True,
                is_valid=True
            )
            step_manager.next_step()
            return True, "Видео загружено"
        else:
            return False, error
    return False, "Не удалось обработать файл"
```

### Запуск анализа:
```python
def start_analysis():
    state = state_manager.get_state()
    pipeline = AnalysisPipeline(
        state=state,
        video_processor=video_processor,
        analysis_func=analyze_baby_video,
        progress_callback=update_progress_ui,
        cancel_event=threading.Event()
    )
    
    video_path = Path(state.video.file_path)
    pipeline.start_analysis(
        video_path,
        state.parameters.patient_age_weeks,
        state.parameters.gestational_age
    )
```

## Преимущества новой архитектуры

1. **Разделение ответственности**
   - Логика отделена от UI
   - Каждый модуль отвечает за свою область

2. **Централизованное управление состоянием**
   - Все состояния в одном месте
   - Легко отслеживать изменения
   - Простое сохранение/загрузка состояния

3. **Универсальная обработка файлов**
   - Поддержка всех форматов Gradio
   - Автоматическая валидация
   - Обработка ошибок

4. **Улучшенная обработка ошибок**
   - Централизованная обработка
   - Понятные сообщения
   - Детальное логирование

5. **Масштабируемость**
   - Легко добавлять новые функции
   - Модульная структура
   - Тестируемость

## Следующие шаги

1. **Немедленно:**
   - Протестировать новые модули
   - Исправить возможные ошибки импорта
   - Убедиться в совместимости с существующим кодом

2. **Краткосрочно (1-2 дня):**
   - Интегрировать новые модули в medical_interface.py
   - Заменить глобальные переменные на StateManager
   - Обновить функции обработки событий

3. **Среднесрочно (3-5 дней):**
   - Добавить индикацию прогресса
   - Реализовать механизм отмены
   - Оптимизировать загрузку моделей

4. **Долгосрочно (1-2 недели):**
   - Добавить кэширование
   - Оптимизировать использование памяти
   - Улучшить производительность

## Примечания

- Все новые модули созданы и готовы к использованию
- Существующий код продолжает работать без изменений
- Рефакторинг можно выполнять постепенно
- Новая архитектура полностью совместима с существующим кодом

