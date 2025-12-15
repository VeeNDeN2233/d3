# Инструкция по использованию MMPOSE Video Processing

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Запуск сервера

```bash
python app.py
```

### 3. Открыть в браузере

```
http://localhost:8000
```

## Обработка видео baby.mp4

1. **Загрузите видео** через веб-интерфейс (drag & drop или кнопка "Выбрать файл")
2. **Нажмите "Обработать видео"**
3. **Дождитесь завершения** обработки (может занять время в зависимости от длины видео)
4. **Скачайте результаты**:
   - Обработанное видео с визуализацией позы
   - Метки движения в формате ZIP (совместимо с MINI-RGBD)

## Использование меток для обучения модели

### Структура сохраненных меток

После обработки в директории `results/{file_id}_processed/keypoints/` будут сохранены:

- **keypoints.json** - все метки в JSON формате
- **joints_2Ddep/** - 2D метки с depth (формат MINI-RGBD)
- **joints_3D/** - 3D метки (нормализованные координаты)
- **jointlist.txt** - список суставов

### Формат меток MINI-RGBD

Каждый файл в `joints_2Ddep/` содержит 25 строк (по одной на сустав):
```
x y depth joint_id
```

Пример:
```
288.47 260.70 881.00 0
304.63 272.64 881.00 1
...
```

### Сравнение с нормой из датасета MINI-RGBD

Метки сохранены в том же формате, что и датасет MINI-RGBD, поэтому их можно использовать напрямую:

1. **Загрузите метки** из `results/{file_id}_processed/keypoints/joints_2Ddep/`
2. **Сравните с нормой** из `MINI-RGBD_web/XX/joints_2Ddep/`
3. **Обучите модель** для детекции отклонений от нормы

### Пример использования меток в Python

```python
import numpy as np
from pathlib import Path

# Загрузка меток из обработанного видео
video_keypoints_dir = Path("results/{file_id}_processed/keypoints/joints_2Ddep")
video_keypoints = []
for frame_file in sorted(video_keypoints_dir.glob("*.txt")):
    frame_data = np.loadtxt(frame_file)
    video_keypoints.append(frame_data)

# Загрузка нормы из датасета MINI-RGBD
norm_keypoints_dir = Path("MINI-RGBD_web/01/joints_2Ddep")
norm_keypoints = []
for frame_file in sorted(norm_keypoints_dir.glob("*.txt")):
    frame_data = np.loadtxt(frame_file)
    norm_keypoints.append(frame_data)

# Сравнение меток
# Здесь можно добавить логику сравнения и обучения модели
```

## API Endpoints

### POST /process
Загрузка и обработка видео

### GET /results/{filename}
Скачивание обработанного видео

### GET /keypoints/{file_id}
Скачивание всех меток в формате ZIP

### GET /keypoints/{file_id}/json
Получение меток в формате JSON

## Примечания

- Метки автоматически конвертируются из формата COCO (17 точек) в формат MINI-RGBD (25 суставов)
- Для суставов, отсутствующих в COCO, используется интерполяция и экстраполяция
- Depth значения в 2D метках установлены в 0 (MMPOSE работает только с 2D)
- Для получения реальных depth значений потребуется дополнительная обработка с использованием RGB-D данных

