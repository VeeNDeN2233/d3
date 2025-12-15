# Процесс извлечения ключевых точек из видео

## 📹 Полный пайплайн обработки видео

### Шаг 1: Обработка видео через MediaPipe (`video_processor.py`)

```python
# В методе _process_video_sync():

# 1. Открываем видео
cap = cv2.VideoCapture(input_path)

# 2. Для каждого кадра:
while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break
    
    # 3. Конвертируем BGR → RGB (MediaPipe требует RGB)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # 4. Обрабатываем через MediaPipe Pose
    results = self._pose.process(frame_rgb)
    # results.pose_landmarks - это объект с 33 точками MediaPipe
    
    # 5. Сохраняем landmarks в список
    if save_keypoints:
        all_keypoints.append({
            "frame": frames_processed,
            "landmarks": self._landmarks_to_list(results.pose_landmarks),
            "mini_rgbd": mini_rgbd_joints,
        })
```

### Шаг 2: Конвертация landmarks в список (`_landmarks_to_list`)

```python
def _landmarks_to_list(self, pose_landmarks) -> Optional[List[Dict[str, float]]]:
    """Конвертирует MediaPipe landmarks в список словарей."""
    if pose_landmarks is None:
        return None
    
    out = []
    for lm in pose_landmarks.landmark:
        out.append({
            "x": float(lm.x),      # Нормализованная координата [0, 1]
            "y": float(lm.y),      # Нормализованная координата [0, 1]
            "z": float(lm.z),      # Относительная глубина
            "visibility": float(getattr(lm, "visibility", 0.0)),  # Уверенность [0, 1]
        })
    return out
```

**Важно:** MediaPipe возвращает координаты в **нормализованном формате [0, 1]**, где:
- `x = 0.0` - левый край изображения
- `x = 1.0` - правый край изображения
- `y = 0.0` - верх изображения
- `y = 1.0` - низ изображения

### Шаг 3: Сохранение в JSON (`_save_keypoints`)

```python
# Сохраняем в keypoints/keypoints.json:
{
    "format": "mini_rgbd",
    "source": "mediapipe_pose",
    "joints": 25,
    "video": {
        "width": 856,
        "height": 472,
        "fps": 29.92,
        "frames": 91
    },
    "frames": [
        {
            "frame": 0,
            "landmarks": [
                {"x": 0.7965, "y": 0.4762, "z": -0.1234, "visibility": 0.98},
                ...
            ],
            "mini_rgbd": [...]
        },
        ...
    ]
}
```

### Шаг 4: Загрузка keypoints для анализа (`inference_advanced.py`)

```python
# Загружаем из JSON
keypoints_path = Path(result["keypoints_path"]) / "keypoints.json"
with open(keypoints_path, "r") as f:
    keypoints_data = json.load(f)

# Конвертируем обратно в numpy массивы
keypoints_list = []
for frame_data in keypoints_data["frames"]:
    landmarks = frame_data.get("landmarks")
    if landmarks:
        kp = np.array(
            [[lm["x"], lm["y"], lm["z"], lm.get("visibility", 0.0)] 
             for lm in landmarks],
            dtype=np.float32,
        )
        keypoints_list.append(kp)  # (33, 4) - 33 точки × 4 значения
    else:
        keypoints_list.append(None)
```

### Шаг 5: Использование keypoints для визуализации

Теперь `keypoints_list` содержит:
- **Формат:** список numpy массивов формы `(33, 4)`
- **Координаты:** нормализованные [0, 1] из MediaPipe
- **Структура:** `[x, y, z, visibility]` для каждой из 33 точек

Для визуализации нужно:
1. Преобразовать нормализованные координаты в пиксели: `x_pixel = x_norm * width`
2. Нарисовать точки и соединения через OpenCV

## 🔄 Полный поток данных

```
Видео (baby.mp4)
    ↓
VideoProcessor._process_video_sync()
    ↓
MediaPipe Pose.process(frame_rgb)
    ↓
results.pose_landmarks (33 точки, нормализованные [0, 1])
    ↓
_landmarks_to_list() → List[Dict[x, y, z, visibility]]
    ↓
_save_keypoints() → keypoints/keypoints.json
    ↓
inference_advanced.process_video()
    ↓
Загрузка из JSON → keypoints_list: List[np.ndarray(33, 4)]
    ↓
create_skeleton_video_from_processed()
    ↓
Преобразование [0, 1] → пиксели → рисование через OpenCV
    ↓
Видео с наложенным скелетом
```

## 📊 Формат данных на каждом этапе

1. **MediaPipe результат:** `results.pose_landmarks` - protobuf объект с 33 landmarks
2. **JSON сохранение:** `{"x": 0.7965, "y": 0.4762, "z": -0.1234, "visibility": 0.98}`
3. **Numpy массив:** `np.array([[x, y, z, visibility], ...])` форма `(33, 4)`
4. **Визуализация:** Преобразование в пиксели и рисование через OpenCV

## ✅ Важно помнить

- **Координаты всегда нормализованные [0, 1]** на этапе сохранения
- **Для визуализации нужно умножить на width/height** чтобы получить пиксели
- **MediaPipe обрабатывает каждый кадр независимо** - нет связи между кадрами
- **33 точки MediaPipe** соответствуют стандартному набору landmarks для человеческой позы

