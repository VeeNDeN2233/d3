"""
Скрипт инференса для новых видео.

Инференс для новых видео:
- MediaPipe на CPU (если нужно)
- Модель на GPU
- Визуализация:
    * Скелет с цветовой кодировкой уверенности
    * Heatmap аномальных суставов
    * График ошибки реконструкции по времени
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Устанавливаем backend для matplotlib перед импортом pyplot
import matplotlib
matplotlib.use('Agg')  # Используем non-interactive backend

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from models.anomaly_detector import AnomalyDetector
from models.autoencoder_gpu import LSTMAutoencoderGPU
from utils.pose_processor import PoseProcessor
from video_processor import VideoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model_and_detector(
    checkpoint_path: Path, config: dict, device: torch.device
) -> Tuple[LSTMAutoencoderGPU, AnomalyDetector]:
    """
    Загрузить модель и детектор из checkpoint.
    
    Args:
        checkpoint_path: Путь к checkpoint
        config: Конфигурация модели
        device: Устройство (GPU)
    
    Returns:
        Tuple (model, detector)
    """
    # Создаем модель
    model = LSTMAutoencoderGPU(
        input_size=config["model"]["input_size"],
        sequence_length=config["pose"]["sequence_length"],
        encoder_hidden_sizes=config["model"]["encoder_hidden_sizes"],
        decoder_hidden_sizes=config["model"]["decoder_hidden_sizes"],
        encoder_num_layers=config["model"]["encoder_num_layers"],
        decoder_num_layers=config["model"]["decoder_num_layers"],
        encoder_dropout=config["model"]["encoder_dropout"],
        decoder_dropout=config["model"]["decoder_dropout"],
        latent_size=config["model"]["latent_size"],
    )
    
    model = model.to_gpu()
    
    # Загружаем детектор (включает модель и порог)
    detector_path = checkpoint_path.parent / "anomaly_detector.pt"
    if not detector_path.exists():
        raise FileNotFoundError(f"Детектор не найден: {detector_path}")
    
    detector = AnomalyDetector.load(detector_path, model, device)
    
    logger.info(f"Модель и детектор загружены из {checkpoint_path}")
    return model, detector


def process_video(
    video_path: Path,
    video_processor: VideoProcessor,
    pose_processor: PoseProcessor,
    detector: AnomalyDetector,
    config: dict,
) -> Tuple[List[np.ndarray], List[float], List[bool]]:
    """
    Обработать видео и получить предсказания аномалий.
    
    Args:
        video_path: Путь к видео
        video_processor: Процессор видео (MediaPipe)
        pose_processor: Процессор позы
        detector: Детектор аномалий
        config: Конфигурация
    
    Returns:
        Tuple (keypoints_list, errors, is_anomaly):
        - keypoints_list: Список ключевых точек для каждого кадра
        - errors: Ошибки реконструкции для каждого кадра
        - is_anomaly: Булев массив аномалий
    """
    logger.info(f"Обработка видео: {video_path}")
    
    # Временный путь для обработанного видео
    temp_output = video_path.parent / f"temp_{video_path.name}"
    
    # Обрабатываем видео через MediaPipe
    result = video_processor._process_video_sync(
        str(video_path), str(temp_output), save_keypoints=True
    )
    
    if not result["success"]:
        raise RuntimeError(f"Ошибка обработки видео: {result.get('error')}")
    
    # Загружаем сохраненные ключевые точки
    keypoints_path = Path(result["keypoints_path"]) / "keypoints.json"
    import json
    
    with open(keypoints_path, "r", encoding="utf-8") as f:
        keypoints_data = json.load(f)
    
    # Извлекаем ключевые точки
    keypoints_list = []
    for frame_data in keypoints_data["frames"]:
        landmarks = frame_data.get("landmarks")
        if landmarks:
            # Конвертируем в numpy
            kp = np.array(
                [[lm["x"], lm["y"], lm["z"], lm.get("visibility", 0.0)] for lm in landmarks],
                dtype=np.float32,
            )
            keypoints_list.append(kp)
        else:
            keypoints_list.append(None)
    
    # Обрабатываем ключевые точки
    sequences = pose_processor.process_keypoints(keypoints_list)
    
    if len(sequences) == 0:
        logger.warning("Нет валидных последовательностей")
        return keypoints_list, [], []
    
    # Преобразуем в плоские векторы
    flattened_sequences = [pose_processor.flatten_sequence(seq) for seq in sequences]
    sequences_tensor = torch.FloatTensor(np.array(flattened_sequences))
    
    # Предсказание аномалий
    is_anomaly, errors = detector.predict(sequences_tensor.to(detector.device))
    
    # Удаляем временный файл
    if temp_output.exists():
        temp_output.unlink()
    
    return keypoints_list, errors.tolist(), is_anomaly.tolist()


def visualize_results(
    video_path: Path,
    keypoints_list: List[Optional[np.ndarray]],
    errors: List[float],
    is_anomaly: List[bool],
    output_dir: Path,
    config: dict,
    detector: AnomalyDetector,
):
    """
    Визуализировать результаты детекции аномалий.
    
    Args:
        video_path: Путь к исходному видео
        keypoints_list: Список ключевых точек
        errors: Ошибки реконструкции
        is_anomaly: Булев массив аномалий
        output_dir: Директория для сохранения результатов
        config: Конфигурация
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. График ошибки реконструкции по времени
    if errors:
        plt.figure(figsize=(12, 6))
        frames = range(len(errors))
        plt.plot(frames, errors, "b-", label="Reconstruction Error", linewidth=2)
        
        # Порог аномалии
        if hasattr(detector, "threshold") and detector.threshold:
            plt.axhline(
                y=detector.threshold,
                color="r",
                linestyle="--",
                label=f"Anomaly Threshold ({detector.threshold:.6f})",
            )
        
        # Закрашиваем аномальные области
        for i, anomaly in enumerate(is_anomaly):
            if anomaly:
                plt.axvspan(i - 0.5, i + 0.5, alpha=0.2, color="red")
        
        plt.xlabel("Frame", fontsize=12)
        plt.ylabel("Reconstruction Error (MSE)", fontsize=12)
        plt.title("Anomaly Detection: Reconstruction Error Over Time", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        error_plot_path = output_dir / "reconstruction_error.png"
        plt.savefig(error_plot_path, dpi=config.get("visualization", {}).get("dpi", 300))
        plt.close()
        logger.info(f"График ошибки сохранен: {error_plot_path}")
    
    # 2. Heatmap аномальных суставов (если есть последовательности)
    # TODO: Реализовать heatmap для отдельных суставов
    
    # 3. Статистика
    stats_path = output_dir / "anomaly_stats.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("ANOMALY DETECTION STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Total frames: {len(keypoints_list)}\n")
        f.write(f"Valid sequences: {len(errors)}\n")
        f.write(f"Anomaly threshold: {detector.threshold if hasattr(detector, 'threshold') else 'N/A'}\n\n")
        
        if errors:
            f.write("Reconstruction Errors:\n")
            f.write(f"  Min: {min(errors):.6f}\n")
            f.write(f"  Max: {max(errors):.6f}\n")
            f.write(f"  Mean: {np.mean(errors):.6f}\n")
            f.write(f"  Std: {np.std(errors):.6f}\n\n")
            
            f.write("Anomaly Detection:\n")
            num_anomalies = sum(is_anomaly)
            f.write(f"  Anomalous sequences: {num_anomalies} / {len(is_anomaly)}\n")
            f.write(f"  Anomaly rate: {num_anomalies / len(is_anomaly) * 100:.2f}%\n")
    
    logger.info(f"Статистика сохранена: {stats_path}")


def generate_medical_report(
    video_path: Path,
    errors: List[float],
    is_anomaly: List[bool],
    detector: AnomalyDetector,
    keypoints_list: List[Optional[np.ndarray]],
) -> Dict:
    """
    Генерировать медицинский отчет.
    
    Args:
        video_path: Путь к видео
        errors: Ошибки реконструкции
        is_anomaly: Булев массив аномалий
        detector: Детектор аномалий
        keypoints_list: Список ключевых точек
    
    Returns:
        Словарь с медицинским отчетом
    """
    if not errors:
        return {
            "status": "error",
            "message": "Недостаточно данных для анализа",
        }
    
    # Вычисляем статистику
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    std_error = np.std(errors)
    
    num_anomalies = sum(is_anomaly)
    total_sequences = len(is_anomaly)
    anomaly_rate = num_anomalies / total_sequences * 100 if total_sequences > 0 else 0
    
    # Оценка риска
    threshold = detector.threshold if hasattr(detector, "threshold") else None
    
    if threshold is None:
        risk_level = "unknown"
    elif mean_error > threshold * 1.5:
        risk_level = "high"
    elif mean_error > threshold:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    # Процент кадров с валидными позами
    valid_poses = sum(1 for kp in keypoints_list if kp is not None)
    pose_detection_rate = valid_poses / len(keypoints_list) * 100 if keypoints_list else 0
    
    report = {
        "video_path": str(video_path),
        "analysis_date": str(Path(video_path).stat().st_mtime),
        "statistics": {
            "total_frames": len(keypoints_list),
            "valid_poses": valid_poses,
            "pose_detection_rate": round(pose_detection_rate, 2),
            "total_sequences": total_sequences,
            "anomalous_sequences": num_anomalies,
            "anomaly_rate": round(anomaly_rate, 2),
        },
        "reconstruction_errors": {
            "mean": round(float(mean_error), 6),
            "max": round(float(max_error), 6),
            "min": round(float(min_error), 6),
            "std": round(float(std_error), 6),
        },
        "anomaly_detection": {
            "threshold": round(float(threshold), 6) if threshold else None,
            "mean_anomaly_score": round(float(mean_error), 6),
            "risk_level": risk_level,
            "anomaly_rate_percent": round(anomaly_rate, 2),
        },
        "recommendations": _generate_recommendations(risk_level, anomaly_rate, mean_error, threshold),
    }
    
    return report


def _generate_recommendations(
    risk_level: str, anomaly_rate: float, mean_error: float, threshold: Optional[float]
) -> List[str]:
    """Генерировать рекомендации на основе анализа."""
    recommendations = []
    
    if risk_level == "high":
        recommendations.append(
            "Высокий риск нарушений моторики. Рекомендуется консультация специалиста."
        )
        recommendations.append(
            "Повторное обследование через 2-4 недели для отслеживания динамики."
        )
    elif risk_level == "medium":
        recommendations.append(
            "Умеренный риск. Рекомендуется наблюдение и повторный анализ."
        )
        recommendations.append(
            "Обратите внимание на симметрию движений и диапазон активности."
        )
    else:
        recommendations.append(
            "Низкий риск нарушений моторики. Движения в пределах нормы."
        )
    
    if anomaly_rate > 30:
        recommendations.append(
            f"Обнаружено {anomaly_rate:.1f}% аномальных последовательностей. "
            "Требуется дополнительное обследование."
        )
    
    if mean_error and threshold and mean_error > threshold * 1.2:
        recommendations.append(
            "Средняя ошибка реконструкции значительно превышает порог. "
            "Возможны паттерны движений, отличающиеся от нормы."
        )
    
    return recommendations


def save_report(report: Dict, output_dir: Path) -> Path:
    """Сохранить отчет в JSON файл."""
    report_path = output_dir / "medical_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Медицинский отчет сохранен: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Инференс для детекции аномалий")
    parser.add_argument("--video", type=str, required=True, help="Путь к видео")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Путь к checkpoint модели"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Путь к config.yaml"
    )
    parser.add_argument(
        "--output", type=str, default="inference_results", help="Директория для результатов"
    )
    parser.add_argument(
        "--save_report", action="store_true", help="Сохранить медицинский отчет в JSON"
    )
    parser.add_argument(
        "--show_heatmap", action="store_true", help="Показать heatmap аномальных суставов"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Минимальный вывод в консоль"
    )
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    import yaml
    
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Проверяем GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU недоступен! Инференс требует GPU.")
    
    logger.info(f"Используется устройство: {device}")
    
    # Загружаем модель и детектор
    checkpoint_path = Path(args.checkpoint)
    model, detector = load_model_and_detector(checkpoint_path, config, device)
    
    # Инициализация процессоров
    video_processor = VideoProcessor(
        model_complexity=config["pose"]["model_complexity"],
        min_detection_confidence=config["pose"]["min_detection_confidence"],
        min_tracking_confidence=config["pose"]["min_tracking_confidence"],
    )
    
    pose_processor = PoseProcessor(
        sequence_length=config["pose"]["sequence_length"],
        sequence_stride=config["pose"]["sequence_stride"],
        normalize=config["pose"]["normalize"],
        normalize_relative_to=config["pose"]["normalize_relative_to"],
        target_hip_distance=config["pose"].get("target_hip_distance"),
        normalize_by_body=config["pose"].get("normalize_by_body", False),
        rotate_to_canonical=config["pose"].get("rotate_to_canonical", False),
    )
    
    # Обработка видео
    video_path = Path(args.video)
    keypoints_list, errors, is_anomaly = process_video(
        video_path, video_processor, pose_processor, detector, config
    )
    
    # Визуализация
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_results(video_path, keypoints_list, errors, is_anomaly, output_dir, config, detector)
    
    # Генерация и сохранение медицинского отчета
    if args.save_report:
        report = generate_medical_report(
            video_path, errors, is_anomaly, detector, keypoints_list
        )
        save_report(report, output_dir)
        
        if not args.quiet:
            logger.info("\n" + "=" * 60)
            logger.info("МЕДИЦИНСКИЙ ОТЧЕТ")
            logger.info("=" * 60)
            logger.info(f"Уровень риска: {report['anomaly_detection']['risk_level'].upper()}")
            logger.info(f"Процент аномалий: {report['statistics']['anomaly_rate']:.2f}%")
            logger.info(f"Средняя ошибка: {report['reconstruction_errors']['mean']:.6f}")
            logger.info("\nРекомендации:")
            for rec in report['recommendations']:
                logger.info(f"  - {rec}")
    
    if not args.quiet:
        logger.info("\nИнференс завершен!")
        logger.info(f"Результаты сохранены в: {output_dir}")


if __name__ == "__main__":
    main()

