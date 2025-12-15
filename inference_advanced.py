"""
Инференс для продвинутой модели (Bidirectional LSTM + Attention).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

from models.anomaly_detector import AnomalyDetector
from models.autoencoder_advanced import BidirectionalLSTMAutoencoder
from utils.pose_processor import PoseProcessor
from video_processor import VideoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model_and_detector(
    checkpoint_path: Path, config: dict, device: torch.device, model_type: str = "bidir_lstm"
) -> Tuple[nn.Module, AnomalyDetector]:
    """
    Загрузить продвинутую модель и детектор из checkpoint.
    
    Args:
        checkpoint_path: Путь к checkpoint
        config: Конфигурация модели
        device: Устройство (GPU)
        model_type: Тип модели ("bidir_lstm" или "transformer")
    
    Returns:
        Tuple (model, detector)
    """
    # Создаем модель
    if model_type == "bidir_lstm":
        model = BidirectionalLSTMAutoencoder(
            input_size=config["model"]["input_size"],
            sequence_length=config["pose"]["sequence_length"],
            encoder_hidden_sizes=[256, 128, 64],
            decoder_hidden_sizes=[64, 128, 256, 75],
            latent_size=64,
            num_attention_heads=4,
            dropout=0.2,
        ).to(device)
    else:
        raise ValueError(f"Модель {model_type} не поддерживается")
    
    # Загружаем веса
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    logger.info(f"Модель загружена из {checkpoint_path}")
    
    # Загружаем детектор
    detector_path = checkpoint_path.parent / "anomaly_detector_advanced.pt"
    if not detector_path.exists():
        raise FileNotFoundError(f"Детектор не найден: {detector_path}")
    
    detector = AnomalyDetector.load(detector_path, model, device)
    
    logger.info(f"Детектор загружен из {detector_path}")
    logger.info(f"Порог аномалии: {detector.threshold:.6f}")
    
    return model, detector


def process_video(
    video_path: Path,
    video_processor: VideoProcessor,
    pose_processor: PoseProcessor,
    detector: AnomalyDetector,
    config: dict,
) -> Tuple[List[np.ndarray], List[float], List[bool], np.ndarray]:
    """Обработать видео и получить предсказания аномалий."""
    logger.info(f"Обработка видео: {video_path}")
    
    temp_output = video_path.parent / f"temp_{video_path.name}"
    
    result = video_processor._process_video_sync(
        str(video_path), str(temp_output), save_keypoints=True
    )
    
    if not result["success"]:
        raise RuntimeError(f"Ошибка обработки видео: {result.get('error')}")
    
    # Загружаем ключевые точки
    keypoints_path = Path(result["keypoints_path"]) / "keypoints.json"
    with open(keypoints_path, "r", encoding="utf-8") as f:
        keypoints_data = json.load(f)
    
    # Извлекаем ключевые точки
    keypoints_list = []
    for frame_data in keypoints_data["frames"]:
        landmarks = frame_data.get("landmarks")
        if landmarks:
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
    sequences_array = np.array(flattened_sequences)  # (N, 30, 75)
    sequences_tensor = torch.FloatTensor(sequences_array)
    
    # Предсказание аномалий
    is_anomaly, errors = detector.predict(sequences_tensor.to(detector.device))
    
    # Удаляем временный файл
    if temp_output.exists():
        temp_output.unlink()
    
    return keypoints_list, errors.tolist(), is_anomaly.tolist(), sequences_array


def visualize_results(
    errors: List[float],
    is_anomaly: List[bool],
    output_dir: Path,
    video_name: str,
    threshold: Optional[float] = None,
) -> Dict[str, Path]:
    """Визуализация результатов."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # График ошибки реконструкции
    fig, ax = plt.subplots(figsize=(12, 6))
    frames = np.arange(len(errors))
    ax.plot(frames, errors, label="Reconstruction Error", linewidth=2, color="blue")
    
    # Порог аномалии
    if threshold is not None:
        ax.axhline(y=threshold, color="r", linestyle="--", label=f"Anomaly Threshold ({threshold:.4f})", linewidth=2)
    
    # Помечаем аномальные последовательности
    anomaly_frames = [i for i, is_anom in enumerate(is_anomaly) if is_anom]
    if anomaly_frames:
        ax.scatter(anomaly_frames, [errors[i] for i in anomaly_frames], 
                  color="red", s=50, label="Anomalous", zorder=5)
    
    ax.set_xlabel("Sequence Index", fontsize=12)
    ax.set_ylabel("Reconstruction Error (MSE)", fontsize=12)
    ax.set_title(f"Reconstruction Error Over Time - {video_name}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    error_plot_path = output_dir / "reconstruction_error.png"
    plt.savefig(error_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"График сохранен: {error_plot_path}")
    
    return {"error_plot": error_plot_path}


def generate_report(
    video_path: Path,
    errors: List[float],
    is_anomaly: List[bool],
    detector: AnomalyDetector,
    output_dir: Path,
    age_weeks: Optional[float] = None,
    gestational_age_weeks: Optional[float] = None,
    sequences_array: Optional[np.ndarray] = None,
) -> Dict:
    """Генерация медицинского отчета в формате GMA."""
    if len(errors) == 0:
        return {}
    
    errors_array = np.array(errors)
    anomaly_rate = np.mean(is_anomaly) * 100
    
    # Определяем уровень риска на основе GMA критериев
    mean_error = float(errors_array.mean())
    
    # GMA критерии риска
    if mean_error > detector.threshold * 1.5:
        risk_level = "high"
        gma_assessment = "АНОМАЛЬНЫЕ общие движения"
        cp_risk = "ВЫСОКИЙ риск церебрального паралича"
    elif mean_error > detector.threshold:
        risk_level = "medium"
        gma_assessment = "ПОДОЗРИТЕЛЬНЫЕ общие движения"
        cp_risk = "УМЕРЕННЫЙ риск неврологических нарушений"
    else:
        risk_level = "low"
        gma_assessment = "НОРМАЛЬНЫЕ общие движения"
        cp_risk = "НИЗКИЙ риск"
    
    # Детальный анализ аномалий по суставам
    detailed_analysis = {}
    if sequences_array is not None and len(sequences_array) > 0 and risk_level != "low":
        try:
            from utils.anomaly_analyzer import analyze_joint_errors
            
            sequences_np = np.array(sequences_array)
            errors_np = np.array(errors)
            
            detailed_analysis = analyze_joint_errors(
                sequences_np,
                errors_np,
                detector.threshold
            )
        except Exception as e:
            logger.warning(f"Ошибка детального анализа: {e}")
            detailed_analysis = {}
    
    # Определяем признаки аномалий на основе детального анализа
    detected_signs = []
    if detailed_analysis.get("has_anomalies", False):
        # Асимметрия
        asymmetry = detailed_analysis.get("asymmetry", {})
        if asymmetry.get("has_asymmetry", False):
            for finding in asymmetry.get("findings", []):
                detected_signs.append(finding["description"])
        
        # Анализ суставов
        joint_analysis = detailed_analysis.get("joint_analysis", {})
        for finding in joint_analysis.get("findings", []):
            if finding["type"] == "reduced_movement":
                detected_signs.append(finding["description"])
            elif finding["type"] == "high_speed":
                detected_signs.append(finding["description"])
        
        # Скорость движений
        speed_analysis = detailed_analysis.get("speed_analysis", {})
        for finding in speed_analysis.get("findings", []):
            detected_signs.append(finding["description"])
        
        # Амплитуда движений
        amplitude_analysis = detailed_analysis.get("amplitude_analysis", {})
        for finding in amplitude_analysis.get("findings", []):
            detected_signs.append(finding["description"])
    
    # Fallback если детальный анализ недоступен
    if len(detected_signs) == 0:
        if anomaly_rate > 30:
            detected_signs.append("высокая частота аномальных паттернов")
        if mean_error > detector.threshold * 1.2:
            detected_signs.append("сниженная вариабельность движений")
        if len(detected_signs) == 0 and risk_level != "low":
            detected_signs.append("отклонения от нормальных паттернов")
    
    # Рекомендации на основе GMA
    recommendations = []
    if risk_level == "low":
        recommendations.append("✅ Рекомендация: Плановая оценка в 4 месяца")
        recommendations.append("Продолжить стандартное наблюдение")
    elif risk_level == "medium":
        recommendations.append("⚠️ Рекомендация: Повторная оценка через 2-4 недели")
        recommendations.append("Наблюдение у педиатра")
    else:  # high
        recommendations.append("🔴 Рекомендация: СРОЧНАЯ консультация детского невролога")
        recommendations.append("Начать раннее вмешательство")
        if detected_signs:
            recommendations.append(f"Выявлены признаки: {', '.join(detected_signs)}")
    
    # Информация о возрасте
    age_info = {}
    if age_weeks is not None:
        age_info["age_weeks"] = float(age_weeks)
        if age_weeks >= 9 and age_weeks <= 20:
            age_info["period"] = "Период суетливых движений (fidgety movements)"
        elif age_weeks < 9:
            age_info["period"] = "Ранний период (writhing movements)"
        else:
            age_info["period"] = "Поздний период"
    
    if gestational_age_weeks is not None:
        age_info["gestational_age_weeks"] = float(gestational_age_weeks)
        if gestational_age_weeks < 37:
            age_info["premature"] = True
            age_info["corrected_age"] = age_weeks - (40 - gestational_age_weeks) if age_weeks else None
    
    report = {
        "video_path": str(video_path),
        "analysis_date": str(Path.cwd()),
        "gma_assessment": {
            "assessment_result": gma_assessment,
            "risk_level": risk_level.upper(),
            "cp_risk": cp_risk,
            "detected_signs": detected_signs,
        },
        "patient_info": age_info,
        "statistics": {
            "total_sequences": len(errors),
            "anomalous_sequences": sum(is_anomaly),
            "anomaly_rate": anomaly_rate,
        },
        "reconstruction_errors": {
            "mean": float(errors_array.mean()),
            "max": float(errors_array.max()),
            "min": float(errors_array.min()),
            "std": float(errors_array.std()),
        },
        "anomaly_detection": {
            "threshold": float(detector.threshold),
            "mean_anomaly_score": mean_error,
            "risk_level": risk_level,
            "anomaly_rate_percent": anomaly_rate,
        },
        "recommendations": recommendations,
        "detailed_analysis": detailed_analysis,
    }
    
    # Сохраняем отчет
    report_path = output_dir / "medical_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Отчет сохранен: {report_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Инференс для продвинутой модели")
    parser.add_argument("--video", type=str, required=True, help="Путь к видео")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model_advanced.pt",
                       help="Путь к checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml", help="Путь к конфигурации")
    parser.add_argument("--output", type=str, help="Путь для сохранения результатов")
    parser.add_argument("--save_report", action="store_true", help="Сохранить отчет")
    parser.add_argument("--model_type", type=str, default="bidir_lstm", 
                       choices=["bidir_lstm", "transformer"], help="Тип модели")
    
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Проверяем GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Требуется GPU для инференса!")
    
    logger.info(f"Используется устройство: {device}")
    
    # Загружаем модель и детектор
    checkpoint_path = Path(args.checkpoint)
    model, detector = load_model_and_detector(checkpoint_path, config, device, args.model_type)
    
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
    
    # Определяем директорию для результатов
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("results") / video_path.stem
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Визуализация
    visualize_results(errors, is_anomaly, output_dir, video_path.stem, detector.threshold)
    
    # Генерация отчета
    if args.save_report:
        report = generate_report(video_path, errors, is_anomaly, detector, output_dir)
        
        logger.info("=" * 60)
        logger.info("РЕЗУЛЬТАТЫ АНАЛИЗА")
        logger.info("=" * 60)
        logger.info(f"Уровень риска: {report['anomaly_detection']['risk_level'].upper()}")
        logger.info(f"Аномалий: {report['anomaly_detection']['anomaly_rate_percent']:.2f}%")
        logger.info(f"Средняя ошибка: {report['reconstruction_errors']['mean']:.6f}")
        logger.info(f"Порог: {report['anomaly_detection']['threshold']:.6f}")
    
    logger.info(f"Результаты сохранены в: {output_dir}")


if __name__ == "__main__":
    main()

