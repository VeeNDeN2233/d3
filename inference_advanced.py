
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any



import matplotlib
matplotlib.use('Agg')

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


def convert_numpy_types(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:

        try:
            return str(obj)
        except:
            return obj


def load_model_and_detector(
    checkpoint_path: Path, config: dict, device: torch.device, model_type: str = "bidir_lstm"
) -> Tuple[nn.Module, AnomalyDetector]:

    # Загружаем checkpoint для получения сохраненной конфигурации
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_config = checkpoint.get("config", {})
    
    # Используем параметры из сохраненного config, если он есть, иначе из текущего config
    if saved_config and "model" in saved_config:
        model_config = saved_config["model"]
        logger.info(f"Используется конфигурация из checkpoint")
    else:
        model_config = config.get("model", {})
        logger.info(f"Используется текущая конфигурация")
    
    if model_type == "bidir_lstm":
        # Берем значения напрямую из model_config без fallback на config
        encoder_sizes = model_config["encoder_hidden_sizes"] if "encoder_hidden_sizes" in model_config else [128, 64, 32]
        decoder_sizes = model_config["decoder_hidden_sizes"] if "decoder_hidden_sizes" in model_config else [64, 128, 75]
        latent_size_val = model_config["latent_size"] if "latent_size" in model_config else 32
        input_size_val = model_config["input_size"] if "input_size" in model_config else 75
        
        logger.info(f"Создание модели: encoder={encoder_sizes}, decoder={decoder_sizes}, latent={latent_size_val}")
        
        model = BidirectionalLSTMAutoencoder(
            input_size=input_size_val,
            sequence_length=config["pose"]["sequence_length"],
            encoder_hidden_sizes=encoder_sizes,
            decoder_hidden_sizes=decoder_sizes,
            latent_size=latent_size_val,
            num_attention_heads=4,
            dropout=model_config.get("encoder_dropout", 0.2),
        ).to(device)
    else:
        raise ValueError(f"Модель {model_type} не поддерживается")
    
    # Загружаем веса модели
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    logger.info(f"Модель загружена из {checkpoint_path}")
    

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
    logger.info(f"Обработка видео: {video_path}")
    
    temp_output = video_path.parent / f"temp_{video_path.name}"
    
    result = video_processor._process_video_sync(
        str(video_path), str(temp_output), save_keypoints=True
    )
    
    if not result["success"]:
        raise RuntimeError(f"Ошибка обработки видео: {result.get('error')}")
    

    keypoints_path = Path(result["keypoints_path"]) / "keypoints.json"
    with open(keypoints_path, "r", encoding="utf-8") as f:
        keypoints_data = json.load(f)
    

    keypoints_list = []
    for frame_data in keypoints_data["frames"]:
        landmarks = frame_data.get("landmarks")

        if landmarks and len(landmarks) == 33:
            kp = np.array(
                [[lm["x"], lm["y"], lm["z"], lm.get("visibility", 0.0)] for lm in landmarks],
                dtype=np.float32,
            )
        else:

            kp = np.zeros((33, 4), dtype=np.float32)
        
        keypoints_list.append(kp)
    

    sequences = pose_processor.process_keypoints(keypoints_list)
    
    if len(sequences) == 0:
        logger.warning("Нет валидных последовательностей")
        return keypoints_list, [], [], np.array([])
    

    flattened_sequences = []
    for seq in sequences:
        flattened_sequences.append(pose_processor.flatten_sequence(seq))
    sequences_array = np.array(flattened_sequences, dtype=np.float32)
    sequences_tensor = torch.FloatTensor(sequences_array)
    

    is_anomaly, errors = detector.predict(sequences_tensor.to(detector.device))
    

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
    output_dir.mkdir(parents=True, exist_ok=True)
    

    fig, ax = plt.subplots(figsize=(12, 6))
    frames = np.arange(len(errors))
    ax.plot(frames, errors, label="Reconstruction Error", linewidth=2, color="blue")
    

    if threshold is not None:
        ax.axhline(y=threshold, color="r", linestyle="--", label=f"Anomaly Threshold ({threshold:.4f})", linewidth=2)
    

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
    if len(errors) == 0:
        return {}
    
    errors_array = np.array(errors)
    anomaly_rate = np.mean(is_anomaly) * 100
    anomalous_count = sum(is_anomaly)
    total_count = len(is_anomaly)
    

    mean_error = float(errors_array.mean())
    max_error = float(errors_array.max())
    

    # Определяем риск на основе процента аномалий И максимальной ошибки
    # Логика: проверяем сначала максимальную ошибку (критические пики), затем процент аномалий
    logger.info(f"Определение риска: anomaly_rate={anomaly_rate:.2f}%, max_error={max_error:.6f}, mean_error={mean_error:.6f}, threshold={detector.threshold:.6f}")
    
    if max_error > detector.threshold * 2.0 or anomaly_rate > 30.0:
        risk_level = "high"
        gma_assessment = "АНОМАЛЬНЫЕ общие движения"
        cp_risk = "ВЫСОКИЙ риск церебрального паралича"
        logger.info(f"Высокий риск: max_error={max_error:.6f} > threshold*2={detector.threshold*2.0:.6f} или anomaly_rate={anomaly_rate:.2f}% > 30%")
    elif max_error > detector.threshold * 1.5 or anomaly_rate > 15.0 or mean_error > detector.threshold:
        risk_level = "medium"
        gma_assessment = "ПОДОЗРИТЕЛЬНЫЕ общие движения"
        cp_risk = "УМЕРЕННЫЙ риск неврологических нарушений"
        logger.info(f"Средний риск: max_error={max_error:.6f} > threshold*1.5={detector.threshold*1.5:.6f} или anomaly_rate={anomaly_rate:.2f}% > 15% или mean_error={mean_error:.6f} > threshold={detector.threshold:.6f}")
    elif anomaly_rate > 5.0 or mean_error > detector.threshold * 0.8 or anomalous_count > 10:
        risk_level = "medium"
        gma_assessment = "ПОДОЗРИТЕЛЬНЫЕ общие движения (легкие отклонения)"
        cp_risk = "НИЗКИЙ-УМЕРЕННЫЙ риск"
        logger.info(f"Средний риск (легкие отклонения): anomaly_rate={anomaly_rate:.2f}% > 5% или mean_error={mean_error:.6f} > threshold*0.8={detector.threshold*0.8:.6f} или anomalous_count={anomalous_count} > 10")
    elif anomalous_count > 0:
        risk_level = "medium"
        gma_assessment = "ПОДОЗРИТЕЛЬНЫЕ общие движения (обнаружены отдельные аномалии)"
        cp_risk = "НИЗКИЙ-УМЕРЕННЫЙ риск (требуется наблюдение)"
        logger.info(f"Средний риск (отдельные аномалии): anomalous_count={anomalous_count} > 0")
    else:
        risk_level = "low"
        gma_assessment = "НОРМАЛЬНЫЕ общие движения"
        cp_risk = "НИЗКИЙ риск"
        logger.info(f"Низкий риск: все метрики в норме")
    



    detailed_analysis = {}
    logger.info(f"Проверка sequences_array для детального анализа: sequences_array is None={sequences_array is None}, "
               f"len={len(sequences_array) if sequences_array is not None else 0}, shape={sequences_array.shape if sequences_array is not None and hasattr(sequences_array, 'shape') else 'N/A'}")
    
    if sequences_array is not None and len(sequences_array) > 0:
        try:
            from utils.anomaly_analyzer import analyze_joint_errors
            from utils.normal_statistics import get_normal_statistics
            
            sequences_np = np.array(sequences_array)
            errors_np = np.array(errors)
            
            logger.info(f"Запуск детального анализа: {len(sequences_np)} последовательностей, {sum(is_anomaly)} аномальных")

            normal_statistics = get_normal_statistics()
            if normal_statistics:
                logger.info("Используются нормальные статистики из тренировочных данных")
            else:
                logger.warning("Нормальные статистики не найдены, будут использоваться статистики из текущего видео")


            detailed_analysis = analyze_joint_errors(
                sequences_np,
                errors_np,
                detector.threshold,
                normal_statistics=normal_statistics,
                age_weeks=age_weeks,
                analyze_all_sequences=True
            )
            
            logger.info(f"Детальный анализ завершен: has_anomalies={detailed_analysis.get('has_anomalies', False)}, "
                       f"joint_findings={len(detailed_analysis.get('joint_analysis', {}).get('findings', []))}, "
                       f"asymmetry={detailed_analysis.get('asymmetry', {}).get('has_asymmetry', False)}")
            

            amplitude_analysis = detailed_analysis.get("amplitude_analysis", {})
            if amplitude_analysis.get("has_amplitude_anomalies", False):

                if amplitude_analysis.get("critical_amplitude_drop", False):
                    if risk_level == "low":
                        risk_level = "high"
                        anomaly_rate = 100.0
                        gma_assessment = "АНОМАЛЬНЫЕ общие движения (критическое снижение активности)"
                        cp_risk = "ВЫСОКИЙ риск (отсутствие/критическое снижение движений)"
                    elif risk_level == "medium":
                        risk_level = "high"
                elif amplitude_analysis.get("moderate_amplitude_drop", False):
                    if risk_level == "low":
                        risk_level = "medium"
                        anomaly_rate = max(anomaly_rate, 50.0)
                        if gma_assessment == "НОРМАЛЬНЫЕ общие движения":
                            gma_assessment = "ПОДОЗРИТЕЛЬНЫЕ общие движения (сниженная активность)"
        except Exception as e:
            logger.error(f"Ошибка детального анализа: {e}", exc_info=True)
            detailed_analysis = {}
    else:
        logger.warning(f"Детальный анализ не выполнен: sequences_array is None или пустой (None={sequences_array is None}, len={len(sequences_array) if sequences_array is not None else 0})")
    

    detected_signs = []
    if detailed_analysis.get("has_anomalies", False):

        asymmetry = detailed_analysis.get("asymmetry", {})
        if asymmetry.get("has_asymmetry", False):
            for finding in asymmetry.get("findings", []):
                detected_signs.append(finding["description"])
        

        joint_analysis = detailed_analysis.get("joint_analysis", {})
        for finding in joint_analysis.get("findings", []):
            if finding["type"] == "reduced_movement":
                detected_signs.append(finding["description"])
            elif finding["type"] == "high_speed":
                detected_signs.append(finding["description"])
        

        speed_analysis = detailed_analysis.get("speed_analysis", {})
        for finding in speed_analysis.get("findings", []):
            detected_signs.append(finding["description"])
        

        amplitude_analysis = detailed_analysis.get("amplitude_analysis", {})
        for finding in amplitude_analysis.get("findings", []):
            detected_signs.append(finding["description"])
    

    # Даже если детальный анализ не выполнился, добавляем информацию об аномалиях
    if len(detected_signs) == 0 and anomalous_count > 0:
        if anomaly_rate > 30:
            detected_signs.append(f"Высокая частота аномальных паттернов движений ({anomaly_rate:.1f}% последовательностей)")
        elif anomaly_rate > 15:
            detected_signs.append(f"Повышенная частота аномальных паттернов движений ({anomaly_rate:.1f}% последовательностей)")
        elif anomaly_rate > 5:
            detected_signs.append(f"Умеренная частота аномальных паттернов движений ({anomaly_rate:.1f}% последовательностей)")
        
        if max_error > detector.threshold * 2.0:
            detected_signs.append(f"Критические отклонения в отдельных последовательностях (макс. ошибка: {max_error:.4f})")
        elif max_error > detector.threshold * 1.5:
            detected_signs.append(f"Значительные отклонения в отдельных последовательностях (макс. ошибка: {max_error:.4f})")
        
        if mean_error > detector.threshold * 1.2:
            detected_signs.append("Сниженная вариабельность движений")
        
        if len(detected_signs) == 0 and risk_level != "low":
            detected_signs.append(f"Обнаружены отклонения от нормальных паттернов движений ({anomalous_count} из {total_count} последовательностей)")
    

    recommendations = []
    if risk_level == "low":
        recommendations.append("✅ Рекомендация: Плановая оценка в 4 месяца")
        recommendations.append("Продолжить стандартное наблюдение")
    elif risk_level == "medium":
        recommendations.append("⚠️ Рекомендация: Повторная оценка через 2-4 недели")
        recommendations.append("Наблюдение у педиатра")
    else:
        recommendations.append("🔴 Рекомендация: СРОЧНАЯ консультация детского невролога")
        recommendations.append("Начать раннее вмешательство")
        if detected_signs:
            recommendations.append(f"Выявлены признаки: {', '.join(detected_signs)}")
    

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
    

    report_serializable = convert_numpy_types(report)
    

    report_path = output_dir / "medical_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_serializable, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Отчет сохранен: {report_path}")
    
    return report_serializable


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
    

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Требуется GPU для инференса!")
    
    logger.info(f"Используется устройство: {device}")
    

    checkpoint_path = Path(args.checkpoint)
    model, detector = load_model_and_detector(checkpoint_path, config, device, args.model_type)
    

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
    

    video_path = Path(args.video)
    keypoints_list, errors, is_anomaly = process_video(
        video_path, video_processor, pose_processor, detector, config
    )
    

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("results") / video_path.stem
    
    output_dir.mkdir(parents=True, exist_ok=True)
    

    visualize_results(errors, is_anomaly, output_dir, video_path.stem, detector.threshold)
    

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

