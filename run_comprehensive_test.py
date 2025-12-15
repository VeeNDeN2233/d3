"""
Комплексное тестирование системы детекции аномалий.

Тестирует систему на различных типах видео:
- Нормальные движения
- Синтетические аномалии (асимметрия, тремор, ригидность)
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_inference(video_path: str, output_dir: str, quiet: bool = True) -> Dict:
    """
    Запустить инференс на видео.
    
    Args:
        video_path: Путь к видео
        output_dir: Директория для результатов
        quiet: Минимальный вывод
    
    Returns:
        Словарь с результатами
    """
    cmd = [
        "python",
        "inference_gpu.py",
        "--video",
        video_path,
        "--checkpoint",
        "checkpoints/anomaly_detector.pt",
        "--config",
        "config.yaml",
        "--output",
        output_dir,
        "--save_report",
    ]
    
    if quiet:
        cmd.append("--quiet")
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )  # 5 минут таймаут
        
        if result.returncode != 0:
            logger.error(f"Ошибка инференса для {video_path}: {result.stderr}")
            return {"error": result.stderr}
        
        # Читаем отчет
        report_path = Path(output_dir) / "medical_report.json"
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            return report
        else:
            logger.warning(f"Отчет не найден: {report_path}")
            return {"error": "Report not found"}
    
    except subprocess.TimeoutExpired:
        logger.error(f"Таймаут для {video_path}")
        return {"error": "Timeout"}
    except Exception as e:
        logger.error(f"Исключение при обработке {video_path}: {e}")
        return {"error": str(e)}


def run_comprehensive_test():
    """Запустить комплексное тестирование."""
    
    logger.info("=" * 60)
    logger.info("КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ СИСТЕМЫ")
    logger.info("=" * 60)
    
    # Проверяем наличие checkpoint
    checkpoint_path = Path("checkpoints/anomaly_detector.pt")
    if not checkpoint_path.exists():
        logger.error("Checkpoint не найден! Сначала обучите модель.")
        return
    
    # Тестовые случаи
    test_cases = [
        {
            "name": "Нормальные движения (MINI-RGBD seq 01)",
            "video": "MINI-RGBD_web/01/rgb",  # Будет обрабатываться как папка с изображениями
            "expected": "normal",
            "type": "normal",
        },
        {
            "name": "Нормальные движения (MINI-RGBD seq 02)",
            "video": "MINI-RGBD_web/02/rgb",
            "expected": "normal",
            "type": "normal",
        },
    ]
    
    # Добавляем синтетические аномалии, если они созданы
    anomaly_dir = Path("test_videos/anomalies")
    if anomaly_dir.exists():
        for anomaly_file in anomaly_dir.glob("*.npy"):
            test_cases.append({
                "name": f"Синтетическая аномалия: {anomaly_file.stem}",
                "video": str(anomaly_file),
                "expected": "anomaly",
                "type": "synthetic",
            })
    
    # Если есть реальные видео
    test_videos_dir = Path("test_videos")
    if test_videos_dir.exists():
        for video_file in test_videos_dir.glob("*.mp4"):
            test_cases.append({
                "name": f"Реальное видео: {video_file.stem}",
                "video": str(video_file),
                "expected": "unknown",  # Неизвестно заранее
                "type": "real",
            })
    
    if not test_cases:
        logger.warning("Нет тестовых случаев! Создайте видео в test_videos/")
        return
    
    logger.info(f"Найдено {len(test_cases)} тестовых случаев")
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Тест {i}/{len(test_cases)}: {test['name']}")
        logger.info(f"{'='*60}")
        
        # Проверяем существование файла/папки
        video_path = Path(test["video"])
        if not video_path.exists():
            logger.warning(f"Пропуск: {video_path} не существует")
            continue
        
        # Создаем директорию для результатов
        output_dir = Path("results") / f"test_{test['name'].replace(' ', '_').replace(':', '_')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Запускаем инференс
        report = run_inference(str(video_path), str(output_dir), quiet=False)
        
        if "error" in report:
            logger.error(f"Ошибка при обработке: {report['error']}")
            result = {
                "test_case": test["name"],
                "expected": test["expected"],
                "detected": "error",
                "anomaly_score": None,
                "threshold": None,
                "correct": False,
                "error": report["error"],
            }
        else:
            # Определяем результат
            anomaly_detection = report.get("anomaly_detection", {})
            anomaly_score = anomaly_detection.get("mean_anomaly_score", 0)
            threshold = anomaly_detection.get("threshold", 2.35)
            
            is_anomaly = anomaly_score > threshold if threshold else False
            detected = "anomaly" if is_anomaly else "normal"
            
            # Проверяем правильность (если expected известен)
            correct = None
            if test["expected"] != "unknown":
                correct = (test["expected"] == "anomaly") == is_anomaly
            
            result = {
                "test_case": test["name"],
                "expected": test["expected"],
                "detected": detected,
                "anomaly_score": round(anomaly_score, 6),
                "threshold": round(threshold, 6) if threshold else None,
                "anomaly_rate": report.get("statistics", {}).get("anomaly_rate", 0),
                "risk_level": anomaly_detection.get("risk_level", "unknown"),
                "correct": correct,
                "report": report,
            }
            
            logger.info(f"  Результат: {detected.upper()}")
            logger.info(f"  Anomaly score: {anomaly_score:.6f}")
            logger.info(f"  Threshold: {threshold:.6f}")
            logger.info(f"  Risk level: {anomaly_detection.get('risk_level', 'unknown')}")
            if correct is not None:
                status = "✅" if correct else "❌"
                logger.info(f"  {status} Правильность: {correct}")
        
        results.append(result)
    
    # Анализ результатов
    logger.info("\n" + "=" * 60)
    logger.info("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    logger.info("=" * 60)
    
    # Фильтруем результаты с известным expected
    known_results = [r for r in results if r.get("correct") is not None]
    
    if known_results:
        correct = sum(1 for r in known_results if r["correct"])
        accuracy = correct / len(known_results) * 100
        
        logger.info(f"\nТочность системы: {accuracy:.1f}% ({correct}/{len(known_results)})")
        logger.info("\nДетальные результаты:")
        
        for r in results:
            if r.get("correct") is not None:
                status = "✅" if r["correct"] else "❌"
            else:
                status = "⚪"
            
            logger.info(
                f"{status} {r['test_case']}: "
                f"Score={r.get('anomaly_score', 'N/A')}, "
                f"Expected={r['expected']}, "
                f"Detected={r['detected']}, "
                f"Risk={r.get('risk_level', 'unknown')}"
            )
    else:
        logger.info("\nНет тестовых случаев с известным expected для вычисления точности")
        logger.info("\nДетальные результаты:")
        
        for r in results:
            logger.info(
                f"⚪ {r['test_case']}: "
                f"Score={r.get('anomaly_score', 'N/A')}, "
                f"Detected={r['detected']}, "
                f"Risk={r.get('risk_level', 'unknown')}"
            )
    
    # Сохраняем сводный отчет
    summary = {
        "total_tests": len(results),
        "known_expected": len(known_results),
        "accuracy": round(accuracy, 2) if known_results else None,
        "correct_predictions": correct if known_results else None,
        "threshold": results[0].get("threshold") if results else None,
        "test_cases": results,
    }
    
    summary_path = Path("results/comprehensive_test_results.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nСводный отчет сохранен: {summary_path}")
    
    return results


if __name__ == "__main__":
    run_comprehensive_test()

