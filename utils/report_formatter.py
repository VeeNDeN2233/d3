"""
Утилита для форматирования медицинских отчетов.
"""

from typing import Dict


def format_medical_report(report: Dict) -> str:
    """Форматировать медицинский отчет в формате GMA для отображения."""
    if not report:
        return "Ошибка: Отчет пуст"
    
    lines = []
    lines.append("=" * 70)
    lines.append("ОТЧЕТ ПО ОЦЕНКЕ ОБЩИХ ДВИЖЕНИЙ (GMA)")
    lines.append("=" * 70)
    lines.append("")
    
    # GMA оценка
    gma = report.get("gma_assessment", {})
    if gma:
        risk_level = gma.get("risk_level", "unknown").upper()
        risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "UNKNOWN": "⚪"}
        
        lines.append("РЕЗУЛЬТАТ GMA ОЦЕНКИ:")
        lines.append(f"  {risk_emoji.get(risk_level, '⚪')} Риск двигательных нарушений: {risk_level}")
        lines.append(f"  Оценка общих движений: {gma.get('assessment_result', 'N/A')}")
        lines.append(f"  Риск ДЦП: {gma.get('cp_risk', 'N/A')}")
        
        lines.append("")
    else:
        # Fallback для старых отчетов
        anomaly = report.get("anomaly_detection", {})
        risk_level = anomaly.get("risk_level", "unknown").upper()
        risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "UNKNOWN": "⚪"}
        lines.append("РЕЗУЛЬТАТ ОЦЕНКИ:")
        lines.append(f"  {risk_emoji.get(risk_level, '⚪')} Риск двигательных нарушений: {risk_level}")
        lines.append("")
    
    # Информация о пациенте
    patient_info = report.get("patient_info", {})
    if patient_info:
        lines.append("ДАННЫЕ ПАЦИЕНТА:")
        if "age_weeks" in patient_info:
            lines.append(f"  Возраст: {patient_info['age_weeks']:.0f} недель после родов")
        if "period" in patient_info:
            lines.append(f"  Период: {patient_info['period']}")
        if patient_info.get("premature"):
            lines.append(f"  Недоношенность: {patient_info.get('gestational_age_weeks', 'N/A')} недель")
            if "corrected_age" in patient_info and patient_info["corrected_age"]:
                lines.append(f"  Скорректированный возраст: {patient_info['corrected_age']:.0f} недель")
        lines.append("")
    
    # Статистика анализа
    stats = report.get("statistics", {})
    lines.append("СТАТИСТИКА АНАЛИЗА:")
    lines.append(f"  Проанализировано последовательностей: {stats.get('total_sequences', 'N/A')}")
    lines.append(f"  Аномальных последовательностей: {stats.get('anomalous_sequences', 'N/A')}")
    lines.append(f"  Процент аномалий: {stats.get('anomaly_rate', 0):.1f}%")
    lines.append("")
    
    # Ошибки реконструкции
    errors = report.get("reconstruction_errors", {})
    if errors:
        lines.append("ОШИБКИ РЕКОНСТРУКЦИИ:")
        lines.append(f"  Средняя ошибка: {errors.get('mean', 0):.6f}")
        lines.append(f"  Максимальная ошибка: {errors.get('max', 0):.6f}")
        lines.append(f"  Минимальная ошибка: {errors.get('min', 0):.6f}")
        lines.append(f"  Стандартное отклонение: {errors.get('std', 0):.6f}")
        lines.append("")
    
    # Обнаруженные признаки
    detected_signs = gma.get("detected_signs", [])
    if detected_signs:
        lines.append("ОБНАРУЖЕННЫЕ ПРИЗНАКИ:")
        for sign in detected_signs:
            lines.append(f"  • {sign}")
        lines.append("")
    
    # Рекомендации
    recommendations = report.get("recommendations", [])
    if recommendations:
        lines.append("РЕКОМЕНДАЦИИ:")
        for rec in recommendations:
            lines.append(f"  {rec}")
        lines.append("")
    
    # Детальный анализ (если есть)
    detailed = report.get("detailed_analysis", {})
    if detailed:
        lines.append("ДЕТАЛЬНЫЙ АНАЛИЗ:")
        
        # Асимметрия
        asymmetry = detailed.get("asymmetry", {})
        if asymmetry.get("has_asymmetry", False):
            lines.append("  Асимметрия движений: Обнаружена")
            findings = asymmetry.get("findings", [])
            for finding in findings:
                lines.append(f"    - {finding.get('description', 'N/A')}")
        
        # Анализ суставов
        joint_analysis = detailed.get("joint_analysis", {})
        findings = joint_analysis.get("findings", [])
        if findings:
            lines.append("  Анализ суставов:")
            for finding in findings:
                lines.append(f"    - {finding.get('description', 'N/A')}")
        
        # Скорость движений
        speed_analysis = detailed.get("speed_analysis", {})
        findings = speed_analysis.get("findings", [])
        if findings:
            lines.append("  Скорость движений:")
            for finding in findings:
                lines.append(f"    - {finding.get('description', 'N/A')}")
        
        # Амплитуда движений
        amplitude_analysis = detailed.get("amplitude_analysis", {})
        findings = amplitude_analysis.get("findings", [])
        if findings:
            lines.append("  Амплитуда движений:")
            for finding in findings:
                lines.append(f"    - {finding.get('description', 'N/A')}")
        
        lines.append("")
    
    lines.append("=" * 70)
    lines.append(f"Дата анализа: {report.get('analysis_date', 'N/A')}")
    lines.append("=" * 70)
    
    return "\n".join(lines)
