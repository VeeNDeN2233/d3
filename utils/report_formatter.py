
from typing import Dict

# Словарь для перевода названий суставов на русский
JOINT_NAMES_RU = {
    "leftThigh": "левое бедро",
    "rightThigh": "правое бедро",
    "leftCalf": "левая голень",
    "rightCalf": "правая голень",
    "leftFoot": "левая стопа",
    "rightFoot": "правая стопа",
    "leftToes": "левые пальцы ног",
    "rightToes": "правые пальцы ног",
    "leftShoulder": "левое плечо",
    "rightShoulder": "правое плечо",
    "leftUpperArm": "левое плечо",
    "rightUpperArm": "правое плечо",
    "leftForeArm": "левое предплечье",
    "rightForeArm": "правое предплечье",
    "leftHand": "левая кисть",
    "rightHand": "правая кисть",
    "leftFingers": "левые пальцы рук",
    "rightFingers": "правые пальцы рук",
    "neck": "шея",
    "head": "голова",
    "spine": "позвоночник",
    "spine1": "позвоночник",
    "spine2": "позвоночник",
}

def translate_joint_name(joint_name: str) -> str:
    """Переводит название сустава на русский"""
    return JOINT_NAMES_RU.get(joint_name, joint_name)


def format_medical_report(report: Dict) -> str:
    if not report:
        return "Ошибка: Отчет пуст"
    
    lines = []
    lines.append("=" * 70)
    lines.append("ОТЧЕТ ПО ОЦЕНКЕ ОБЩИХ ДВИЖЕНИЙ (GMA)")
    lines.append("=" * 70)
    lines.append("")
    

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

        anomaly = report.get("anomaly_detection", {})
        risk_level = anomaly.get("risk_level", "unknown").upper()
        risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "UNKNOWN": "⚪"}
        lines.append("РЕЗУЛЬТАТ ОЦЕНКИ:")
        lines.append(f"  {risk_emoji.get(risk_level, '⚪')} Риск двигательных нарушений: {risk_level}")
        lines.append("")
    

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
    

    stats = report.get("statistics", {})
    lines.append("СТАТИСТИКА АНАЛИЗА:")
    lines.append(f"  Проанализировано последовательностей: {stats.get('total_sequences', 'N/A')}")
    lines.append(f"  Аномальных последовательностей: {stats.get('anomalous_sequences', 'N/A')}")
    lines.append(f"  Процент аномалий: {stats.get('anomaly_rate', 0):.1f}%")
    lines.append("")
    

    errors = report.get("reconstruction_errors", {})
    if errors:
        lines.append("ОШИБКИ РЕКОНСТРУКЦИИ:")
        lines.append(f"  Средняя ошибка: {errors.get('mean', 0):.6f}")
        lines.append(f"  Максимальная ошибка: {errors.get('max', 0):.6f}")
        lines.append(f"  Минимальная ошибка: {errors.get('min', 0):.6f}")
        lines.append(f"  Стандартное отклонение: {errors.get('std', 0):.6f}")
        lines.append("")
    

    detected_signs = gma.get("detected_signs", [])
    if detected_signs:
        lines.append("ОБНАРУЖЕННЫЕ ПРИЗНАКИ:")
        for sign in detected_signs:
            lines.append(f"  • {sign}")
        lines.append("")
    

    recommendations = report.get("recommendations", [])
    if recommendations:
        lines.append("РЕКОМЕНДАЦИИ:")
        for rec in recommendations:
            lines.append(f"  {rec}")
        lines.append("")
    

    detailed = report.get("detailed_analysis", {})
    if detailed:
        lines.append("=" * 70)
        lines.append("ДЕТАЛЬНЫЙ АНАЛИЗ ДВИЖЕНИЙ")
        lines.append("=" * 70)
        lines.append("")
        

        asymmetry = detailed.get("asymmetry", {})
        if asymmetry.get("has_asymmetry", False):
            lines.append("АСИММЕТРИЯ ДВИЖЕНИЙ:")
            lines.append("  ⚠️ Обнаружена асимметрия между левой и правой сторонами тела")
            findings = asymmetry.get("findings", [])
            for finding in findings:
                desc = finding.get('description', 'N/A')
                confidence = finding.get('confidence', '')
                if confidence:
                    lines.append(f"    - {desc} (уверенность: {confidence})")
                else:
                    lines.append(f"    - {desc}")
            lines.append("")
        

        joint_analysis = detailed.get("joint_analysis", {})
        findings = joint_analysis.get("findings", [])
        affected_joints = joint_analysis.get("affected_joints", [])
        
        # Показываем анализ суставов, даже если findings пустой, но есть затронутые суставы
        if findings or affected_joints:
            lines.append("  АНАЛИЗ СУСТАВОВ И КОНЕЧНОСТЕЙ:")
            
            if findings:
                # Группируем по типу нарушения
                reduced_movements = [f for f in findings if f.get('type') == 'reduced_movement']
                high_speed = [f for f in findings if f.get('type') == 'high_speed']
                
                if reduced_movements:
                    lines.append("    Сниженная амплитуда движений:")
                    for finding in reduced_movements:
                        joint_en = finding.get('joint', 'N/A')
                        joint = translate_joint_name(joint_en)
                        severity = finding.get('severity', 'unknown')
                        confidence = finding.get('confidence', 'unknown')
                        data = finding.get('data', {})
                        reduction = data.get('reduction_percent', 0)
                        z_score = data.get('z_score', 0)
                        
                        severity_emoji = "🔴" if severity == "high" else ("🟡" if severity == "medium" else "⚪")
                        severity_text = "высокая" if severity == "high" else ("средняя" if severity == "medium" else "низкая")
                        lines.append(f"      {severity_emoji} {joint}: снижение амплитуды на {reduction:.1f}% (степень: {severity_text}, z-score: {z_score:.2f}, уверенность: {confidence})")
                
                if high_speed:
                    lines.append("    Повышенная скорость движений:")
                    for finding in high_speed:
                        joint_en = finding.get('joint', 'N/A')
                        joint = translate_joint_name(joint_en)
                        severity = finding.get('severity', 'unknown')
                        confidence = finding.get('confidence', 'unknown')
                        data = finding.get('data', {})
                        ratio = data.get('ratio', 1.0)
                        z_score = data.get('z_score', 0)
                        
                        severity_emoji = "🔴" if severity == "high" else ("🟡" if severity == "medium" else "⚪")
                        severity_text = "высокая" if severity == "high" else ("средняя" if severity == "medium" else "низкая")
                        lines.append(f"      {severity_emoji} {joint}: увеличение скорости в {ratio:.2f}x (степень: {severity_text}, z-score: {z_score:.2f}, уверенность: {confidence})")
            
            # Показываем все затронутые суставы
            affected_joints_en = joint_analysis.get("affected_joints", [])
            if affected_joints_en:
                affected_joints_ru = [translate_joint_name(j) for j in affected_joints_en]
                lines.append(f"    Всего затронутых суставов: {len(affected_joints_ru)}")
                lines.append(f"    Список: {', '.join(affected_joints_ru)}")
        else:
            # Если нет конкретных findings, но есть аномалии, показываем общую информацию
            if detailed.get("has_anomalies", False):
                lines.append("  АНАЛИЗ СУСТАВОВ: Обнаружены нарушения движений, требуют детального рассмотрения")
        

        speed_analysis = detailed.get("speed_analysis", {})
        findings = speed_analysis.get("findings", [])
        if findings:
            lines.append("  Скорость движений:")
            for finding in findings:
                lines.append(f"    - {finding.get('description', 'N/A')}")
        

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
