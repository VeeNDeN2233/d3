
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np


JOINT_NAMES = [
    "global", "leftThigh", "rightThigh", "spine", "leftCalf", "rightCalf",
    "spine1", "leftFoot", "rightFoot", "spine2", "leftToes", "rightToes",
    "neck", "leftShoulder", "rightShoulder", "head",
    "leftUpperArm", "rightUpperArm", "leftForeArm", "rightForeArm",
    "leftHand", "rightHand", "leftFingers", "rightFingers", "noseVertex"
]


LEFT_JOINTS = {
    "arm": [13, 16, 18, 20, 22],
    "leg": [1, 4, 7, 10],
}
RIGHT_JOINTS = {
    "arm": [14, 17, 19, 21, 23],
    "leg": [2, 5, 8, 11],
}

logger = logging.getLogger(__name__)


def analyze_joint_errors(
    sequences: np.ndarray,
    reconstruction_errors: np.ndarray,
    threshold: float,
    normal_statistics: Optional[Dict] = None,
    age_weeks: Optional[float] = None,
    analyze_all_sequences: bool = False,
) -> Dict:
    if len(sequences) == 0:
        return {}
    

    anomalous_mask = reconstruction_errors > threshold
    normal_mask = ~anomalous_mask
    


    if analyze_all_sequences:

        analysis_mask = np.ones(len(sequences), dtype=bool)

        if np.sum(anomalous_mask) > 0:

            pass
        else:


            pass
    else:

        analysis_mask = anomalous_mask
        if np.sum(analysis_mask) == 0:
            return {
                "has_anomalies": False,
                "message": "Аномалий не обнаружено"
            }
    
    if np.sum(analysis_mask) == 0:
        return {
            "has_anomalies": False,
            "message": "Нет последовательностей для анализа"
        }
    

    if normal_statistics is None:
        try:
            from utils.normal_statistics import get_normal_statistics
            normal_statistics = get_normal_statistics()
            logger.info("Нормальные статистики загружены из тренировочных данных MINI-RGBD")
        except Exception as e:
            logger.warning(f"Не удалось загрузить нормальные статистики: {e}, используем данные из видео")
            normal_statistics = None
    

    analysis_sequences = sequences[analysis_mask]
    



    n_seqs, seq_len, _ = sequences.shape
    sequences_reshaped = sequences.reshape(n_seqs, seq_len, 25, 3)
    
    analysis_reshaped = sequences_reshaped[analysis_mask]
    

    analysis_amplitudes = np.std(analysis_reshaped, axis=1)
    analysis_amplitude_mean = np.mean(analysis_amplitudes, axis=0)
    analysis_amplitude_total = np.linalg.norm(analysis_amplitude_mean, axis=1)
    

    analysis_velocities = []
    for seq in analysis_reshaped:
        diffs = np.diff(seq, axis=0)
        velocities = np.linalg.norm(diffs, axis=2)
        analysis_velocities.append(np.mean(velocities, axis=0))
    analysis_velocity_mean = np.mean(analysis_velocities, axis=0) if analysis_velocities else np.zeros(25)
    

    anomalous_amplitude_total = analysis_amplitude_total
    anomalous_velocity_mean = analysis_velocity_mean
    

    if normal_statistics:
        normal_amplitude_total = np.array(normal_statistics["joint_amplitudes"])
        normal_amplitude_std = np.array(normal_statistics["joint_amplitudes_std"])
        normal_velocity_mean = np.array(normal_statistics["joint_velocities"])
        normal_velocity_std = np.array(normal_statistics["joint_velocities_std"])
        normal_left_right_ratios = normal_statistics["left_right_ratios"]
    else:

        normal_sequences = sequences[~anomalous_mask] if np.sum(~anomalous_mask) > 0 else sequences
        normal_reshaped = sequences_reshaped[~anomalous_mask] if np.sum(~anomalous_mask) > 0 else sequences_reshaped
        
        normal_amplitudes = np.std(normal_reshaped, axis=1)
        normal_amplitude_mean = np.mean(normal_amplitudes, axis=0)
        normal_amplitude_total = np.linalg.norm(normal_amplitude_mean, axis=1)
        normal_amplitude_std = np.std([np.linalg.norm(amp, axis=1) for amp in normal_amplitudes], axis=0)
        
        normal_velocities = []
        for seq in normal_reshaped:
            diffs = np.diff(seq, axis=0)
            velocities = np.linalg.norm(diffs, axis=2)
            normal_velocities.append(np.mean(velocities, axis=0))
        normal_velocity_mean = np.mean(normal_velocities, axis=0) if normal_velocities else np.zeros(25)
        normal_velocity_std = np.std(normal_velocities, axis=0) if normal_velocities else np.zeros(25)
        

        left_arm_amps = normal_amplitudes[:, LEFT_JOINTS["arm"]].mean(axis=1)
        right_arm_amps = normal_amplitudes[:, RIGHT_JOINTS["arm"]].mean(axis=1)
        arm_ratios = left_arm_amps / (right_arm_amps + 1e-6)
        
        left_leg_amps = normal_amplitudes[:, LEFT_JOINTS["leg"]].mean(axis=1)
        right_leg_amps = normal_amplitudes[:, RIGHT_JOINTS["leg"]].mean(axis=1)
        leg_ratios = left_leg_amps / (right_leg_amps + 1e-6)
        
        normal_left_right_ratios = {
            "arm": {"mean": float(np.mean(arm_ratios)), "std": float(np.std(arm_ratios))},
            "leg": {"mean": float(np.mean(leg_ratios)), "std": float(np.std(leg_ratios))},
        }
    

    asymmetry_analysis = analyze_asymmetry(
        anomalous_amplitude_total, normal_amplitude_total,
        normal_left_right_ratios, age_weeks
    )
    

    joint_analysis = analyze_specific_joints(
        anomalous_amplitude_total, normal_amplitude_total,
        anomalous_velocity_mean, normal_velocity_mean,
        normal_amplitude_std, normal_velocity_std
    )
    

    speed_analysis = analyze_movement_speed(
        anomalous_velocity_mean, normal_velocity_mean,
        normal_velocity_std
    )
    

    amplitude_analysis = analyze_movement_amplitude(
        anomalous_amplitude_total, normal_amplitude_total,
        normal_amplitude_std
    )
    


    has_anomalies = (
        np.sum(anomalous_mask) > 0 or
        amplitude_analysis.get("has_amplitude_anomalies", False) or
        asymmetry_analysis.get("has_asymmetry", False) or
        len(joint_analysis.get("findings", [])) > 0 or
        speed_analysis.get("has_speed_anomalies", False)
    )
    

    severity_score = {}
    if has_anomalies:
        severity_score = calculate_severity_score(
            asymmetry_analysis, joint_analysis, speed_analysis, amplitude_analysis
        )
    
    return {
        "has_anomalies": has_anomalies,
        "anomalous_sequences_count": int(np.sum(anomalous_mask)),
        "total_sequences": len(sequences),
        "asymmetry": asymmetry_analysis,
        "joint_analysis": joint_analysis,
        "speed_analysis": speed_analysis,
        "amplitude_analysis": amplitude_analysis,
        "severity_score": severity_score,
        "normal_statistics_source": "MINI-RGBD training data" if normal_statistics else "current video",
    }


def analyze_asymmetry(
    anomalous_amplitude: np.ndarray,
    normal_amplitude: np.ndarray,
    normal_left_right_ratios: Dict,
    age_weeks: Optional[float] = None,
) -> Dict:
    findings = []
    

    left_arm_amplitude = np.mean(anomalous_amplitude[LEFT_JOINTS["arm"]])
    right_arm_amplitude = np.mean(anomalous_amplitude[RIGHT_JOINTS["arm"]])
    

    arm_ratio_anomalous = left_arm_amplitude / (right_arm_amplitude + 1e-6)
    

    normal_arm_ratio_mean = normal_left_right_ratios["arm"]["mean"]
    normal_arm_ratio_std = normal_left_right_ratios["arm"]["std"]
    



    if normal_arm_ratio_std > 1e-6:
        z_score = abs(arm_ratio_anomalous - normal_arm_ratio_mean) / normal_arm_ratio_std
    else:

        z_score = abs(arm_ratio_anomalous - 1.0) * 2
    


    threshold_sigma = 2.0
    

    if age_weeks is not None and age_weeks < 12:
        threshold_sigma = 2.5
    
    if z_score > threshold_sigma:

        if arm_ratio_anomalous < 0.7:
            severity = "high" if z_score > 3.0 else "medium"
            findings.append({
                "type": "asymmetry",
                "body_part": "руки",
                "description": "Сниженная амплитуда движений левой руки по сравнению с правой",
                "severity": severity,
                "confidence": calculate_confidence(z_score, threshold_sigma),
                "data": {
                    "left_amplitude": float(left_arm_amplitude),
                    "right_amplitude": float(right_arm_amplitude),
                    "ratio": float(arm_ratio_anomalous),
                    "z_score": float(z_score),
                    "normal_ratio": float(normal_arm_ratio_mean),
                    "deviation_sigma": float(z_score),
                }
            })
        elif arm_ratio_anomalous > 1.43:
            severity = "high" if z_score > 3.0 else "medium"
            findings.append({
                "type": "asymmetry",
                "body_part": "руки",
                "description": "Сниженная амплитуда движений правой руки по сравнению с левой",
                "severity": severity,
                "confidence": calculate_confidence(z_score, threshold_sigma),
                "data": {
                    "left_amplitude": float(left_arm_amplitude),
                    "right_amplitude": float(right_arm_amplitude),
                    "ratio": float(arm_ratio_anomalous),
                    "z_score": float(z_score),
                    "normal_ratio": float(normal_arm_ratio_mean),
                    "deviation_sigma": float(z_score),
                }
            })
    

    left_leg_amplitude = np.mean(anomalous_amplitude[LEFT_JOINTS["leg"]])
    right_leg_amplitude = np.mean(anomalous_amplitude[RIGHT_JOINTS["leg"]])
    
    leg_ratio_anomalous = left_leg_amplitude / (right_leg_amplitude + 1e-6)
    normal_leg_ratio_mean = normal_left_right_ratios["leg"]["mean"]
    normal_leg_ratio_std = normal_left_right_ratios["leg"]["std"]
    
    if normal_leg_ratio_std > 1e-6:
        z_score_leg = abs(leg_ratio_anomalous - normal_leg_ratio_mean) / normal_leg_ratio_std
    else:
        z_score_leg = abs(leg_ratio_anomalous - 1.0) * 2
    
    if age_weeks is not None and age_weeks < 12:
        threshold_sigma_leg = 2.5
    else:
        threshold_sigma_leg = 2.0
    
    if z_score_leg > threshold_sigma_leg:
        if leg_ratio_anomalous < 0.7:
            severity = "high" if z_score_leg > 3.0 else "medium"
            findings.append({
                "type": "asymmetry",
                "body_part": "ноги",
                "description": "Сниженная амплитуда движений левой ноги по сравнению с правой",
                "severity": severity,
                "confidence": calculate_confidence(z_score_leg, threshold_sigma_leg),
                "data": {
                    "left_amplitude": float(left_leg_amplitude),
                    "right_amplitude": float(right_leg_amplitude),
                    "ratio": float(leg_ratio_anomalous),
                    "z_score": float(z_score_leg),
                    "normal_ratio": float(normal_leg_ratio_mean),
                    "deviation_sigma": float(z_score_leg),
                }
            })
        elif leg_ratio_anomalous > 1.43:
            severity = "high" if z_score_leg > 3.0 else "medium"
            findings.append({
                "type": "asymmetry",
                "body_part": "ноги",
                "description": "Сниженная амплитуда движений правой ноги по сравнению с левой",
                "severity": severity,
                "confidence": calculate_confidence(z_score_leg, threshold_sigma_leg),
                "data": {
                    "left_amplitude": float(left_leg_amplitude),
                    "right_amplitude": float(right_leg_amplitude),
                    "ratio": float(leg_ratio_anomalous),
                    "z_score": float(z_score_leg),
                    "normal_ratio": float(normal_leg_ratio_mean),
                    "deviation_sigma": float(z_score_leg),
                }
            })
    
    return {
        "findings": findings,
        "has_asymmetry": len(findings) > 0
    }


def calculate_confidence(z_score: float, threshold: float) -> str:
    if z_score > 3.0:
        return "высокая (p < 0.001)"
    elif z_score > 2.5:
        return "высокая (p < 0.01)"
    elif z_score > 2.0:
        return "средняя (p < 0.05)"
    else:
        return "низкая"


def analyze_specific_joints(
    anomalous_amplitude: np.ndarray,
    normal_amplitude: np.ndarray,
    anomalous_velocity: np.ndarray,
    normal_velocity: np.ndarray,
    normal_amplitude_std: np.ndarray,
    normal_velocity_std: np.ndarray,
) -> Dict:
    findings = []
    


    amplitude_z_scores = (normal_amplitude - anomalous_amplitude) / (normal_amplitude_std + 1e-6)

    low_amplitude_joints = np.where(amplitude_z_scores > 2.0)[0]
    
    for joint_idx in low_amplitude_joints:
        if joint_idx < len(JOINT_NAMES):
            joint_name = JOINT_NAMES[joint_idx]

            if joint_name not in ["global", "spine", "spine1", "spine2", "neck", "head", "noseVertex"]:
                z_score = amplitude_z_scores[joint_idx]
                severity = "high" if z_score > 3.0 else "medium"
                reduction_pct = (1 - anomalous_amplitude[joint_idx] / (normal_amplitude[joint_idx] + 1e-6)) * 100
                
                findings.append({
                    "type": "reduced_movement",
                    "joint": joint_name,
                    "description": f"Сниженная амплитуда движений {joint_name}",
                    "severity": severity,
                    "confidence": calculate_confidence(z_score, 2.0),
                    "data": {
                        "anomalous_amplitude": float(anomalous_amplitude[joint_idx]),
                        "normal_amplitude": float(normal_amplitude[joint_idx]),
                        "reduction_percent": float(reduction_pct),
                        "z_score": float(z_score),
                        "deviation_sigma": float(z_score),
                    }
                })
    

    velocity_z_scores = (anomalous_velocity - normal_velocity) / (normal_velocity_std + 1e-6)
    high_velocity_joints = np.where(velocity_z_scores > 2.0)[0]
    
    for joint_idx in high_velocity_joints:
        if joint_idx < len(JOINT_NAMES):
            joint_name = JOINT_NAMES[joint_idx]
            if joint_name not in ["global", "spine", "spine1", "spine2", "neck", "head", "noseVertex"]:
                z_score = velocity_z_scores[joint_idx]
                severity = "high" if z_score > 3.0 else "medium"
                speed_ratio = anomalous_velocity[joint_idx] / (normal_velocity[joint_idx] + 1e-6)
                
                findings.append({
                    "type": "high_speed",
                    "joint": joint_name,
                    "description": f"Повышенная скорость движений {joint_name}",
                    "severity": severity,
                    "confidence": calculate_confidence(z_score, 2.0),
                    "data": {
                        "anomalous_velocity": float(anomalous_velocity[joint_idx]),
                        "normal_velocity": float(normal_velocity[joint_idx]),
                        "ratio": float(speed_ratio),
                        "z_score": float(z_score),
                        "deviation_sigma": float(z_score),
                    }
                })
    
    return {
        "findings": findings,
        "affected_joints": [JOINT_NAMES[i] for i in low_amplitude_joints if i < len(JOINT_NAMES)]
    }


def analyze_movement_speed(
    anomalous_velocity: np.ndarray,
    normal_velocity: np.ndarray,
    normal_velocity_std: np.ndarray,
) -> Dict:
    findings = []
    

    total_anomalous_velocity = np.mean(anomalous_velocity)
    total_normal_velocity = np.mean(normal_velocity)
    total_normal_velocity_std = np.mean(normal_velocity_std)
    
    if total_normal_velocity_std > 1e-6:
        z_score_total = (total_anomalous_velocity - total_normal_velocity) / total_normal_velocity_std
    else:
        z_score_total = (total_anomalous_velocity - total_normal_velocity) / (total_normal_velocity + 1e-6) * 2
    
    if z_score_total > 2.0:
        severity = "high" if z_score_total > 3.0 else "medium"
        findings.append({
            "type": "overall_high_speed",
            "description": "Общая скорость движений выше нормы",
            "severity": severity,
            "confidence": calculate_confidence(z_score_total, 2.0),
            "data": {
                "anomalous_speed": float(total_anomalous_velocity),
                "normal_speed": float(total_normal_velocity),
                "ratio": float(total_anomalous_velocity / (total_normal_velocity + 1e-6)),
                "z_score": float(z_score_total),
                "deviation_sigma": float(z_score_total),
            }
        })
    elif z_score_total < -2.0:
        severity = "high" if z_score_total < -3.0 else "medium"
        findings.append({
            "type": "overall_low_speed",
            "description": "Общая скорость движений ниже нормы",
            "severity": severity,
            "confidence": calculate_confidence(abs(z_score_total), 2.0),
            "data": {
                "anomalous_speed": float(total_anomalous_velocity),
                "normal_speed": float(total_normal_velocity),
                "ratio": float(total_anomalous_velocity / (total_normal_velocity + 1e-6)),
                "z_score": float(z_score_total),
                "deviation_sigma": float(abs(z_score_total)),
            }
        })
    

    left_arm_velocity = np.mean(anomalous_velocity[LEFT_JOINTS["arm"]])
    right_arm_velocity = np.mean(anomalous_velocity[RIGHT_JOINTS["arm"]])
    left_arm_normal = np.mean(normal_velocity[LEFT_JOINTS["arm"]])
    right_arm_normal = np.mean(normal_velocity[RIGHT_JOINTS["arm"]])
    left_arm_std = np.mean(normal_velocity_std[LEFT_JOINTS["arm"]])
    right_arm_std = np.mean(normal_velocity_std[RIGHT_JOINTS["arm"]])
    
    if left_arm_std > 1e-6:
        z_score_left = (left_arm_velocity - left_arm_normal) / left_arm_std
        if z_score_left > 2.0:
            severity = "high" if z_score_left > 3.0 else "medium"
            findings.append({
                "type": "high_speed",
                "body_part": "левая рука",
                "description": "Повышенная скорость движений левой руки",
                "severity": severity,
                "confidence": calculate_confidence(z_score_left, 2.0),
                "data": {
                    "z_score": float(z_score_left),
                    "deviation_sigma": float(z_score_left),
                }
            })
    
    if right_arm_std > 1e-6:
        z_score_right = (right_arm_velocity - right_arm_normal) / right_arm_std
        if z_score_right > 2.0:
            severity = "high" if z_score_right > 3.0 else "medium"
            findings.append({
                "type": "high_speed",
                "body_part": "правая рука",
                "description": "Повышенная скорость движений правой руки",
                "severity": severity,
                "confidence": calculate_confidence(z_score_right, 2.0),
                "data": {
                    "z_score": float(z_score_right),
                    "deviation_sigma": float(z_score_right),
                }
            })
    
    return {
        "findings": findings,
        "has_speed_anomalies": len(findings) > 0
    }


def analyze_movement_amplitude(
    anomalous_amplitude: np.ndarray,
    normal_amplitude: np.ndarray,
    normal_amplitude_std: np.ndarray,
) -> Dict:
    findings = []
    

    total_anomalous_amplitude = np.mean(anomalous_amplitude)
    total_normal_amplitude = np.mean(normal_amplitude)
    total_normal_amplitude_std = np.mean(normal_amplitude_std)
    

    critical_amplitude_threshold = 0.01
    low_amplitude_threshold = 0.03
    

    critical_amplitude_drop = False
    moderate_amplitude_drop = False
    

    if total_anomalous_amplitude < critical_amplitude_threshold:
        critical_amplitude_drop = True
        reduction_pct = (1 - total_anomalous_amplitude / (total_normal_amplitude + 1e-6)) * 100
        findings.append({
            "type": "critical_absence_of_movement",
            "description": f"КРИТИЧЕСКОЕ СНИЖЕНИЕ: амплитуда {total_anomalous_amplitude:.4f} < {critical_amplitude_threshold} (практически отсутствие движений)",
            "severity": "high",
            "confidence": "высокая (абсолютный порог)",
            "data": {
                "anomalous_amplitude": float(total_anomalous_amplitude),
                "normal_amplitude": float(total_normal_amplitude),
                "reduction_percent": float(reduction_pct),
                "threshold": float(critical_amplitude_threshold),
            }
        })
    

    elif total_anomalous_amplitude < low_amplitude_threshold:
        moderate_amplitude_drop = True
        reduction_pct = (1 - total_anomalous_amplitude / (total_normal_amplitude + 1e-6)) * 100
        findings.append({
            "type": "moderate_reduced_amplitude",
            "description": f"СНИЖЕННАЯ АКТИВНОСТЬ: амплитуда {total_anomalous_amplitude:.4f} < {low_amplitude_threshold} (значительно сниженная активность)",
            "severity": "medium",
            "confidence": "высокая (абсолютный порог)",
            "data": {
                "anomalous_amplitude": float(total_anomalous_amplitude),
                "normal_amplitude": float(total_normal_amplitude),
                "reduction_percent": float(reduction_pct),
                "threshold": float(low_amplitude_threshold),
            }
        })
    

    if total_normal_amplitude_std > 1e-6:
        z_score_amp = (total_normal_amplitude - total_anomalous_amplitude) / total_normal_amplitude_std
    else:
        z_score_amp = (total_normal_amplitude - total_anomalous_amplitude) / (total_normal_amplitude + 1e-6) * 2
    

    if z_score_amp > 2.0 and not critical_amplitude_drop and not moderate_amplitude_drop:
        severity = "high" if z_score_amp > 3.0 else "medium"
        reduction_pct = (1 - total_anomalous_amplitude / (total_normal_amplitude + 1e-6)) * 100
        findings.append({
            "type": "overall_reduced_amplitude",
            "description": "Общая амплитуда движений снижена",
            "severity": severity,
            "confidence": calculate_confidence(z_score_amp, 2.0),
            "data": {
                "anomalous_amplitude": float(total_anomalous_amplitude),
                "normal_amplitude": float(total_normal_amplitude),
                "reduction_percent": float(reduction_pct),
                "z_score": float(z_score_amp),
                "deviation_sigma": float(z_score_amp),
            }
        })
    
    return {
        "findings": findings,
        "has_amplitude_anomalies": len(findings) > 0,
        "critical_amplitude_drop": critical_amplitude_drop,
        "moderate_amplitude_drop": moderate_amplitude_drop,
        "total_amplitude": float(total_anomalous_amplitude),
        "normal_total_amplitude": float(total_normal_amplitude),
        "amplitude_ratio": float(total_anomalous_amplitude / (total_normal_amplitude + 1e-6)),
    }


def calculate_severity_score(
    asymmetry_analysis: Dict,
    joint_analysis: Dict,
    speed_analysis: Dict,
    amplitude_analysis: Dict,
) -> Dict:
    score = 0
    breakdown = {}
    

    asymmetry_findings = asymmetry_analysis.get("findings", [])
    for finding in asymmetry_findings:
        z_score = finding.get("data", {}).get("deviation_sigma", 0)
        if z_score > 3.0:
            score += 3
            breakdown["asymmetry"] = "тяжелая"
        elif z_score > 2.5:
            score += 2
            breakdown["asymmetry"] = "средняя"
        elif z_score > 2.0:
            score += 1
            breakdown["asymmetry"] = "легкая"
    

    joint_findings = joint_analysis.get("findings", [])
    reduced_count = sum(1 for f in joint_findings if f["type"] == "reduced_movement")
    if reduced_count >= 5:
        score += 3
        breakdown["reduced_movement"] = "тяжелая (множественные суставы)"
    elif reduced_count >= 3:
        score += 2
        breakdown["reduced_movement"] = "средняя (несколько суставов)"
    elif reduced_count >= 1:
        score += 1
        breakdown["reduced_movement"] = "легкая"
    

    speed_findings = speed_analysis.get("findings", [])
    for finding in speed_findings:
        z_score = finding.get("data", {}).get("deviation_sigma", 0)
        if z_score > 3.0:
            score += 2
        elif z_score > 2.0:
            score += 1
    

    amplitude_findings = amplitude_analysis.get("findings", [])
    for finding in amplitude_findings:
        z_score = finding.get("data", {}).get("deviation_sigma", 0)
        if z_score > 3.0:
            score += 2
        elif z_score > 2.0:
            score += 1
    

    if score >= 5:
        severity_level = "ВЫСОКИЙ РИСК"
        color = "red"
    elif score >= 3:
        severity_level = "СРЕДНИЙ РИСК"
        color = "orange"
    else:
        severity_level = "НИЗКИЙ РИСК"
        color = "green"
    
    return {
        "total_score": score,
        "severity_level": severity_level,
        "color": color,
        "breakdown": breakdown,
    }

