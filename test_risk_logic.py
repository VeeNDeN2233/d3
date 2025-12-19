#!/usr/bin/env python3
"""Тестовый скрипт для проверки логики определения риска"""
threshold = 0.02010086178779602
max_error = 0.031631
mean_error = 0.011839
anomaly_rate = 6.2
anomalous_count = 177

print(f"Threshold: {threshold}")
print(f"Max error: {max_error}")
print(f"Mean error: {mean_error}")
print(f"Anomaly rate: {anomaly_rate}%")
print(f"Anomalous count: {anomalous_count}")
print()

if max_error > threshold * 2.0 or anomaly_rate > 30.0:
    risk_level = "high"
    print("RESULT: HIGH")
    print(f"  Reason: max_error={max_error:.6f} > threshold*2={threshold*2.0:.6f} OR anomaly_rate={anomaly_rate:.2f}% > 30%")
elif max_error > threshold * 1.5 or anomaly_rate > 15.0 or mean_error > threshold:
    risk_level = "medium"
    print("RESULT: MEDIUM")
    print(f"  Reason: max_error={max_error:.6f} > threshold*1.5={threshold*1.5:.6f} ({max_error > threshold * 1.5})")
    print(f"          OR anomaly_rate={anomaly_rate:.2f}% > 15% ({anomaly_rate > 15.0})")
    print(f"          OR mean_error={mean_error:.6f} > threshold={threshold} ({mean_error > threshold})")
elif anomaly_rate > 5.0 or mean_error > threshold * 0.8 or anomalous_count > 10:
    risk_level = "medium"
    print("RESULT: MEDIUM (легкие отклонения)")
    print(f"  Reason: anomaly_rate={anomaly_rate:.2f}% > 5% OR mean_error > threshold*0.8 OR anomalous_count={anomalous_count} > 10")
elif anomalous_count > 0:
    risk_level = "medium"
    print("RESULT: MEDIUM (отдельные аномалии)")
    print(f"  Reason: anomalous_count={anomalous_count} > 0")
else:
    risk_level = "low"
    print("RESULT: LOW")

print(f"\nFinal risk_level: {risk_level}")

