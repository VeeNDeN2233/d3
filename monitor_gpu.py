"""Скрипт для мониторинга GPU utilization во время обучения."""

import time
import subprocess
import sys

def get_gpu_utilization():
    """Получить GPU utilization через nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        if lines:
            parts = lines[0].split(', ')
            if len(parts) >= 3:
                gpu_util = int(parts[0])
                mem_used = int(parts[1])
                mem_total = int(parts[2])
                mem_percent = (mem_used / mem_total) * 100
                return gpu_util, mem_percent
    except Exception as e:
        return None, None
    return None, None

if __name__ == "__main__":
    print("Мониторинг GPU... (Ctrl+C для остановки)")
    print("=" * 60)
    
    try:
        while True:
            gpu_util, mem_percent = get_gpu_utilization()
            if gpu_util is not None:
                status = "✓" if gpu_util > 80 else "⚠"
                print(f"{status} GPU Utilization: {gpu_util}% | Memory: {mem_percent:.1f}%")
            else:
                print("⚠ Не удалось получить данные GPU")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nМониторинг остановлен")

