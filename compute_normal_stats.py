"""Скрипт для вычисления нормальных статистик из тренировочных данных MINI-RGBD."""

import logging
from utils.normal_statistics import calculate_normal_statistics

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

if __name__ == "__main__":
    print("=" * 70)
    print("ВЫЧИСЛЕНИЕ НОРМАЛЬНЫХ СТАТИСТИК ИЗ ТРЕНИРОВОЧНЫХ ДАННЫХ MINI-RGBD")
    print("=" * 70)
    print()
    
    try:
        stats = calculate_normal_statistics(force_recalculate=True)
        
        print()
        print("=" * 70)
        print("СТАТИСТИКИ УСПЕШНО ВЫЧИСЛЕНЫ И СОХРАНЕНЫ!")
        print("=" * 70)
        print(f"  Образцов: {stats['sample_size']}")
        print(f"  Средняя амплитуда: {stats['overall_statistics']['amplitude_mean']:.6f} ± {stats['overall_statistics']['amplitude_std']:.6f}")
        print(f"  Средняя скорость: {stats['overall_statistics']['velocity_mean']:.6f} ± {stats['overall_statistics']['velocity_std']:.6f}")
        print()
        print("  Соотношения левая/правая:")
        print(f"    Руки: {stats['left_right_ratios']['arm']['mean']:.3f} ± {stats['left_right_ratios']['arm']['std']:.3f}")
        print(f"    Ноги: {stats['left_right_ratios']['leg']['mean']:.3f} ± {stats['left_right_ratios']['leg']['std']:.3f}")
        print()
        print(f"  Файл сохранен: checkpoints/normal_statistics.json")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

