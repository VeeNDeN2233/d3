"""
Сравнение разных архитектур моделей для детекции аномалий.
"""

import logging
from pathlib import Path

import torch
import yaml

from models.autoencoder_advanced import BidirectionalLSTMAutoencoder, TransformerAutoencoder
from models.autoencoder_gpu import LSTMAutoencoderGPU

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compare_model_architectures():
    """Сравнить разные архитектуры моделей."""
    logger.info("=" * 60)
    logger.info("СРАВНЕНИЕ АРХИТЕКТУР МОДЕЛЕЙ")
    logger.info("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    sequence_length = 30
    input_size = 75
    
    # Тестовые данные
    test_input = torch.randn(batch_size, sequence_length, input_size).to(device)
    
    models_to_test = [
        {
            "name": "Текущая LSTM",
            "model": LSTMAutoencoderGPU(
                input_size=input_size,
                sequence_length=sequence_length,
                encoder_hidden_sizes=[128, 64, 32],
                decoder_hidden_sizes=[64, 128, 75],
                latent_size=32,
            ).to(device),
            "description": "Базовая LSTM: [128,64,32] → 32 → [64,128,75]",
        },
        {
            "name": "Bidirectional LSTM + Attention",
            "model": BidirectionalLSTMAutoencoder(
                input_size=input_size,
                sequence_length=sequence_length,
                encoder_hidden_sizes=[256, 128, 64],
                decoder_hidden_sizes=[64, 128, 256, 75],
                latent_size=64,
                num_attention_heads=4,
            ).to(device),
            "description": "Bidirectional LSTM с attention: [256,128,64] → 64 → [64,128,256,75]",
        },
        {
            "name": "Transformer",
            "model": TransformerAutoencoder(
                input_size=input_size,
                sequence_length=sequence_length,
                d_model=128,
                nhead=8,
                num_encoder_layers=4,
                num_decoder_layers=4,
                latent_size=128,
            ).to(device),
            "description": "Transformer: d_model=128, 4 encoder/decoder layers, 8 heads",
        },
    ]
    
    results = []
    
    for model_info in models_to_test:
        name = model_info["name"]
        model = model_info["model"]
        description = model_info["description"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Тестирование: {name}")
        logger.info(f"Описание: {description}")
        logger.info(f"{'='*60}")
        
        # Подсчет параметров
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Параметры: {num_params:,} (trainable: {num_trainable:,})")
        
        # Тест forward pass
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            try:
                # Warmup
                _ = model(test_input)
                
                # Измерение времени
                torch.cuda.synchronize() if device.type == "cuda" else None
                import time
                start_time = time.time()
                
                for _ in range(10):
                    reconstructed, latent = model(test_input)
                
                torch.cuda.synchronize() if device.type == "cuda" else None
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10
                
                # Проверка формы выхода
                assert reconstructed.shape == test_input.shape, \
                    f"Неверная форма выхода: {reconstructed.shape} != {test_input.shape}"
                
                # Вычисление ошибки реконструкции (на тестовых данных)
                mse = torch.nn.functional.mse_loss(test_input, reconstructed)
                
                logger.info(f"✅ Forward pass успешен")
                logger.info(f"   Время (10 итераций): {avg_time*1000:.2f} ms")
                logger.info(f"   MSE (на тестовых данных): {mse.item():.6f}")
                logger.info(f"   Latent shape: {latent.shape}")
                
                results.append({
                    "name": name,
                    "description": description,
                    "num_params": num_params,
                    "num_trainable": num_trainable,
                    "avg_time_ms": avg_time * 1000,
                    "test_mse": mse.item(),
                    "latent_shape": list(latent.shape),
                    "success": True,
                })
                
            except Exception as e:
                logger.error(f"❌ Ошибка: {e}")
                results.append({
                    "name": name,
                    "description": description,
                    "num_params": num_params,
                    "num_trainable": num_trainable,
                    "success": False,
                    "error": str(e),
                })
    
    # Сравнение результатов
    logger.info("\n" + "=" * 60)
    logger.info("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    logger.info("=" * 60)
    
    logger.info(f"\n{'Модель':<30} {'Параметры':<15} {'Время (ms)':<12} {'Test MSE':<12} {'Статус'}")
    logger.info("-" * 90)
    
    for result in results:
        if result.get("success", False):
            logger.info(
                f"{result['name']:<30} "
                f"{result['num_params']:,<15} "
                f"{result['avg_time_ms']:.2f}<12 "
                f"{result['test_mse']:.6f}<12 "
                f"✅"
            )
        else:
            logger.info(
                f"{result['name']:<30} "
                f"{result['num_params']:,<15} "
                f"{'N/A':<12} "
                f"{'N/A':<12} "
                f"❌"
            )
    
    # Рекомендация
    successful_results = [r for r in results if r.get("success", False)]
    if successful_results:
        # Сортируем по test_mse (меньше = лучше на тестовых данных)
        best_model = min(successful_results, key=lambda x: x.get("test_mse", float("inf")))
        
        logger.info(f"\n✅ Рекомендуемая модель: {best_model['name']}")
        logger.info(f"   Test MSE: {best_model['test_mse']:.6f}")
        logger.info(f"   Параметры: {best_model['num_params']:,}")
        logger.info(f"   Время: {best_model['avg_time_ms']:.2f} ms")
    
    return results


if __name__ == "__main__":
    compare_model_architectures()

