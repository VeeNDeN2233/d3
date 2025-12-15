"""
Скрипт обучения LSTM автоэнкодера для детекции аномалий движений младенцев.

Строгий тренировочный скрипт:
- ВСЕ тензоры на GPU: .cuda()
- Mixed precision: torch.cuda.amp
- No data leakage: разделение по последовательностям
- Сохранение checkpoints
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml

from models.autoencoder_gpu import LSTMAutoencoderGPU
from models.anomaly_detector import AnomalyDetector
from utils.data_augmentation import DataAugmentation
from utils.data_loader import MiniRGBDDataLoader
from utils.pose_processor import PoseProcessor
from utils.data_augmentation import DataAugmentation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Загрузить конфигурацию из YAML файла."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(
    data_loader: MiniRGBDDataLoader,
    pose_processor: PoseProcessor,
    split: str = "train",
    max_frames_per_seq: Optional[int] = None,
    augmentation: Optional[DataAugmentation] = None,
) -> torch.Tensor:
    """
    Подготовить данные для обучения/валидации/теста.
    
    Args:
        data_loader: Загрузчик данных
        pose_processor: Процессор позы
        split: "train", "val", или "test"
    
    Returns:
        Тензор последовательностей (N, seq_len, input_size)
    """
    logger.info(f"Подготовка данных для {split}...")
    
    # Загружаем данные в зависимости от split
    if split == "train":
        _, keypoints_list = data_loader.load_train_data(max_frames_per_seq)
    elif split == "val":
        _, keypoints_list = data_loader.load_val_data(max_frames_per_seq)
    elif split == "test":
        _, keypoints_list = data_loader.load_test_data(max_frames_per_seq)
    else:
        raise ValueError(f"Неизвестный split: {split}")
    
    # Обрабатываем ключевые точки
    sequences = pose_processor.process_keypoints(keypoints_list)
    
    if len(sequences) == 0:
        logger.warning(f"Нет последовательностей для {split}")
        return torch.empty(0, pose_processor.sequence_length, 75)
    
    # Преобразуем в плоские векторы
    flattened_sequences = [
        pose_processor.flatten_sequence(seq) for seq in sequences
    ]
    
    # Конвертируем в тензор
    sequences_tensor = torch.FloatTensor(np.array(flattened_sequences))
    
    # Применяем аугментацию только для train split
    if split == "train" and augmentation is not None and augmentation.enabled:
        logger.info(f"Применение аугментации к {split} данным...")
        sequences_np = sequences_tensor.numpy()
        augmented_np = augmentation.augment_sequence(sequences_np)
        sequences_tensor = torch.FloatTensor(augmented_np)
        logger.info(f"Аугментация применена: {sequences_tensor.shape}")
    
    logger.info(
        f"{split}: {len(sequences_tensor)} последовательностей, "
        f"форма {sequences_tensor.shape}"
    )
    
    return sequences_tensor


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    """
    Обучение на одной эпохе.
    
    Returns:
        Средняя ошибка на эпохе
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        batch = batch[0].to(device)  # (batch, seq_len, input_size)
        
        optimizer.zero_grad()
        
        # Forward pass с mixed precision
        if use_amp:
            with torch.cuda.amp.autocast():
                reconstructed, _ = model(batch)
                loss = nn.functional.mse_loss(batch, reconstructed)
        else:
            reconstructed, _ = model(batch)
            loss = nn.functional.mse_loss(batch, reconstructed)
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    """
    Валидация модели.
    
    Returns:
        Средняя ошибка на validation данных
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch[0].to(device)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    reconstructed, _ = model(batch)
                    loss = nn.functional.mse_loss(batch, reconstructed)
            else:
                reconstructed, _ = model(batch)
                loss = nn.functional.mse_loss(batch, reconstructed)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Обучение LSTM автоэнкодера на GPU")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Путь к config.yaml"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Путь к checkpoint для продолжения"
    )
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    config = load_config(args.config)
    
    # Проверяем GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU недоступен! Обучение требует GPU.")
    
    logger.info(f"Используется устройство: {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA версия: {torch.version.cuda}")
    
    # Проверка GPU памяти
    if torch.cuda.is_available():
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Создаем директории
    save_dir = Path(config["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Инициализация загрузчика данных
    data_loader = MiniRGBDDataLoader(
        data_root=config["data"]["mini_rgbd_path"],
        train_sequences=config["data"]["train_sequences"],
        val_sequences=config["data"]["val_sequences"],
        test_sequences=config["data"]["test_sequences"],
        model_complexity=config["pose"]["model_complexity"],
        min_detection_confidence=config["pose"]["min_detection_confidence"],
        min_tracking_confidence=config["pose"]["min_tracking_confidence"],
    )
    
    # Инициализация процессора позы
    pose_processor = PoseProcessor(
        sequence_length=config["pose"]["sequence_length"],
        sequence_stride=config["pose"]["sequence_stride"],
        normalize=config["pose"]["normalize"],
        normalize_relative_to=config["pose"]["normalize_relative_to"],
        target_hip_distance=config["pose"].get("target_hip_distance"),
        normalize_by_body=config["pose"].get("normalize_by_body", False),
        rotate_to_canonical=config["pose"].get("rotate_to_canonical", False),
    )
    
    # Инициализация аугментации (только для train)
    augmentation = None
    if config.get("pose", {}).get("augmentation", {}).get("enabled", False):
        aug_config = config["pose"]["augmentation"]
        augmentation = DataAugmentation(
            noise_level=aug_config.get("noise_level", 0.02),
            asymmetry_range=tuple(aug_config.get("asymmetry_range", [-0.03, 0.03])),
            add_natural_variation=aug_config.get("add_natural_variation", True),
            rotation_range=aug_config.get("rotation_range", 5.0),
            scale_range=tuple(aug_config.get("scale_range", [0.95, 1.05])),
        )
        logger.info("Аугментация данных включена для обучения")
    
    # Подготовка данных
    logger.info("=" * 60)
    logger.info("ПОДГОТОВКА ДАННЫХ (БЕЗ УТЕЧКИ ДАННЫХ)")
    logger.info("=" * 60)
    
    # Ограничение кадров для быстрого теста (можно убрать для полного обучения)
    max_frames_per_seq = config.get("data", {}).get("max_frames_per_seq", None)
    if max_frames_per_seq:
        logger.info(f"ОГРАНИЧЕНИЕ: {max_frames_per_seq} кадров на последовательность (ТЕСТОВЫЙ РЕЖИМ)")
    
    train_data = prepare_data(data_loader, pose_processor, "train", max_frames_per_seq, augmentation)
    val_data = prepare_data(data_loader, pose_processor, "val", max_frames_per_seq, None)  # Без аугментации
    test_data = prepare_data(data_loader, pose_processor, "test", max_frames_per_seq, None)  # Без аугментации
    
    # Создаем DataLoader
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,  # Windows может иметь проблемы с num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    
    # Инициализация модели
    model = LSTMAutoencoderGPU(
        input_size=config["model"]["input_size"],
        sequence_length=config["pose"]["sequence_length"],
        encoder_hidden_sizes=config["model"]["encoder_hidden_sizes"],
        decoder_hidden_sizes=config["model"]["decoder_hidden_sizes"],
        encoder_num_layers=config["model"]["encoder_num_layers"],
        decoder_num_layers=config["model"]["decoder_num_layers"],
        encoder_dropout=config["model"]["encoder_dropout"],
        decoder_dropout=config["model"]["decoder_dropout"],
        latent_size=config["model"]["latent_size"],
    )
    
    model = model.to_gpu()
    
    # Оптимизатор
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    # Scheduler
    if config["training"]["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["training"]["num_epochs"]
        )
    else:
        scheduler = None
    
    # Mixed precision scaler
    use_amp = config["training"]["use_amp"]
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Обучение
    logger.info("=" * 60)
    logger.info("НАЧАЛО ОБУЧЕНИЯ")
    logger.info("=" * 60)
    
    best_val_loss = float("inf")
    prev_val_loss = None
    patience_counter = 0
    start_epoch = 0
    
    # Resume from checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Resumed from epoch {start_epoch}")
    
    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        # Обучение
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        
        # Валидация
        val_loss = validate(model, val_loader, device, use_amp)
        
        # Scheduler step
        if scheduler:
            scheduler.step()
        
        # Мониторинг GPU
        gpu_memory = "N/A"
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            gpu_memory = f"{gpu_memory_allocated:.2f}/{gpu_memory_reserved:.2f} GB"
        
        logger.info(
            f"Epoch {epoch+1}/{config['training']['num_epochs']}: "
            f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
            f"lr={optimizer.param_groups[0]['lr']:.6f}, "
            f"GPU_mem={gpu_memory}"
        )
        
        # Проверка overfitting
        if prev_val_loss is not None and val_loss > prev_val_loss * 1.1:
            logger.warning(f"⚠️  ВОЗМОЖНЫЙ OVERFITTING: val_loss увеличился с {prev_val_loss:.6f} до {val_loss:.6f}")
        prev_val_loss = val_loss
        
        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint_path = save_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "config": config,
                },
                checkpoint_path,
            )
            logger.info(f"Сохранена лучшая модель: {checkpoint_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config["training"]["early_stopping_patience"]:
            logger.info(f"Early stopping на эпохе {epoch+1}")
            break
        
        # Периодическое сохранение
        if (epoch + 1) % config["training"]["save_every"] == 0:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "config": config,
                },
                checkpoint_path,
            )
    
    logger.info("=" * 60)
    logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    logger.info("=" * 60)
    
    # Создаем детектор аномалий и вычисляем порог на validation данных
    logger.info("Создание детектора аномалий...")
    detector = AnomalyDetector(
        model=model,
        device=device,
        threshold_percentile=config["anomaly"]["threshold_percentile"],
    )
    
    # Вычисляем порог ТОЛЬКО на validation данных
    threshold = detector.fit_threshold(val_data.to(device))
    
    # Сохраняем детектор
    detector_path = save_dir / "anomaly_detector.pt"
    detector.save(detector_path)
    logger.info(f"Детектор сохранен: {detector_path}")
    
    # Статистика
    stats = detector.get_statistics()
    logger.info("Статистика детектора:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()

