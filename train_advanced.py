"""
Обучение продвинутой модели (Bidirectional LSTM + Attention) с улучшенной нормализацией.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import yaml

from models.autoencoder_advanced import BidirectionalLSTMAutoencoder
from models.anomaly_detector import AnomalyDetector
from utils.data_augmentation import DataAugmentation
from utils.data_loader import MiniRGBDDataLoader
from utils.pose_processor import PoseProcessor

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
) -> List[np.ndarray]:
    """
    Подготовить данные для обучения/валидации/теста.
    
    Args:
        data_loader: Загрузчик данных
        pose_processor: Процессор поз
        split: Раздел данных ("train", "val", "test")
        max_frames_per_seq: Максимальное количество кадров на последовательность
        augmentation: Объект аугментации (только для train)
    
    Returns:
        Список последовательностей
    """
    logger.info(f"Подготовка данных для {split}...")
    
    # Загрузка данных
    if split == "train":
        images, keypoints = data_loader.load_train_data(max_frames_per_seq)
    elif split == "val":
        images, keypoints = data_loader.load_val_data(max_frames_per_seq)
    elif split == "test":
        images, keypoints = data_loader.load_test_data(max_frames_per_seq)
    else:
        raise ValueError(f"Неизвестный split: {split}")
    
    logger.info(f"Загружено {len(keypoints)} кадров для {split}")
    
    # Фильтруем None значения
    keypoints_filtered = [kp for kp in keypoints if kp is not None]
    
    if len(keypoints_filtered) == 0:
        logger.warning(f"Нет валидных ключевых точек для {split}")
        return []
    
    # Обработка в последовательности
    sequences = pose_processor.process_keypoints(keypoints_filtered)
    
    # Аугментация данных (только для train)
    if split == "train" and augmentation is not None:
        logger.info("Применение аугментации данных...")
        augmented_sequences = []
        for seq in sequences:
            # Применяем аугментацию к последовательности
            augmented_seq = augmentation.augment_keypoints(seq, apply_all=True)
            augmented_sequences.append(augmented_seq)
        sequences.extend(augmented_sequences)
        logger.info(f"После аугментации: {len(sequences)} последовательностей")
    
    logger.info(f"Создано {len(sequences)} последовательностей для {split}")
    return sequences


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = True,
) -> float:
    """Обучить модель на одной эпохе."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        # DataLoader возвращает кортеж, извлекаем тензор
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                reconstructed, _ = model(batch)
                loss = criterion(reconstructed, batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Валидация модели."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # DataLoader возвращает кортеж, извлекаем тензор
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            
            batch = batch.to(device)
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Обучение продвинутой модели")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Путь к конфигурационному файлу",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model_advanced.pt",
        help="Путь для сохранения checkpoint",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Путь к checkpoint для возобновления обучения",
    )
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = load_config(args.config)
    
    # Проверка GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Требуется GPU для обучения!")
    
    logger.info(f"Используется устройство: {device} ({torch.cuda.get_device_name(0)})")
    
    # Создание директории для checkpoints
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
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
    
    # Инициализация процессора поз
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
    if config.get("augmentation", {}).get("enabled", False) and config["pose"].get("augmentation", {}).get("enabled", False):
        aug_config = config["pose"]["augmentation"]
        augmentation = DataAugmentation(
            noise_level=aug_config.get("noise_level", 0.02),
            asymmetry_range=tuple(aug_config.get("asymmetry_range", [-0.03, 0.03])),
            add_natural_variation=aug_config.get("add_natural_variation", True),
            rotation_range=aug_config.get("rotation_range", 5.0),
            scale_range=tuple(aug_config.get("scale_range", [0.95, 1.05])),
        )
        logger.info("Аугментация данных включена")
    
    # Подготовка данных
    max_frames = config["data"].get("max_frames_per_seq")
    
    train_sequences = prepare_data(
        data_loader, pose_processor, "train", max_frames, augmentation
    )
    val_sequences = prepare_data(
        data_loader, pose_processor, "val", max_frames, None
    )
    test_sequences = prepare_data(
        data_loader, pose_processor, "test", max_frames, None
    )
    
    if len(train_sequences) == 0:
        raise ValueError("Нет данных для обучения! Проверьте путь к датасету в config.yaml")
    
    # Преобразование в тензоры
    train_tensor = torch.FloatTensor(np.array(train_sequences))
    val_tensor = torch.FloatTensor(np.array(val_sequences))
    
    logger.info(f"Размер train данных: {train_tensor.shape}")
    logger.info(f"Размер val данных: {val_tensor.shape}")
    
    # DataLoader
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    
    # Создание модели
    model = BidirectionalLSTMAutoencoder(
        input_size=config["model"]["input_size"],
        sequence_length=config["pose"]["sequence_length"],
        encoder_hidden_sizes=config["model"]["encoder_hidden_sizes"],
        decoder_hidden_sizes=config["model"]["decoder_hidden_sizes"],
        latent_size=config["model"]["latent_size"],
        num_attention_heads=4,
        dropout=config["model"].get("encoder_dropout", 0.2),
    ).to(device)
    
    logger.info(f"Модель создана: {sum(p.numel() for p in model.parameters())} параметров")
    
    # Оптимизатор
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    # Scheduler
    scheduler = None
    if config["training"]["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["num_epochs"],
        )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Mixed precision
    use_amp = config["training"].get("use_amp", True)
    scaler = GradScaler() if use_amp else None
    
    # Возобновление обучения
    start_epoch = 0
    best_val_loss = float("inf")
    
    if args.resume:
        logger.info(f"Возобновление обучения из {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    # Обучение
    logger.info("Начало обучения...")
    
    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        # Обучение
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, use_amp
        )
        
        # Валидация
        val_loss = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        if scheduler:
            scheduler.step()
        
        logger.info(
            f"Epoch {epoch+1}/{config['training']['num_epochs']} - "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"Новая лучшая модель! Val Loss: {val_loss:.6f}")
            
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "config": config,
                },
                checkpoint_path,
            )
        
        # Сохранение каждые N эпох
        if (epoch + 1) % config["training"]["save_every"] == 0:
            checkpoint_path_epoch = checkpoint_path.parent / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "config": config,
                },
                checkpoint_path_epoch,
            )
    
    logger.info("Обучение завершено!")
    logger.info(f"Лучшая модель сохранена: {checkpoint_path}")
    
    # Создание детектора аномалий
    logger.info("Создание детектора аномалий...")
    
    detector = AnomalyDetector(
        model=model,
        device=device,
        threshold_percentile=config["anomaly"]["threshold_percentile"],
    )
    
    # Вычисление порога на validation данных
    val_tensor_for_detector = torch.FloatTensor(np.array(val_sequences)).to(device)
    threshold = detector.fit_threshold(val_tensor_for_detector)
    
    logger.info(f"Порог аномалии: {threshold:.6f}")
    
    # Сохранение детектора
    detector_path = checkpoint_path.parent / "anomaly_detector_advanced.pt"
    detector.save(detector_path)
    
    logger.info(f"Детектор сохранен: {detector_path}")
    logger.info("Готово! Модель готова к использованию.")
    logger.info(f"Файлы сохранены:")
    logger.info(f"  - Модель: {checkpoint_path}")
    logger.info(f"  - Детектор: {detector_path}")


if __name__ == "__main__":
    main()
