"""
Обучение продвинутой модели (Bidirectional LSTM + Attention) с улучшенной нормализацией.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
) -> torch.Tensor:
    """Подготовить данные для обучения/валидации/теста."""
    logger.info(f"Подготовка данных для {split}...")
    
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
    
    # Применяем аугментацию только для train split
    if split == "train" and augmentation is not None:
        logger.info(f"Применение аугментации к {split} данным...")
        augmented_sequences = []
        for seq in sequences:
            augmented_seq = np.array([
                augmentation.augment_keypoints(frame, apply_all=True)
                for frame in seq
            ])
            augmented_sequences.append(augmented_seq)
        sequences = augmented_sequences
    
    # Преобразуем в плоские векторы
    flattened_sequences = [
        pose_processor.flatten_sequence(seq) for seq in sequences
    ]
    
    # Конвертируем в тензор
    sequences_tensor = torch.FloatTensor(np.array(flattened_sequences))
    
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
    """Одна эпоха обучения."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (batch,) in enumerate(train_loader):
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                reconstructed, _ = model(batch)
                loss = nn.functional.mse_loss(reconstructed, batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            reconstructed, _ = model(batch)
            loss = nn.functional.mse_loss(reconstructed, batch)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    """Валидация модели."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch, in val_loader:
            batch = batch.to(device)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    reconstructed, _ = model(batch)
                    loss = nn.functional.mse_loss(reconstructed, batch)
            else:
                reconstructed, _ = model(batch)
                loss = nn.functional.mse_loss(reconstructed, batch)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Обучение Bidirectional LSTM + Attention модели")
    parser.add_argument("--config", type=str, default="config.yaml", help="Путь к конфигурации")
    parser.add_argument("--model", type=str, default="bidir_lstm", choices=["bidir_lstm", "transformer"],
                       help="Тип модели")
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    config = load_config(args.config)
    
    # Проверяем GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Требуется GPU для обучения!")
    
    logger.info(f"Используется устройство: {device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
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
    
    # Инициализация аугментации
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
        logger.info("Аугментация данных включена")
    
    # Подготовка данных
    max_frames_per_seq = config.get("data", {}).get("max_frames_per_seq", None)
    train_data = prepare_data(data_loader, pose_processor, "train", max_frames_per_seq, augmentation)
    val_data = prepare_data(data_loader, pose_processor, "val", max_frames_per_seq, None)
    
    # Создаем DataLoader
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    
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
    
    # Инициализация модели
    if args.model == "bidir_lstm":
        model = BidirectionalLSTMAutoencoder(
            input_size=config["model"]["input_size"],
            sequence_length=config["pose"]["sequence_length"],
            encoder_hidden_sizes=[256, 128, 64],
            decoder_hidden_sizes=[64, 128, 256, 75],
            latent_size=64,
            num_attention_heads=4,
            dropout=0.2,
        ).to(device)
    else:
        raise ValueError(f"Модель {args.model} не поддерживается")
    
    logger.info(f"Модель: {args.model}")
    logger.info(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    # Оптимизатор и scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["num_epochs"],
    )
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if config["training"]["use_amp"] else None
    
    # Обучение
    best_val_loss = float("inf")
    save_dir = Path(config["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("НАЧАЛО ОБУЧЕНИЯ")
    logger.info("=" * 60)
    
    for epoch in range(1, config["training"]["num_epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, config["training"]["use_amp"])
        val_loss = validate(model, val_loader, device, config["training"]["use_amp"])
        
        scheduler.step()
        
        logger.info(
            f"Epoch {epoch}/{config['training']['num_epochs']}: "
            f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
        )
        
        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "config": config,
            }
            torch.save(checkpoint, save_dir / "best_model_advanced.pt")
            logger.info(f"✅ Сохранена лучшая модель (val_loss={val_loss:.6f})")
        
        # Сохранение checkpoint
        if epoch % config["training"]["save_every"] == 0:
            torch.save(checkpoint, save_dir / f"checkpoint_epoch_{epoch}_advanced.pt")
    
    # Создание детектора аномалий
    logger.info("Создание детектора аномалий...")
    detector = AnomalyDetector(
        model=model,
        device=device,
        threshold_percentile=config["anomaly"]["threshold_percentile"],
    )
    
    # Вычисление порога на validation данных
    detector.fit_threshold(val_data.to(device))
    
    # Сохранение детектора
    detector.save(save_dir / "anomaly_detector_advanced.pt")
    logger.info(f"✅ Детектор сохранен: {save_dir / 'anomaly_detector_advanced.pt'}")
    logger.info(f"Порог аномалии: {detector.threshold:.6f}")
    
    logger.info("=" * 60)
    logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    logger.info("=" * 60)
    logger.info(f"Лучший val_loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
