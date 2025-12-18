
import argparse
import time
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import yaml

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich import box

from models.autoencoder_advanced import BidirectionalLSTMAutoencoder
from models.anomaly_detector import AnomalyDetector
from utils.data_augmentation import DataAugmentation
from utils.data_loader import MiniRGBDDataLoader
from utils.pose_processor import PoseProcessor

console = Console()


def load_config(config_path: str) -> Dict:
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
    with console.status(f"[cyan]Подготовка данных для {split}...", spinner="dots"):
        if split == "train":
            images, keypoints = data_loader.load_train_data(max_frames_per_seq)
        elif split == "val":
            images, keypoints = data_loader.load_val_data(max_frames_per_seq)
        elif split == "test":
            images, keypoints = data_loader.load_test_data(max_frames_per_seq)
        else:
            raise ValueError(f"Неизвестный split: {split}")
        
        console.print(f"[green]✓[/green] Загружено [bold]{len(keypoints)}[/bold] кадров для {split}")
        
        keypoints_filtered = [kp for kp in keypoints if kp is not None]
        
        if len(keypoints_filtered) == 0:
            console.print(f"[red]✗[/red] Нет валидных ключевых точек для {split}")
            return []
        
        sequences = pose_processor.process_keypoints(keypoints_filtered)
        
        if split == "train" and augmentation is not None:
            console.print("[yellow]→[/yellow] Применение аугментации данных...")
            augmented_sequences = []
            for seq in sequences:
                augmented_seq = augmentation.augment_keypoints(seq, apply_all=True)
                augmented_sequences.append(augmented_seq)
            sequences.extend(augmented_sequences)
            console.print(f"[green]✓[/green] После аугментации: [bold]{len(sequences)}[/bold] последовательностей")
        
        console.print(f"[green]✓[/green] Создано [bold]{len(sequences)}[/bold] последовательностей для {split}")
    return sequences


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = True,
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
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
        
        if progress and task_id is not None:
            progress.update(task_id, advance=1, loss=loss.item())
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            
            batch = batch.to(device)
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            total_loss += loss.item()
            num_batches += 1
            
            if progress and task_id is not None:
                progress.update(task_id, advance=1, loss=loss.item())
    
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
    
    # Заголовок
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]🚀 Обучение модели Bidirectional LSTM Autoencoder[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED
    ))
    console.print()
    
    # Загрузка конфигурации
    with console.status("[cyan]Загрузка конфигурации...", spinner="dots"):
        config = load_config(args.config)
        console.print(f"[green]✓[/green] Конфигурация загружена из [bold]{args.config}[/bold]")
    
    # Проверка устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        console.print("[red]✗[/red] [bold]ОШИБКА:[/bold] Требуется GPU для обучения!")
        raise RuntimeError("Требуется GPU для обучения!")
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    console.print(f"[green]✓[/green] Устройство: [bold cyan]{gpu_name}[/bold cyan] ({gpu_memory:.1f} GB)")
    console.print()
    
    # Подготовка путей
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Инициализация загрузчиков данных
    console.print(Panel("[bold]📊 Инициализация загрузчиков данных[/bold]", border_style="blue"))
    
    data_loader = MiniRGBDDataLoader(
        data_root=config["data"]["mini_rgbd_path"],
        train_sequences=config["data"]["train_sequences"],
        val_sequences=config["data"]["val_sequences"],
        test_sequences=config["data"]["test_sequences"],
        model_complexity=config["pose"]["model_complexity"],
        min_detection_confidence=config["pose"]["min_detection_confidence"],
        min_tracking_confidence=config["pose"]["min_tracking_confidence"],
    )
    
    pose_processor = PoseProcessor(
        sequence_length=config["pose"]["sequence_length"],
        sequence_stride=config["pose"]["sequence_stride"],
        normalize=config["pose"]["normalize"],
        normalize_relative_to=config["pose"]["normalize_relative_to"],
        target_hip_distance=config["pose"].get("target_hip_distance"),
        normalize_by_body=config["pose"].get("normalize_by_body", False),
        rotate_to_canonical=config["pose"].get("rotate_to_canonical", False),
    )
    
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
        console.print("[yellow]→[/yellow] Аугментация данных [bold]включена[/bold]")
    else:
        console.print("[dim]→[/dim] Аугментация данных [dim]выключена[/dim]")
    
    console.print()
    
    # Подготовка данных
    console.print(Panel("[bold]🔄 Подготовка данных[/bold]", border_style="yellow"))
    
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
        console.print("[red]✗[/red] [bold]ОШИБКА:[/bold] Нет данных для обучения! Проверьте путь к датасету в config.yaml")
        raise ValueError("Нет данных для обучения! Проверьте путь к датасету в config.yaml")
    
    # Преобразование в тензоры
    train_array = np.array(train_sequences)
    val_array = np.array(val_sequences)
    
    # Преобразуем форму из [N, seq_len, keypoints, coords] в [N, seq_len, features]
    # где features = keypoints * coords
    if len(train_array.shape) == 4:
        # [N, seq_len, keypoints, coords] -> [N, seq_len, keypoints * coords]
        train_array = train_array.reshape(train_array.shape[0], train_array.shape[1], -1)
        val_array = val_array.reshape(val_array.shape[0], val_array.shape[1], -1)
    
    train_tensor = torch.FloatTensor(train_array)
    val_tensor = torch.FloatTensor(val_array)
    
    console.print(f"[green]✓[/green] Размер train данных: [bold]{train_tensor.shape}[/bold]")
    console.print(f"[green]✓[/green] Размер val данных: [bold]{val_tensor.shape}[/bold]")
    console.print()
    
    # Создание DataLoader
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
    console.print(Panel("[bold]🧠 Создание модели[/bold]", border_style="magenta"))
    
    model = BidirectionalLSTMAutoencoder(
        input_size=config["model"]["input_size"],
        sequence_length=config["pose"]["sequence_length"],
        encoder_hidden_sizes=config["model"]["encoder_hidden_sizes"],
        decoder_hidden_sizes=config["model"]["decoder_hidden_sizes"],
        latent_size=config["model"]["latent_size"],
        num_attention_heads=4,
        dropout=config["model"].get("encoder_dropout", 0.2),
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    console.print(f"[green]✓[/green] Модель создана")
    console.print(f"   • Всего параметров: [bold]{total_params:,}[/bold]")
    console.print(f"   • Обучаемых параметров: [bold]{trainable_params:,}[/bold]")
    console.print()
    
    # Оптимизатор и планировщик
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    scheduler = None
    if config["training"]["scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["num_epochs"],
        )
        console.print("[yellow]→[/yellow] Планировщик: [bold]CosineAnnealingLR[/bold]")
    else:
        console.print("[dim]→[/dim] Планировщик: [dim]не используется[/dim]")
    
    criterion = nn.MSELoss()
    use_amp = config["training"].get("use_amp", True)
    scaler = GradScaler() if use_amp else None
    
    if use_amp:
        console.print("[yellow]→[/yellow] Mixed Precision Training: [bold]включено[/bold]")
    console.print()
    
    # Возобновление обучения
    start_epoch = 0
    best_val_loss = float("inf")
    
    if args.resume:
        console.print(f"[cyan]→[/cyan] Возобновление обучения из [bold]{args.resume}[/bold]")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        console.print(f"[green]✓[/green] Обучение возобновлено с эпохи [bold]{start_epoch}[/bold]")
        console.print()
    
    # Обучение
    console.print(Panel.fit(
        f"[bold green]🎯 Начало обучения[/bold green]\n"
        f"Эпох: [bold]{config['training']['num_epochs']}[/bold] | "
        f"Batch size: [bold]{config['training']['batch_size']}[/bold] | "
        f"LR: [bold]{config['training']['learning_rate']}[/bold]",
        border_style="green",
        box=box.ROUNDED
    ))
    console.print()
    
    # Таблица для метрик
    metrics_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    metrics_table.add_column("Эпоха", style="cyan", justify="center")
    metrics_table.add_column("Train Loss", style="yellow", justify="right")
    metrics_table.add_column("Val Loss", style="green", justify="right")
    metrics_table.add_column("LR", style="blue", justify="right")
    metrics_table.add_column("Статус", justify="center")
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        epoch_start_time = time.time()
        
        # Обучение
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("• Loss: {task.fields[loss]:.6f}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            train_task = progress.add_task(
                f"[cyan]Epoch {epoch+1}/{config['training']['num_epochs']} - Обучение",
                total=len(train_loader),
                loss=0.0
            )
            
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device, scaler, use_amp,
                progress, train_task
            )
        
        # Валидация
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("• Loss: {task.fields[loss]:.6f}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            val_task = progress.add_task(
                "[green]Валидация",
                total=len(val_loader),
                loss=0.0
            )
            
            val_loss = validate(model, val_loader, criterion, device, progress, val_task)
        
        # Обновление планировщика
        if scheduler:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time
        
        # Определение статуса
        is_best = val_loss < best_val_loss
        status = "[bold green]★ Лучшая![/bold green]" if is_best else "[dim]—[/dim]"
        
        # Добавление в таблицу
        metrics_table.add_row(
            str(epoch + 1),
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            f"{current_lr:.6f}",
            status
        )
        
        # Сохранение лучшей модели
        if is_best:
            best_val_loss = val_loss
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
        
        # Периодическое сохранение
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
        
        # Вывод таблицы каждые 5 эпох или на последней
        if (epoch + 1) % 5 == 0 or (epoch + 1) == config["training"]["num_epochs"]:
            console.print()
            console.print(metrics_table)
            console.print()
    
    total_time = time.time() - training_start_time
    
    console.print()
    console.print(Panel.fit(
        f"[bold green]✓ Обучение завершено![/bold green]\n"
        f"Время обучения: [bold]{total_time/60:.1f} минут[/bold]\n"
        f"Лучшая Val Loss: [bold green]{best_val_loss:.6f}[/bold green]",
        border_style="green",
        box=box.ROUNDED
    ))
    console.print()
    
    # Создание детектора аномалий
    console.print(Panel("[bold]🔍 Создание детектора аномалий[/bold]", border_style="cyan"))
    
    with console.status("[cyan]Вычисление порога аномалий...", spinner="dots"):
        detector = AnomalyDetector(
            model=model,
            device=device,
            threshold_percentile=config["anomaly"]["threshold_percentile"],
        )
        
        val_array_for_detector = np.array(val_sequences)
        # Преобразуем форму из [N, seq_len, keypoints, coords] в [N, seq_len, features]
        if len(val_array_for_detector.shape) == 4:
            val_array_for_detector = val_array_for_detector.reshape(
                val_array_for_detector.shape[0], 
                val_array_for_detector.shape[1], 
                -1
            )
        val_tensor_for_detector = torch.FloatTensor(val_array_for_detector).to(device)
        threshold = detector.fit_threshold(val_tensor_for_detector)
        
        console.print(f"[green]✓[/green] Порог аномалии: [bold cyan]{threshold:.6f}[/bold cyan]")
        
        detector_path = checkpoint_path.parent / "anomaly_detector_advanced.pt"
        detector.save(detector_path)
        
        console.print(f"[green]✓[/green] Детектор сохранен: [bold]{detector_path}[/bold]")
    
    console.print()
    console.print(Panel.fit(
        "[bold green]🎉 Готово! Модель готова к использованию[/bold green]\n\n"
        f"[cyan]📁 Файлы сохранены:[/cyan]\n"
        f"   • Модель: [bold]{checkpoint_path}[/bold]\n"
        f"   • Детектор: [bold]{detector_path}[/bold]",
        border_style="green",
        box=box.ROUNDED
    ))
    console.print()


if __name__ == "__main__":
    main()
