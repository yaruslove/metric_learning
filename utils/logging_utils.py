"""
Утилиты для логирования.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from torch.utils.tensorboard import SummaryWriter


def setup_logging(experiment_dir: Path) -> logging.Logger:
    """
    Настраивает логирование в файл и консоль.
    
    Args:
        experiment_dir: Директория эксперимента
        
    Returns:
        Настроенный логгер
    """
    log_file = experiment_dir / "logs" / "training.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Настраиваем формат логирования
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Создаем обработчики
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Настраиваем логгер
    logger = logging.getLogger("deepfake_detector")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def setup_tensorboard(experiment_dir: Path, enabled: bool = True) -> Optional[SummaryWriter]:
    """
    Настраивает TensorBoard для логирования метрик.
    
    Args:
        experiment_dir: Директория эксперимента
        enabled: Включить TensorBoard
        
    Returns:
        Объект SummaryWriter или None, если TensorBoard отключен
    """
    if not enabled:
        return None
    
    tb_dir = experiment_dir / "tensorboard"
    os.makedirs(tb_dir, exist_ok=True)
    
    return SummaryWriter(tb_dir)


def log_batch_metrics(
    writer: Optional[SummaryWriter], 
    metrics: Dict[str, float], 
    global_step: int
) -> None:
    """
    Логирует метрики батча в TensorBoard.
    
    Args:
        writer: Объект SummaryWriter
        metrics: Словарь с метриками
        global_step: Глобальный шаг
    """
    if writer is None:
        return
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"Batch/{key}", value, global_step)


def log_epoch_metrics(
    writer: Optional[SummaryWriter], 
    metrics: Dict[str, float], 
    epoch: int
) -> None:
    """
    Логирует метрики эпохи в TensorBoard.
    
    Args:
        writer: Объект SummaryWriter
        metrics: Словарь с метриками
        epoch: Номер эпохи
    """
    if writer is None:
        return
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"Epoch/{key}", value, epoch)


def log_validation_metrics(
    writer: Optional[SummaryWriter], 
    metrics: Dict[str, float], 
    epoch: int
) -> None:
    """
    Логирует метрики валидации в TensorBoard.
    
    Args:
        writer: Объект SummaryWriter
        metrics: Словарь с метриками
        epoch: Номер эпохи
    """
    if writer is None:
        return
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"Validation/{key}", value, epoch)


def log_hyperparameters(
    writer: Optional[SummaryWriter], 
    config: Dict[str, Any]
) -> None:
    """
    Логирует гиперпараметры в TensorBoard.
    
    Args:
        writer: Объект SummaryWriter
        config: Конфигурация
    """
    if writer is None:
        return
    
    # Извлекаем основные гиперпараметры
    hparams = {}
    
    # Модель
    hparams["model"] = config["model"]["name"]
    
    # Оптимизатор
    hparams["optimizer"] = config["optimizer"]["name"]
    hparams["lr"] = config["optimizer"]["lr"]
    
    # Функция потерь
    hparams["loss"] = config["loss"]["name"]
    hparams["margin"] = config["loss"].get("margin", 0)
    
    # Майнер
    hparams["miner"] = config["miner"]["name"]
    
    # Сэмплер
    hparams["sampler"] = config["sampler"]["name"]
    hparams["n_labels"] = config["sampler"]["n_labels"]
    hparams["n_instances"] = config["sampler"]["n_instances"]
    
    # Планировщик
    hparams["scheduler"] = config["scheduler"]["name"]
    
    # Другие параметры обучения
    hparams["batch_size"] = config["training"]["batch_size"]
    hparams["epochs"] = config["training"]["epochs"]
    
    writer.add_hparams(hparams, {})
