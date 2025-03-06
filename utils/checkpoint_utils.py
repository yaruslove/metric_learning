"""
Утилиты для работы с чекпоинтами моделей.
"""
import os
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("deepfake_detector")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, float],
    experiment_dir: Path,
    config: Dict[str, Any],
    best_metric: Optional[float] = None,
    is_best: bool = False
) -> None:
    """
    Сохраняет чекпоинт модели.
    
    Args:
        model: Модель
        optimizer: Оптимизатор
        scheduler: Планировщик скорости обучения
        epoch: Номер эпохи
        metrics: Метрики
        experiment_dir: Директория эксперимента
        config: Конфигурация
        best_metric: Лучшее значение метрики
        is_best: Является ли текущий чекпоинт лучшим
    """
    # Создаем директорию для сохранения моделей
    models_dir = experiment_dir / "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Создаем чекпоинт
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'best_metric': best_metric
    }
    
    # Периодическое сохранение
    # save_every = config["experiment"].get("save_every", 5)
    # if epoch % save_every == 0:
    #     torch.save(
    #         checkpoint,
    #         models_dir / f'checkpoint_epoch_{epoch}.pth'
    #     )
    
    # Сохраняем как лучшую, если это лучшая модель
    if is_best:
        torch.save(checkpoint, models_dir / 'best_model.pth')
        torch.save(model.state_dict(), models_dir / 'best_weights.pth')
        logger.info(f"Сохранена новая лучшая модель на эпохе {epoch}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    checkpoint_path: Optional[str] = None
) -> tuple:
    """
    Загружает состояние модели из чекпоинта.
    
    Args:
        model: Модель
        optimizer: Оптимизатор (опционально)
        scheduler: Планировщик скорости обучения (опционально)
        checkpoint_path: Путь к чекпоинту
        
    Returns:
        Кортеж из (model, optimizer, scheduler, epoch, metrics, best_metric)
    """
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        logger.warning(f"Чекпоинт {checkpoint_path} не найден")
        return model, optimizer, scheduler, 0, {}, None
    
    # Загружаем чекпоинт
    checkpoint = torch.load(checkpoint_path)
    
    # Загружаем состояние модели
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Загружаем состояние оптимизатора, если он передан
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Загружаем состояние планировщика, если он передан
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Получаем эпоху, метрики и лучшую метрику
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    best_metric = checkpoint.get('best_metric', None)
    
    logger.info(f"Загружен чекпоинт с эпохи {epoch}")
    
    return model, optimizer, scheduler, epoch, metrics, best_metric
