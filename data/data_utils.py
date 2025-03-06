"""
Утилиты для работы с данными.
"""
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional

from oml import datasets as d
from oml.registry import get_transforms_for_pretrained
from oml.samplers import BalanceSampler, CategoryBalanceSampler


def get_transforms(model_name: str) -> Tuple:
    """
    Получает трансформации для модели.
    
    Args:
        model_name: Название модели
        
    Returns:
        Кортеж из (transform_train, transform_val)
    """
    transform, _ = get_transforms_for_pretrained(model_name)
    return transform, transform


def load_datasets(config: Dict[str, Any]) -> Tuple:
    """
    Загружает датасеты для обучения и валидации.
    
    Args:
        config: Конфигурация
        
    Returns:
        Кортеж из (train_dataset, val_dataset, transform_train, transform_val)
    """
    # Получаем трансформации
    transform_train, transform_val = get_transforms(config["model"]["name"])
    
    # Загружаем данные
    df_train = pd.read_csv(config["data"]["train_path"])
    df_val = pd.read_csv(config["data"]["val_path"])
    
    # Создаем датасеты
    train_dataset = d.ImageLabeledDataset(df_train, transform=transform_train)
    val_dataset = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform_val)
    
    return train_dataset, val_dataset, transform_train, transform_val


def get_sampler(config: Dict[str, Any], dataset) -> torch.utils.data.Sampler:
    """
    Создает сэмплер согласно конфигурации.
    
    Args:
        config: Конфигурация
        dataset: Датасет для сэмплирования
        
    Returns:
        Сэмплер
    """
    sampler_name = config["sampler"]["name"]
    
    if sampler_name == "BalanceSampler":
        return BalanceSampler(
            dataset.get_labels(), 
            n_labels=config["sampler"].get("n_labels", 4), 
            n_instances=config["sampler"].get("n_instances", 8)
        )
    elif sampler_name == "CategoryBalanceSampler":
        return CategoryBalanceSampler(
            dataset.get_labels(), 
            dataset.get_label2category(), 
            n_labels=config["sampler"].get("n_labels", 4), 
            n_instances=config["sampler"].get("n_instances", 8), 
            n_categories=config["sampler"].get("n_categories", 2)
        )
    else:
        raise ValueError(f"Неподдерживаемый сэмплер: {sampler_name}")


def create_train_dataloader(
    config: Dict[str, Any],
    dataset,
    sampler: Optional[torch.utils.data.Sampler] = None
) -> DataLoader:
    """
    Создает DataLoader для обучения.
    
    Args:
        config: Конфигурация
        dataset: Датасет
        sampler: Сэмплер (опционально)
        
    Returns:
        DataLoader
    """
    if sampler is None:
        sampler = get_sampler(config, dataset)
    
    return DataLoader(
        dataset, 
        batch_sampler=sampler,
        num_workers=config["training"].get("num_workers", 4)
    )


def create_val_dataloader(
    config: Dict[str, Any],
    dataset
) -> DataLoader:
    """
    Создает DataLoader для валидации.
    
    Args:
        config: Конфигурация
        dataset: Датасет
        
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset, 
        batch_size=config["validation"].get("batch_size", 32),
        shuffle=False,
        num_workers=config["validation"].get("num_workers", 4)
    )
