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


# def load_datasets(config: Dict[str, Any]) -> Tuple:
#     """
#     Загружает датасеты для обучения и валидации.
    
#     Args:
#         config: Конфигурация
        
#     Returns:
#         Кортеж из (train_dataset, val_dataset, transform_train, transform_val)
#     """
#     # Получаем трансформации
#     transform_train, transform_val = get_transforms(config["model"]["name"])
    
#     # Загружаем данные
#     df_train = pd.read_csv(config["data"]["train_path"])
#     df_val = pd.read_csv(config["data"]["val_path"])
    
#     # Создаем датасеты
#     train_dataset = d.ImageLabeledDataset(df_train, transform=transform_train)
#     val_dataset = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform_val)
    
#     return train_dataset, val_dataset, transform_train, transform_val



def load_datasets(config: Dict[str, Any]) -> Tuple:
    """
    Загружает датасеты для обучения и валидации.
    
    Args:
        config: Конфигурация
        
    Returns:
        Кортеж из (train_dataset, val_dataset, transform_train, transform_val, label_to_index)
    """
    # Получаем трансформации
    transform_train, transform_val = get_transforms(config["model"]["name"])
    
    # Загружаем данные
    df_train = pd.read_csv(config["data"]["train_path"])
    df_val = pd.read_csv(config["data"]["val_path"])
    
    # Проверяем, нужен ли ремаппинг меток для ArcFace
    need_remap = (config["loss"]["name"] == "ArcFaceLoss" and 
                 config["loss"].get("remap_labels2ArcFace", False))
    
    label_to_index = None
    
    if need_remap: # Только для случая loss = ArcFace 
        # Создаем отображение меток в индексы от 0 до num_classes-1
        unique_labels = sorted(df_train['label'].unique())
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Применяем отображение к данным
        df_train_mapped = df_train.copy()
        df_val_mapped = df_val.copy()
        
        df_train_mapped['original_label'] = df_train_mapped['label']
        df_train_mapped['label'] = df_train_mapped['label'].map(label_to_index)
        
        df_val_mapped['original_label'] = df_val_mapped['label']
        df_val_mapped['label'] = df_val_mapped['label'].map(label_to_index)
        
        # Создаем датасеты с ремаппингом
        train_dataset = d.ImageLabeledDataset(df_train_mapped, transform=transform_train)
        val_dataset = d.ImageQueryGalleryLabeledDataset(df_val_mapped, transform=transform_val)
        
        # Сохраняем отображение в датасетах для возможного восстановления
        train_dataset.label_to_index = label_to_index
        val_dataset.label_to_index = label_to_index
    else: # Для случая Triplet loss
        # Создаем датасеты без ремаппинга
        train_dataset = d.ImageLabeledDataset(df_train, transform=transform_train)
        val_dataset = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform_val)
    
    return train_dataset, val_dataset, transform_train, transform_val, label_to_index


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
