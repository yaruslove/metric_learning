"""
Утилиты для управления экспериментами.
"""
import os
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from config.config_utils import copy_config


def create_experiment_dir(config: Dict[str, Any]) -> Path:
    """
    Создает директорию для эксперимента с уникальным хешем и временной меткой.
    
    Args:
        config: Конфигурация эксперимента
        
    Returns:
        Путь к директории эксперимента
    """
    # Создаем хеш на основе важных параметров конфигурации
    config_str = str(config)
    hash_obj = hashlib.md5(config_str.encode())
    hash_str = hash_obj.hexdigest()[:8]
    
    # Формируем метку времени
    timestamp = datetime.now().strftime("%d" + "d" + "%m" + "m" + "%Y" + "_" + "%H" + "h" + "%M" + "m" + "%S")
    
    # Формируем название директории
    experiment_name = f"{hash_str}_{timestamp}"
    
    # Создаем директорию
    save_dir = Path(config["experiment"]["save_dir"])
    experiment_dir = save_dir / experiment_name
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Создаем поддиректории для моделей, логов и результатов
    os.makedirs(experiment_dir / "models", exist_ok=True)
    os.makedirs(experiment_dir / "logs", exist_ok=True)
    os.makedirs(experiment_dir / "tensorboard", exist_ok=True)
    
    return experiment_dir


def setup_experiment(config_path: str) -> tuple:
    """
    Настраивает эксперимент - создает директории, копирует конфиг.
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        Кортеж из (config, experiment_dir)
    """
    from config.config_utils import load_config
    
    # Загружаем конфигурацию
    config = load_config(config_path)
    
    # Создаем директорию для эксперимента
    experiment_dir = create_experiment_dir(config)
    
    # Копируем конфигурацию
    config_dst = experiment_dir / "config.yaml"
    copy_config(config_path, config_dst)
    
    return config, experiment_dir


def create_results_dataframe() -> pd.DataFrame:
    """
    Создает пустой DataFrame для результатов тренировки.
    
    Returns:
        DataFrame для результатов
    """
    columns = [
        "epoch", 
        "train_loss", 
        "val_loss", 
        "eer_val", 
        "eer_threshold", 
        "map@10", 
        "cmc@1", 
        "cmc@5", 
        "lr"
    ]
    
    return pd.DataFrame(columns=columns)


def update_results(
    results_df: pd.DataFrame, 
    epoch: int, 
    train_loss: float, 
    val_metrics: Dict[str, float],
    lr: float
) -> pd.DataFrame:
    """
    Обновляет DataFrame с результатами, добавляя новую строку.
    
    Args:
        results_df: DataFrame с результатами
        epoch: Номер эпохи
        train_loss: Значение функции потерь на тренировочном наборе
        val_metrics: Словарь с метриками на валидационном наборе
        lr: Текущая скорость обучения
        
    Returns:
        Обновленный DataFrame
    """
    # Создаем строку с данными текущей эпохи
    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "lr": lr
    }
    
    # Добавляем метрики валидации
    if val_metrics:
        # Добавляем валидационную функцию потерь, если она доступна
        if "loss" in val_metrics:
            row["val_loss"] = val_metrics["loss"]
        
        row["eer_val"] = val_metrics.get("eer", None)
        row["eer_threshold"] = val_metrics.get("eer_threshold", None)
        row["map@10"] = val_metrics.get("map@10", None)
        row["cmc@1"] = val_metrics.get("cmc@1", None)
        row["cmc@5"] = val_metrics.get("cmc@5", None)
    
    # Если запись с таким epoch уже существует, обновляем её
    existing_row = results_df[results_df["epoch"] == epoch]
    if len(existing_row) > 0:
        # Удаляем существующую запись
        results_df = results_df[results_df["epoch"] != epoch]
    
    # Добавляем новую строку в DataFrame (современный подход вместо append)
    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    
    # Сортируем по epoch для сохранения порядка
    results_df = results_df.sort_values("epoch").reset_index(drop=True)
    
    return results_df


def save_results(results_df: pd.DataFrame, save_path: Path) -> None:
    """
    Сохраняет результаты в CSV-файл.
    
    Args:
        results_df: DataFrame с результатами
        save_path: Путь для сохранения
    """
    # Создаем директорию, если она не существует
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Сохраняем в CSV, заменяем NaN на пустую строку
    results_df.to_csv(save_path, index=False, float_format='%.6f', na_rep='')