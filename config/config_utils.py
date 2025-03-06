"""
Утилиты для работы с конфигурацией.
"""
import os
import yaml
import shutil
from pathlib import Path


def load_config(config_path: str) -> dict:
    """
    Загружает конфигурацию из YAML-файла.
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        Словарь с конфигурацией
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, save_path: str) -> None:
    """
    Сохраняет конфигурацию в YAML-файл.
    
    Args:
        config: Словарь с конфигурацией
        save_path: Путь для сохранения
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def copy_config(src_path: str, dst_path: str) -> None:
    """
    Копирует файл конфигурации.
    
    Args:
        src_path: Исходный путь
        dst_path: Целевой путь
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)
