"""
Основной скрипт для запуска обучения модели.
"""
import os
import argparse
import torch
import logging
import random
import numpy as np
from pathlib import Path

from config.config_utils import load_config
from utils.experiment_utils import setup_experiment, create_results_dataframe
from utils.logging_utils import setup_logging, setup_tensorboard, log_hyperparameters
from data.data_utils import load_datasets, create_train_dataloader, get_sampler
from models.model_utils import setup_model_components
from trainer.trainer import Trainer
from utils.checkpoint_utils import load_checkpoint


def fix_seed(seed: int) -> None:
    """Фиксирует все генераторы случайных чисел для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Обучение модели распознавания лиц с защитой от DeepFake")
    parser.add_argument("--config", type=str, default="config.yaml", help="Путь к файлу конфигурации")
    parser.add_argument("--checkpoint", type=str, default=None, help="Путь к чекпоинту для продолжения обучения")
    parser.add_argument("--seed", type=int, default=None, help="Seed для воспроизводимости")
    return parser.parse_args()


def main():
    """Основная функция запуска обучения."""
    args = parse_args()
    
    # Настройка эксперимента
    config, experiment_dir = setup_experiment(args.config)
    
    # Настройка логирования
    logger = setup_logging(experiment_dir)
    writer = setup_tensorboard(
        experiment_dir, 
        enabled=config["experiment"].get("tensorboard", True)
    )
    
    # Логируем гиперпараметры
    log_hyperparameters(writer, config)
    
    # Устанавливаем устройство
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Используемое устройство: {device}")
    
    # Устанавливаем seed
    seed = args.seed if args.seed is not None else config["training"].get("seed", 0)
    fix_seed(seed)
    logger.info(f"Установлен seed: {seed}")
    
    # Загружаем данные
    train_dataset, val_dataset, _, _ = load_datasets(config)
    logger.info(f"Загружено {len(train_dataset)} тренировочных образцов")
    logger.info(f"Загружено {len(val_dataset)} валидационных образцов")
    
    # Настраиваем модель и компоненты
    model, optimizer, scheduler, criterion, miner = setup_model_components(config, device)
    logger.info(f"Модель {config['model']['name']} инициализирована")
    logger.info(f"Оптимизатор: {config['optimizer']['name']}")
    logger.info(f"Планировщик: {config['scheduler']['name']}")
    logger.info(f"Функция потерь: {config['loss']['name']}")
    logger.info(f"Майнер: {config['miner']['name']}")
    
    # Загружаем чекпоинт, если указан
    start_epoch = 0
    if args.checkpoint:
        model, optimizer, scheduler, start_epoch, _, best_metric = load_checkpoint(
            model, optimizer, scheduler, args.checkpoint
        )
        logger.info(f"Загружен чекпоинт с эпохи {start_epoch}, обучение продолжится с эпохи {start_epoch + 1}")
    
    # Создаем сэмплер и загрузчик данных
    sampler = get_sampler(config, train_dataset)
    train_loader = create_train_dataloader(config, train_dataset, sampler)
    logger.info(f"Настроен сэмплер: {config['sampler']['name']}")
    
    # Создаем DataFrame для результатов
    results_df = create_results_dataframe()
    
    # Создаем тренер
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=config,
        experiment_dir=experiment_dir,
        writer=writer,
        results_df=results_df
    )
    
    # Запускаем обучение
    logger.info("Начало обучения")
    results = trainer.train(train_loader, val_dataset, start_epoch)
    logger.info("Обучение завершено")
    
    # Закрываем writer TensorBoard
    if writer:
        writer.close()
    
    return results


if __name__ == "__main__":
    main()