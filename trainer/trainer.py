"""
Модуль для обучения модели.
"""
import torch
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple

from utils.checkpoint_utils import save_checkpoint
from utils.logging_utils import log_batch_metrics, log_epoch_metrics, log_validation_metrics
from utils.experiment_utils import update_results, save_results
from validation.validator import validate, is_better_checkpoint

logger = logging.getLogger("deepfake_detector")


class Trainer:
    """Класс для обучения модели."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion,
        device: torch.device,
        config: Dict[str, Any],
        experiment_dir: Path,
        writer,
        results_df: pd.DataFrame
    ):
        """
        Инициализирует тренер.
        
        Args:
            model: Модель
            optimizer: Оптимизатор
            scheduler: Планировщик скорости обучения
            criterion: Функция потерь
            device: Устройство
            config: Конфигурация
            experiment_dir: Директория эксперимента
            writer: TensorBoard writer
            results_df: DataFrame для результатов
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        self.experiment_dir = experiment_dir
        self.writer = writer
        self.results_df = results_df
        
        # Инициализируем метрики
        self.best_metric = float('inf') if self.config["experiment"]["metric_for_best"] == "eer" else float('inf')
        self.best_epoch = -1
        
        # Определяем, является ли метрика такой, что большее значение лучше
        self.higher_is_better = self.config["experiment"]["metric_for_best"] not in ["eer", "loss"]
    
    def train_epoch(
        self,
        epoch: int,
        train_loader,
        val_dataset
    ) -> Tuple[float, Dict[str, float]]:
        """
        Обучает модель в течение одной эпохи.
        
        Args:
            epoch: Номер эпохи
            train_loader: DataLoader для обучения
            val_dataset: Валидационный датасет
            
        Returns:
            Кортеж из (средний лосс за эпоху, метрики валидации)
        """
        logger.info(f"Начало эпохи {epoch+1}/{self.config['training']['epochs']}")
        
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Создаем прогресс-бар
        pbar = tqdm(train_loader)
        pbar.set_description(f"Эпоха {epoch+1}/{self.config['training']['epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            # Прямой проход
            inputs = batch["input_tensors"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            embeddings = self.model(inputs)
            loss = self.criterion(embeddings, labels)
            
            # Обратный проход
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Обновляем метрики
            epoch_loss += loss.item()
            num_batches += 1
            
            # Получаем текущий LR
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Обновляем прогресс-бар
            log_dict = {
                **self.criterion.last_logs,
                'lr': current_lr
            }
            pbar.set_postfix(log_dict)
            
            # Логируем метрики батча в TensorBoard
            if self.writer:
                global_step = epoch * len(train_loader) + batch_idx
                log_dict.update({'loss': loss.item()})
                log_batch_metrics(self.writer, log_dict, global_step)
        
        # Вычисляем среднюю потерю за эпоху
        avg_epoch_loss = epoch_loss / num_batches
        logger.info(f"Эпоха {epoch+1} - Средняя потеря: {avg_epoch_loss:.4f}")
        
        # Логируем метрики эпохи
        epoch_metrics = {'loss': avg_epoch_loss, 'lr': current_lr}
        log_epoch_metrics(self.writer, epoch_metrics, epoch)
        
        # Валидация
        val_metrics = {}
        if (epoch + 1) % self.config["training"].get("validate_every", 1) == 0:
            val_metrics = validate(
                model=self.model,
                val_dataset=val_dataset,
                device=self.device,
                config=self.config,
                epoch=epoch,
                criterion=self.criterion  # Передаем критерий для расчета val_loss
            )
            log_validation_metrics(self.writer, val_metrics, epoch)
        
        # Обновляем планировщик
        if self.config["scheduler"]["name"] == "ReduceLROnPlateau":
            # Если используем ReduceLROnPlateau, используем val_loss если доступен, иначе train_loss
            val_loss = val_metrics.get("loss", avg_epoch_loss)
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()
        
        return avg_epoch_loss, val_metrics
    
    def update_best_model(
        self,
        epoch: int,
        avg_loss: float,
        val_metrics: Dict[str, float]
    ) -> bool:
        """
        Обновляет информацию о лучшей модели.
        
        Args:
            epoch: Номер эпохи
            avg_loss: Средняя потеря за эпоху
            val_metrics: Метрики валидации
            
        Returns:
            True, если текущая модель является лучшей
        """
        is_best = False
        metric_for_best = self.config["experiment"].get("metric_for_best", "eer")
        
        if metric_for_best == "loss":
            # Для потери, проверяем сначала val_loss, если доступен, иначе train_loss
            if "loss" in val_metrics:
                val_loss = val_metrics["loss"]
                if val_loss < self.best_metric:
                    self.best_metric = val_loss
                    self.best_epoch = epoch
                    is_best = True
                    logger.info(f"Новая лучшая validation loss: {self.best_metric:.4f}")
            else:
                # Если val_loss недоступен, используем train_loss
                if avg_loss < self.best_metric:
                    self.best_metric = avg_loss
                    self.best_epoch = epoch
                    is_best = True
                    logger.info(f"Новая лучшая train loss: {self.best_metric:.4f}")
        else:
            # Для метрик валидации (EER, mAP, etc.)
            if is_better_checkpoint(
                val_metrics, 
                self.best_metric, 
                metric_for_best, 
                higher_is_better=self.higher_is_better
            ):
                self.best_metric = val_metrics[metric_for_best]
                self.best_epoch = epoch
                is_best = True
                logger.info(f"Новый лучший {metric_for_best}: {self.best_metric:.4f}")
        
        return is_best
    
    def train(
        self,
        train_loader,
        val_dataset,
        start_epoch: int = 0
    ) -> Dict[str, Any]:
        """
        Обучает модель на заданное количество эпох.
        
        Args:
            train_loader: DataLoader для обучения
            val_dataset: Валидационный датасет
            start_epoch: Начальная эпоха
            
        Returns:
            Словарь с результатами обучения
        """
        num_epochs = self.config["training"]["epochs"]
        
        for epoch in range(start_epoch, num_epochs):
            # Обучение на одну эпоху
            avg_loss, val_metrics = self.train_epoch(epoch, train_loader, val_dataset)
            
            # Проверка, является ли это лучшей моделью
            is_best = self.update_best_model(epoch, avg_loss, val_metrics)
            
            # Обновляем результаты
            self.results_df = update_results(
                self.results_df, 
                epoch + 1, 
                avg_loss, 
                val_metrics, 
                self.optimizer.param_groups[0]['lr']
            )
            
            # Сохраняем результаты после каждой эпохи
            results_path = self.experiment_dir / "results.csv"
            save_results(self.results_df, results_path)
            logger.info(f"Результаты эпохи сохранены в {results_path}")
            
            # Сохраняем чекпоинт
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics=val_metrics,
                experiment_dir=self.experiment_dir,
                config=self.config,
                best_metric=self.best_metric,
                is_best=is_best
            )
        
        # Финальная валидация
        final_metrics = validate(
            model=self.model,
            val_dataset=val_dataset,
            device=self.device,
            config=self.config,
            epoch=num_epochs,
            criterion=self.criterion  # Передаем критерий для расчета val_loss
        )
        
        # Обновляем результаты финальной эпохи
        self.results_df = update_results(
            self.results_df, 
            num_epochs, 
            avg_loss, 
            final_metrics, 
            self.optimizer.param_groups[0]['lr']
        )
        save_results(self.results_df, self.experiment_dir / "results.csv")
        
        # Сохраняем финальную модель
        final_path = self.experiment_dir / "models" / "final_model.pth"
        torch.save(self.model.state_dict(), final_path)
        logger.info(f"Обучение завершено. Финальная модель сохранена в {final_path}")
        
        metric_for_best = self.config["experiment"]["metric_for_best"]
        logger.info(f"Лучшая модель на эпохе {self.best_epoch+1} с {metric_for_best}={self.best_metric:.4f}")
        
        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "final_metrics": final_metrics
        }