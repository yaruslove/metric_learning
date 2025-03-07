"""
Модуль для валидации модели.
"""
import torch
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from torch.utils.data import DataLoader

from oml.inference import inference
from oml.metrics import calc_retrieval_metrics_rr
from oml.retrieval import RetrievalResults, AdaptiveThresholding

from metrics.metrics import calculate_eer, collect_similarity_scores, format_retrieval_metrics
# Импортируем функцию создания dataloader и получения сэмплера
from data.data_utils import get_sampler

logger = logging.getLogger("deepfake_detector")


# def calculate_val_loss(
#     model: torch.nn.Module,
#     val_dataset,
#     criterion,
#     device: torch.device,
#     config: Dict[str, Any],
# ) -> float:
#     """
#     Рассчитывает потери на валидационном наборе с использованием сэмплера из конфигурации.
    
#     Args:
#         model: Модель
#         val_dataset: Валидационный датасет
#         criterion: Функция потерь
#         device: Устройство
#         config: Конфигурация
        
#     Returns:
#         Значение потерь на валидационном наборе
#     """
#     # Проверяем, что критерий существует
#     assert criterion is not None, "Критерий (функция потерь) должен быть предоставлен"
    

    
#     # Создаем сэмплер из конфигурации
#     sampler = get_sampler(config, val_dataset)
    
#     # Создаем загрузчик данных с сэмплером
#     dataloader = torch.utils.data.DataLoader(
#         val_dataset, 
#         batch_sampler=sampler,
#         num_workers=config["validation"].get("num_workers", 4)
#     )
    
#     total_loss = 0.0
#     num_batches = 0
    
#     # Переводим модель в режим оценки
#     model.eval()
#     with torch.no_grad():
#         for batch in dataloader:
#             # Проверяем, что в батче есть метки
#             assert "labels" in batch, "В валидационном батче отсутствуют метки (labels)"
                
#             inputs = batch["input_tensors"].to(device)
#             labels = batch["labels"].to(device)
            
#             # Получаем эмбеддинги
#             embeddings = model(inputs)
            
#             # Проверка достаточного количества образцов для каждой метки
#             # Это избегает ошибки в _check_input_labels в oml/interfaces/miners.py
#             from collections import Counter
#             labels_counter = Counter(labels.cpu().numpy())
#             if not all(n > 1 for n in labels_counter.values()):
#                 raise AssertionError(
#                     "В валидационном батче недостаточно образцов на класс для формирования триплетов. "
#                     "Для каждого класса нужно минимум 2 образца. "
#                     f"Текущее распределение: {dict(labels_counter)}"
#                 )
            
#             # Вычисляем функцию потерь
#             loss = criterion(embeddings, labels)
            
#             # Обновляем счетчики
#             total_loss += loss.item()
#             num_batches += 1
            
#             # Очищаем память
#             del embeddings, inputs, labels
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
    
#     # Проверяем, что обработан хотя бы один батч
#     assert num_batches > 0, "Не удалось обработать ни одного валидационного батча"
    
#     # Возвращаем среднюю потерю
#     return total_loss / num_batches

def calculate_val_loss(
    model: torch.nn.Module,
    val_dataset,
    criterion,
    device: torch.device,
    config: Dict[str, Any],
) -> float:
    """
    Рассчитывает потери на валидационном наборе с ручным формированием батчей.
    """
    # Проверяем, что критерий существует
    assert criterion is not None, "Критерий (функция потерь) должен быть предоставлен"
    
    import torch
    import logging
    from collections import defaultdict, Counter
    
    logger = logging.getLogger("deepfake_detector")
    
    # Получаем метки и преобразуем их в numpy, если это тензор
    labels = val_dataset.get_labels()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Группируем индексы по меткам
    label_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        label_to_indices[label].append(i)
    
    # Оставляем только метки с минимум 2 экземплярами
    valid_labels = [label for label, indices in label_to_indices.items() if len(indices) >= 2]
    
    if len(valid_labels) < 2:
        logger.error("Недостаточно меток с 2+ экземплярами для валидации")
        return 0.0
    
    logger.info(f"Найдено {len(valid_labels)} меток с минимум 2 экземплярами")
    
    # Формируем батчи вручную
    batch_size = config["validation"].get("batch_size", 32)
    batches = []
    
    # Берем по 2 экземпляра каждой метки
    for label in valid_labels:
        indices = label_to_indices[label]
        # Берем первые 2 экземпляра каждой метки
        batches.append(indices[:2])
    
    # Объединяем маленькие батчи в более крупные
    merged_batches = []
    current_batch = []
    
    for batch in batches:
        if len(current_batch) + len(batch) > batch_size:
            merged_batches.append(current_batch)
            current_batch = batch
        else:
            current_batch.extend(batch)
    
    if current_batch:
        merged_batches.append(current_batch)
    
    logger.info(f"Сформировано {len(merged_batches)} батчей для валидации")
    
    # Переводим модель в режим оценки
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_indices in merged_batches:
            # Получаем данные для текущего батча
            batch_data = [val_dataset[i] for i in batch_indices]
            
            # Объединяем данные в единый батч
            inputs = torch.stack([item["input_tensors"] for item in batch_data]).to(device)
            batch_labels = torch.tensor([item["labels"] for item in batch_data]).to(device)
            
            # Получаем эмбеддинги
            embeddings = model(inputs)
            
            # Проверяем, что есть хотя бы 2 экземпляра каждой метки в батче
            label_counts = Counter(batch_labels.cpu().numpy())
            if all(count >= 2 for count in label_counts.values()):
                # Вычисляем функцию потерь
                loss = criterion(embeddings, batch_labels)
                total_loss += loss.item()
                num_batches += 1
            else:
                logger.warning(f"Пропуск батча с недостаточным количеством экземпляров на метку")
            
            # Очищаем память
            del embeddings, inputs, batch_labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Проверяем, что обработан хотя бы один батч
    if num_batches == 0:
        logger.warning("Не удалось обработать ни одного валидационного батча")
        return 0.0
    
    # Возвращаем среднюю потерю
    return total_loss / num_batches


def validate(
    model: torch.nn.Module,
    val_dataset,
    device: torch.device,
    config: Dict[str, Any],
    epoch: int,
    criterion: Optional[Any] = None
) -> Dict[str, float]:
    """Проводит валидацию модели и рассчитывает метрики."""
    logger.info(f"Начало валидации для эпохи {epoch}...")
    metrics = {}
    
    # Расчет метрик на основе эмбеддингов
    model.eval()
    with torch.no_grad():
        # Получение эмбеддингов
        embeddings = inference(
            model, 
            val_dataset, 
            batch_size=config["validation"].get("batch_size", 32), 
            num_workers=config["validation"].get("num_workers", 4), 
            verbose=True
        )
        assert embeddings is not None and len(embeddings) > 0, "Не удалось получить эмбеддинги"
        
        # Построение результатов поиска
        rr = RetrievalResults.from_embeddings(embeddings, val_dataset, n_items=10)
        rr = AdaptiveThresholding(n_std=config["validation"].get("n_std", 2)).process(rr)
        
        # Вычисление стандартных метрик
        retrieval_metrics = calc_retrieval_metrics_rr(rr, map_top_k=(10,), cmc_top_k=(1, 5))
        metrics.update(format_retrieval_metrics(retrieval_metrics))
        
        # Расчет EER
        similarity_scores, labels = collect_similarity_scores(rr, val_dataset)
        eer, threshold = calculate_eer(similarity_scores, labels)
        metrics["eer"] = eer
        metrics["eer_threshold"] = threshold

    # Расчет функции потерь валидации
    if criterion is not None:
        val_loss = calculate_val_loss(model, val_dataset, criterion, device, config)
        metrics["loss"] = val_loss
        logger.info(f"Validation Loss: {val_loss:.4f}")
    
    # Логирование результатов
    for name, value in metrics.items():
        if name != "loss":  # Loss уже залогирован выше
            logger.info(f"{name}: {value:.4f}")
    
    model.train()
    return metrics


def is_better_checkpoint(
    current_metrics: Dict[str, float],
    best_metric: float,
    metric_name: str,
    higher_is_better: bool = False
) -> bool:
    """
    Проверяет, является ли текущий чекпоинт лучшим.
    
    Args:
        current_metrics: Текущие метрики
        best_metric: Лучшее значение метрики
        metric_name: Имя метрики для сравнения
        higher_is_better: True, если большее значение метрики лучше
        
    Returns:
        True, если текущий чекпоинт лучше предыдущего лучшего
    """
    # Проверяем, что метрика присутствует в текущих метриках
    assert metric_name in current_metrics, f"Метрика {metric_name} отсутствует в текущих метриках"
    
    current_value = current_metrics[metric_name]
    
    if best_metric is None:
        return True
    
    if higher_is_better:
        return current_value > best_metric
    else:
        return current_value < best_metric