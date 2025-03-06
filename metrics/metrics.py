"""
Модуль с метриками для оценки качества модели.
"""
import numpy as np
from typing import Dict, Tuple, List
from sklearn.metrics import roc_curve


def calculate_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Вычисляет Equal Error Rate (EER).
    
    Args:
        scores: Массив с предсказанными вероятностями или скорами
        labels: Массив с реальными метками (0 или 1)
        
    Returns:
        Кортеж (EER, порог)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # Находим точку, где FAR (fpr) = FRR (fnr)
    idx = np.nanargmin(np.absolute(fpr - fnr))
    eer_threshold = thresholds[idx]
    eer = fpr[idx]
    
    return eer, eer_threshold


def collect_similarity_scores(
    retrieval_results, 
    dataset
) -> Tuple[List[float], List[int]]:
    """
    Собирает оценки сходства и метки для всех пар запрос-галерея.
    
    Args:
        retrieval_results: Результаты поиска (RetrievalResults)
        dataset: Датасет
        
    Returns:
        Кортеж из списков (similarity_scores, labels)
    """
    all_similarity_scores = []
    all_labels = []
    
    # Получаем все метки из датасета
    all_dataset_labels = dataset.get_labels()
    
    # Получаем ID запросов и галереи
    query_ids = dataset.get_query_ids()
    gallery_ids = dataset.get_gallery_ids()
    
    # Для каждого запроса
    for i, query_id in enumerate(query_ids):
        query_idx = query_id.item()  # Преобразуем тензор в число
        query_label = all_dataset_labels[query_idx]
        
        # Для каждого найденного элемента галереи этого запроса
        for gallery_rel_idx, similarity in zip(
            retrieval_results.retrieved_ids[i], 
            retrieval_results.distances[i]
        ):
            # Получаем абсолютный индекс элемента галереи
            gallery_abs_idx = gallery_ids[gallery_rel_idx.item()].item()
            
            # Пропускаем сравнение с самим собой
            if query_idx == gallery_abs_idx:
                continue
                
            gallery_label = all_dataset_labels[gallery_abs_idx]
            
            # Преобразуем расстояние в сходство (косинусное сходство = 1 - расстояние)
            similarity_value = 1.0 - similarity.item()
            
            # Если метки совпадают - это положительная пара (1), иначе негативная (0)
            label = 1 if query_label == gallery_label else 0
            
            all_similarity_scores.append(similarity_value)
            all_labels.append(label)
    
    return all_similarity_scores, all_labels


def format_retrieval_metrics(results: Dict) -> Dict[str, float]:
    """
    Форматирует метрики из результатов retrieval.
    
    Args:
        results: Словарь с результатами из calc_retrieval_metrics_rr
        
    Returns:
        Словарь с форматированными метриками
    """
    metrics = {}
    
    for metric_name in results.keys():
        for k, v in results[metric_name].items():
            metric_key = f"{metric_name}@{k}"
            metrics[metric_key] = v.item()
    
    return metrics