"""
Модуль с метриками для оценки качества модели.
"""
import numpy as np
from typing import Dict, Tuple, List
from sklearn.metrics import roc_curve


### Работало до ArcFace
# def calculate_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
#     """
#     Вычисляет Equal Error Rate (EER).
    
#     Args:
#         scores: Массив с предсказанными вероятностями или скорами
#         labels: Массив с реальными метками (0 или 1)
        
#     Returns:
#         Кортеж (EER, порог)
#     """
#     fpr, tpr, thresholds = roc_curve(labels, scores)
#     fnr = 1 - tpr
    
#     # Находим точку, где FAR (fpr) = FRR (fnr)
#     idx = np.nanargmin(np.absolute(fpr - fnr))
#     eer_threshold = thresholds[idx]
#     eer = fpr[idx]
    
#     return eer, eer_threshold

def calculate_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Вычисляет Equal Error Rate (EER) и порог."""
    # Проверки входных данных
    assert len(scores) == len(labels) > 0, "Массивы скоров и меток должны быть непустыми и одинаковой длины"
    
    n_positive = np.sum(labels)
    n_negative = len(labels) - n_positive
    
    assert n_positive > 0 and n_negative > 0, "Для расчета EER необходимы как положительные, так и отрицательные примеры"
    
    # Расчет EER
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    
    # Находим точку пересечения кривых FAR и FRR
    abs_diff = np.absolute(fpr - fnr)
    assert not np.all(np.isnan(abs_diff)), "Не удалось вычислить разность между FAR и FRR (все значения NaN)"
    
    idx = np.nanargmin(abs_diff)
    threshold = thresholds[idx]
    # Усредненное значение для более точной оценки
    eer = (fpr[idx] + fnr[idx]) / 2.0
    
    return eer, threshold


### Работало до ArcFace
# def collect_similarity_scores(
#     retrieval_results, 
#     dataset
# ) -> Tuple[List[float], List[int]]:
#     """
#     Собирает оценки сходства и метки для всех пар запрос-галерея.
    
#     Args:
#         retrieval_results: Результаты поиска (RetrievalResults)
#         dataset: Датасет
        
#     Returns:
#         Кортеж из списков (similarity_scores, labels)
#     """
#     all_similarity_scores = []
#     all_labels = []
    
#     # Получаем все метки из датасета
#     all_dataset_labels = dataset.get_labels()
    
#     # Получаем ID запросов и галереи
#     query_ids = dataset.get_query_ids()
#     gallery_ids = dataset.get_gallery_ids()
    
#     # Для каждого запроса
#     for i, query_id in enumerate(query_ids):
#         query_idx = query_id.item()  # Преобразуем тензор в число
#         query_label = all_dataset_labels[query_idx]
        
#         # Для каждого найденного элемента галереи этого запроса
#         for gallery_rel_idx, similarity in zip(
#             retrieval_results.retrieved_ids[i], 
#             retrieval_results.distances[i]
#         ):
#             # Получаем абсолютный индекс элемента галереи
#             gallery_abs_idx = gallery_ids[gallery_rel_idx.item()].item()
            
#             # Пропускаем сравнение с самим собой
#             if query_idx == gallery_abs_idx:
#                 continue
                
#             gallery_label = all_dataset_labels[gallery_abs_idx]
            
#             # Преобразуем расстояние в сходство (косинусное сходство = 1 - расстояние)
#             similarity_value = 1.0 - similarity.item()
            
#             # Если метки совпадают - это положительная пара (1), иначе негативная (0)
#             label = 1 if query_label == gallery_label else 0
            
#             all_similarity_scores.append(similarity_value)
#             all_labels.append(label)
    
#     return all_similarity_scores, all_labels

def collect_similarity_scores(retrieval_results, dataset) -> Tuple[List[float], List[int]]:
    """Собирает оценки сходства и метки для всех пар запрос-галерея."""
    def get_label_mapping():
        """Создает отображение индексов к оригинальным меткам."""
        has_remapped_labels = hasattr(dataset, 'label_to_index') and dataset.label_to_index is not None
        
        if not has_remapped_labels:
            return {i: label.item() for i, label in enumerate(all_dataset_labels)}
            
        print("Используются оригинальные метки для сравнения в EER")
        
        # Если есть оригинальные метки в датафрейме
        if hasattr(dataset, 'df') and 'original_label' in dataset.df.columns:
            original_labels = dataset.df['original_label'].values
            return {i: original_labels[i] for i in range(len(original_labels))}
        
        # Иначе используем обратный маппинг
        index_to_label = {idx: label for label, idx in dataset.label_to_index.items()}
        return {i: index_to_label.get(label.item(), label.item()) 
                for i, label in enumerate(all_dataset_labels)}
    
    # Получаем все необходимые данные
    all_dataset_labels = dataset.get_labels()
    query_ids = dataset.get_query_ids()
    gallery_ids = dataset.get_gallery_ids()
    label_mapping = get_label_mapping()
    
    all_similarity_scores = []
    all_labels = []
    
    # Собираем пары для сравнения
    for i, query_id in enumerate(query_ids):
        query_idx = query_id.item()
        query_label = label_mapping[query_idx]
        
        for gallery_rel_idx, distance in zip(retrieval_results.retrieved_ids[i], 
                                            retrieval_results.distances[i]):
            gallery_abs_idx = gallery_ids[gallery_rel_idx.item()].item()
            
            # Пропускаем сравнение с самим собой
            if query_idx == gallery_abs_idx:
                continue
                
            gallery_label = label_mapping[gallery_abs_idx]
            similarity_value = 1.0 - distance.item()  # Косинусное сходство = 1 - расстояние
            match_label = 1 if query_label == gallery_label else 0
            
            all_similarity_scores.append(similarity_value)
            all_labels.append(match_label)
    
    # Диагностика и проверки
    n_positive = sum(all_labels)
    n_negative = len(all_labels) - n_positive
    print(f"Собрано {len(all_labels)} пар: {n_positive} положительных, {n_negative} отрицательных.")
    
    assert len(all_similarity_scores) > 0, "Не удалось собрать оценки сходства"
    assert n_positive > 0, "В выборке отсутствуют положительные пары"
    assert n_negative > 0, "В выборке отсутствуют отрицательные пары"
    
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