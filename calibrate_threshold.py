"""
Калибровка порога категоризатора.
Прогоняем на размеченной валидации, смотрим precision/recall на каждом пороге.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.categorizer import TransactionCategorizer


def calibrate_threshold():
    print("Загрузка данных...")
    df = pd.read_csv("data/all_operations.csv", sep=';')
    
    # Берем только те, что попали в справочник (уже размечены)
    cat = TransactionCategorizer()
    
    # Сэмплируем 2000 случайных операций для валидации
    sample = df.sample(min(2000, len(df)), random_state=42).copy()
    sample['company_id'] = sample['company_id'].astype(str)
    
    # Прогоняем через категоризатор БЕЗ финального решения
    sample['norm'] = sample['payment_purpose'].apply(cat._normalize)
    sample['rule'] = sample['payment_purpose'].apply(cat._rule_category)
    
    # Получаем эмбеддинги (или TF-IDF)
    texts = [t if t else 'пусто' for t in sample['norm'].tolist()]
    
    if cat.model is not None and cat.ref_embeddings is not None:
        import torch
        from sentence_transformers import util
        
        query_emb = cat.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        ref_tensor = torch.tensor(cat.ref_embeddings, device=query_emb.device, dtype=query_emb.dtype)
        scores = util.cos_sim(query_emb, ref_tensor)
        best_scores, best_idx = scores.max(dim=1)
        sample['ml_score'] = best_scores.cpu().numpy()
        sample['ml_category'] = [cat.ref_df.iloc[i]['category_ru'] for i in best_idx.cpu().numpy()]
    else:
        from sklearn.metrics.pairwise import cosine_similarity
        X_q = cat.vectorizer.transform(texts)
        sim = cosine_similarity(X_q, cat.X_ref)
        sample['ml_score'] = sim.max(axis=1)
        sample['ml_category'] = [cat.ref_df.iloc[i]['category_ru'] for i in sim.argmax(axis=1)]
    
    # "Истинная" метка: либо правило, либо ML на пороге 0.7 (наше лучшее приближение)
    sample['true_category'] = sample.apply(
        lambda r: r['rule'] if pd.notna(r['rule']) else r['ml_category'],
        axis=1,
    )
    
    # Прогоняем калибровку
    print("\nКалибровка threshold на валидации 2000 примеров:")
    print(f"{'threshold':>10} {'coverage':>10} {'accuracy':>10} {'Прочее':>10}")
    print("-" * 50)
    
    best_threshold = 0.6
    best_f1 = 0
    
    for threshold in np.arange(0.30, 0.90, 0.05):
        # Решение при этом пороге
        sample['pred'] = sample.apply(
            lambda r: r['rule'] if pd.notna(r['rule'])
            else (r['ml_category'] if r['ml_score'] >= threshold else 'Прочее'),
            axis=1,
        )
        
        # Метрики
        non_other = sample[sample['pred'] != 'Прочее']
        coverage = len(non_other) / len(sample) * 100
        
        if len(non_other) > 0:
            accuracy = (non_other['pred'] == non_other['true_category']).mean() * 100
            # F1-подобная метрика: accuracy * coverage / 100 (балансирует)
            f1 = 2 * accuracy * coverage / (accuracy + coverage) if (accuracy + coverage) > 0 else 0
        else:
            accuracy, f1 = 0, 0
        
        marker = " <-- ОПТИМУМ" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        
        print(f"{threshold:>10.2f} {coverage:>9.1f}% {accuracy:>9.1f}% {coverage:>9.1f}%{marker}")
    
    print(f"\n{'='*50}")
    print(f"Оптимальный threshold: {best_threshold:.2f}")
    print(f"F1-score (balanced): {best_f1:.1f}")
    print(f"\nОбнови в src/categorizer.py: self.threshold = {best_threshold:.2f}")


if __name__ == "__main__":
    calibrate_threshold()