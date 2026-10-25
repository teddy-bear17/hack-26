"""
Скрипт обучения категоризатора.
Запускать ОДИН раз после изменений в данных.
Создает:
  - data/ai_generated_categories.csv
  - data/category_embeddings.npy
  - data/category_texts_for_embeddings.csv
  - data/clustering_metrics.json
"""

import os
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("[train] sentence-transformers не установлен. Используем TF-IDF.")


STOP_WORDS = {
    'оплата', 'счет', 'счету', 'договор', 'договору', 'номер', 'без', 'ндс',
    'руб', 'за', 'по', 'на', 'от', 'до', 'для', 'в', 'и', 'с', 'то', 'не',
    'услуг', 'товар', 'продукц', 'облагается', 'числе', 'сумма', 'перечисление',
    'права', 'предоставление', 'поставки', 'получения', 'согласно', 'том',
    'шт', 'кг', 'весь', 'данный', 'его', 'она', 'они', 'также', 'либо',
}

GARBAGE_PATTERN = r'^AI:\s*(Группа\s*\d+|\(\d+%\)\s*\(\d+%\)|\(\d+%\)|\d+(\s|$))'


def find_data_file():
    paths = [
        "data/all_operations.csv",
        "data/C001.csv",
        "sample_data/C001.csv",
        "sample_data/all_operations.csv",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def extract_category_name(texts):
    words = []
    for text in texts:
        clean = re.sub(r'[^\w\s]', ' ', str(text).lower())
        for word in clean.split():
            if len(word) > 3 and not word.isdigit() and word not in STOP_WORDS:
                words.append(word)

    if not words:
        return None

    top_words = [w[0] for w in Counter(words).most_common(2)]
    return f"AI: {' '.join(top_words).capitalize()}"


def find_optimal_k(embeddings, k_min=20, k_max=None, sample_size=1500):
    """
    Авто-подбор K через Silhouette Score.
    Семплирует точки для скорости. Косинусная метрика — стандарт для текста.
    """
    n_samples = len(embeddings)
    if k_max is None:
        k_max = min(150, n_samples // 5)
    if k_max <= k_min:
        return k_min, []

    # Семплируем для скорости
    if n_samples > sample_size:
        np.random.seed(42)
        idx = np.random.choice(n_samples, sample_size, replace=False)
        emb_sample = embeddings[idx]
    else:
        emb_sample = embeddings

    print(f"[train] Авто-подбор K в диапазоне [{k_min}, {k_max}]...")

    best_k, best_score = k_min, -1
    results = []

    # Шаг: перебираем 10-12 значений
    step = max(5, (k_max - k_min) // 10)
    for k in range(k_min, k_max + 1, step):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=200)
            labels = km.fit_predict(emb_sample)

            if len(set(labels)) < 2:
                continue

            score = silhouette_score(
                emb_sample, labels,
                metric='cosine',
                sample_size=min(800, len(emb_sample)),
            )
            results.append({"k": k, "score": float(score)})
            print(f"[train]   K={k}: Silhouette={score:.3f}")

            if score > best_score:
                best_score = score
                best_k = k
        except Exception as e:
            print(f"[train]   K={k}: ошибка {e}")
            continue

    print(f"[train] Оптимальное K = {best_k} (Silhouette={best_score:.3f})")
    return best_k, results


def train_new_categories():
    path = find_data_file()
    if not path:
        print("[train] ОШИБКА: файл с операциями не найден.")
        return

    print(f"[train] Загрузка: {path}")
    df = pd.read_csv(path, sep=';')

    purposes = (df[df['direction'] == 'Списание']['payment_purpose']
                .dropna().unique())
    print(f"[train] Уникальных назначений: {len(purposes)}")

    if len(purposes) < 30:
        print("[train] Слишком мало данных (минимум 30).")
        return

    print("[train] Векторизация...")
    if HAS_EMBEDDINGS:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        embeddings = model.encode(purposes.tolist(), show_progress_bar=True)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
        embeddings = vec.fit_transform(purposes).toarray()

    # --- АВТО-ПОДБОР K ---
    n_clusters, silhouette_results = find_optimal_k(embeddings, k_min=20)

    print(f"[train] Финальная кластеризация K={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

        # --- ГЕНЕРАЦИЯ КАТЕГОРИЙ ---
    print("[train] Генерация названий и отбор эталонов...")
    results = []
    for cid in range(n_clusters):
        idx = np.where(cluster_labels == cid)[0]
        if len(idx) == 0:
            continue

        # Получаем все тексты этого кластера
        cluster_texts = [purposes[i] for i in idx]
        
        # Генерируем название по всем текстам кластера
        category_name = extract_category_name(cluster_texts)

        if not category_name or re.match(GARBAGE_PATTERN, category_name):
            print(f"[train] Кластер {cid}: мусор, пропуск")
            continue

        # --- САМОЕ ВАЖНОЕ: БЕРЕМ ТОЛЬКО ТОП-20 САМЫХ ЧАСТЫХ ФРАЗ ---
        text_counts = Counter(cluster_texts)
        top_texts = [t for t, _ in text_counts.most_common(20)]
        
        print(f"[train] Кластер {cid}: {len(idx)} операций -> {category_name} (взято {len(top_texts)} эталонов)")

        for t in top_texts:
            results.append({
                "cluster": cid,
                "representative_purpose": t,
                "category_ru": category_name,
                "category_group": "AI_Generated",
            })

    if not results:
        print("[train] Все кластеры мусорные.")
        return

    ref_df = pd.DataFrame(results)
    ref_df = ref_df.drop_duplicates(subset=['representative_purpose'], keep='first').reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    ref_df.to_csv("data/ai_generated_categories.csv", sep=';', index=False)
    print(f"[train] Сохранено категорий: {len(ref_df)}")

    # --- СОХРАНЕНИЕ МЕТРИК КЛАСТЕРИЗАЦИИ ---
    with open("data/clustering_metrics.json", "w", encoding='utf-8') as f:
        json.dump({
            "optimal_k": n_clusters,
            "silhouette_scores": silhouette_results,
            "n_purposes": len(purposes),
            "n_categories": int(ref_df['category_ru'].nunique()),
            "n_records": len(ref_df),
        }, f, indent=2, ensure_ascii=False)

    # --- СОХРАНЕНИЕ ВЕКТОРОВ ---
    if not HAS_EMBEDDINGS:
        print("[train] Без sentence-transformers векторы не сохранены.")
        return

    print("[train] Кодирование финальных эталонов...")
    final_texts = ref_df['representative_purpose'].tolist()
    final_embeddings = model.encode(final_texts, show_progress_bar=True)

    assert len(final_texts) == len(final_embeddings)

    np.save("data/category_embeddings.npy", final_embeddings)
    pd.DataFrame({
        'representative_purpose': final_texts,
        'category_ru': ref_df['category_ru'].tolist(),
    }).to_csv("data/category_texts_for_embeddings.csv", sep=';', index=False)

    print(f"[train] Векторы сохранены: {final_embeddings.shape}")
    print("[train] Готово.")


if __name__ == "__main__":
    train_new_categories()