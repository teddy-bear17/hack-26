"""
Агрегация ML-метрик по всем компаниям.
Считается один раз, кэшируется в JSON.
"""
import os
import json
import pandas as pd
import numpy as np

from src.forecaster import predict_balance_ml, train_global_model
from src.categorizer import TransactionCategorizer
from src.data_loader import load_tenant_data


CACHE_PATH = "data/aggregate_metrics.json"


def compute_all_metrics(companies_df, sample_size=None, force_recompute=False):
    """
    Считает метрики для каждой компании из companies_df.
    Кэширует результат в JSON.
    sample_size — если задан, ограничивает число компаний (для скорости).
    """
    if not force_recompute and os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass

    print("[aggregate] Начинаю расчет метрик по всем компаниям...")
    global_model = train_global_model("sample_data/all_operations.csv")
    categorizer = TransactionCategorizer()

    if sample_size:
        companies_df = companies_df.head(sample_size)

    results = []
    total = len(companies_df)

    for i, (_, comp) in enumerate(companies_df.iterrows()):
        comp_id = comp['id']
        comp_name = comp['name']

        # Ищем CSV
        paths = [f"data/statements/{comp_id}.csv", f"sample_data/{comp_id}.csv"]
        path = next((p for p in paths if os.path.exists(p)), None)

        if not path:
            continue

        try:
            df = load_tenant_data(comp_id, path)
            df['operation_date'] = pd.to_datetime(df['operation_date']).dt.date
            df = categorizer.predict(df)
            df['balance_eod'] = pd.to_numeric(df['balance_eod'], errors='coerce').fillna(0)

            if len(df) < 30:
                continue

            history = df.sort_values('operation_date').copy()

            _, _, metrics, _, _, _, _ = predict_balance_ml(
                history, days_ahead=30, global_model=global_model,
            )

            results.append({
                'company_id': comp_id,
                'company_name': comp_name,
                'MAE': metrics.get('MAE', 0),
                'MAPE': metrics.get('MAPE', 0),
                'RMSE': metrics.get('RMSE', 0),
                'Accuracy_10pct': metrics.get('Accuracy_10pct', 0),
                'days': len(df),
            })
            print(f"[aggregate] {i+1}/{total} {comp_id}: MAE={metrics.get('MAE', 0):,.0f}, Acc@10={metrics.get('Accuracy_10pct', 0):.0f}%")
        except Exception as e:
            print(f"[aggregate] {comp_id}: error {e}")
            continue

    if not results:
        return None

    df_results = pd.DataFrame(results)
    summary = {
        'n_companies': len(df_results),
        'MAE_mean': float(df_results['MAE'].mean()),
        'MAE_median': float(df_results['MAE'].median()),
        'MAE_std': float(df_results['MAE'].std()),
        'MAPE_mean': float(df_results['MAPE'].mean()),
        'MAPE_median': float(df_results['MAPE'].median()),
        'RMSE_mean': float(df_results['RMSE'].mean()),
        'Accuracy_10pct_mean': float(df_results['Accuracy_10pct'].mean()),
        'Accuracy_10pct_median': float(df_results['Accuracy_10pct'].median()),
        'Accuracy_10pct_std': float(df_results['Accuracy_10pct'].std()),
        'Accuracy_10pct_min': float(df_results['Accuracy_10pct'].min()),
        'Accuracy_10pct_max': float(df_results['Accuracy_10pct'].max()),
        'per_company': results,
    }

    os.makedirs("data", exist_ok=True)
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[aggregate] Готово. Обработано {len(df_results)} компаний.")
    return summary