import os
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
import plotly.graph_objects as go
from sklearn.model_selection import learning_curve
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


# ==================== FEATURE ENGINEERING ====================

def _add_features(daily):
    daily = daily.copy()
    daily['day_of_week'] = daily.index.dayofweek
    daily['day_of_month'] = daily.index.day
    daily['is_weekend'] = (daily.index.dayofweek >= 5).astype(int)
    daily['is_month_start'] = (daily.index.day <= 3).astype(int)
    daily['is_month_end'] = (daily.index.day >= 27).astype(int)
    daily['is_tax_day'] = (daily.index.day == 28).astype(int)
    daily['is_salary_day'] = daily.index.day.isin([5, 10, 25]).astype(int)

    for k in [1, 2]:
        daily[f'sin_week_{k}'] = np.sin(2 * np.pi * k * daily['day_of_week'] / 7)
        daily[f'cos_week_{k}'] = np.cos(2 * np.pi * k * daily['day_of_week'] / 7)
        daily[f'sin_month_{k}'] = np.sin(2 * np.pi * k * daily['day_of_month'] / 30)
        daily[f'cos_month_{k}'] = np.cos(2 * np.pi * k * daily['day_of_month'] / 30)

    daily['lag_1'] = daily['balance_eod'].shift(1).fillna(daily['balance_eod'].mean())
    daily['lag_3'] = daily['balance_eod'].shift(3).fillna(daily['balance_eod'].mean())
    daily['lag_7'] = daily['balance_eod'].shift(7).fillna(daily['balance_eod'].mean())
    daily['lag_14'] = daily['balance_eod'].shift(14).fillna(daily['balance_eod'].mean())

    daily['rolling_mean_7'] = daily['balance_eod'].rolling(7, min_periods=1).mean()
    daily['rolling_mean_14'] = daily['balance_eod'].rolling(14, min_periods=1).mean()
    daily['rolling_std_7'] = daily['balance_eod'].rolling(7, min_periods=1).std().fillna(0)

    daily['delta'] = daily['balance_eod'].diff().fillna(0)
    daily['delta_lag_1'] = daily['delta'].shift(1).fillna(0)
    daily['delta_lag_7'] = daily['delta'].shift(7).fillna(0)

    return daily


def _feature_list():
    return [
        'day_of_week', 'day_of_month', 'is_weekend',
        'is_month_start', 'is_month_end',
        'is_tax_day', 'is_salary_day',
        'sin_week_1', 'cos_week_1', 'sin_week_2', 'cos_week_2',
        'sin_month_1', 'cos_month_1', 'sin_month_2', 'cos_month_2',
        'lag_1', 'lag_3', 'lag_7', 'lag_14',
        'rolling_mean_7', 'rolling_mean_14', 'rolling_std_7',
        'delta_lag_1', 'delta_lag_7',
    ]


def _recency_weights(n_samples, decay=0.7):
    return np.exp(np.linspace(np.log(decay), 0, n_samples))


def train_global_model(path="sample_data/all_operations.csv"):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep=';', dtype={'company_inn': str})
        df['operation_date'] = pd.to_datetime(df['operation_date'])
        daily = (df.groupby('operation_date')['balance_eod'].mean().reset_index().set_index('operation_date'))
        idx = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
        daily = daily.reindex(idx).ffill().fillna(0)

        daily = _add_features(daily)
        features = _feature_list()

        X = daily[features].fillna(0)
        y = daily['balance_eod']
        weights = _recency_weights(len(X))

        model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
        model.fit(X, y, sample_weight=weights)
        print(f"[forecaster] Global модель обучена на {len(X)} днях.")
        return model
    except Exception as e:
        print(f"[forecaster] Global error: {e}")
        return None


def _build_ensemble():
    """Собирает ансамбль из доступных моделей."""
    models = {}

    if HAS_XGB:
        models['xgb'] = XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
            random_state=42, n_jobs=-1, verbosity=0,
        )

    if HAS_CATBOOST:
        models['catboost'] = CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=5,
            loss_function='RMSE',
            verbose=False,
            random_seed=42,
        )

    # RandomForest всегда присутствует как базовая стабильная модель
    models['rf'] = RandomForestRegressor(
        n_estimators=100, max_depth=7,
        min_samples_leaf=5, min_samples_split=10,
        max_features='sqrt',
        random_state=42, n_jobs=-1,
    )

    # Если нет ни XGB, ни CatBoost — добавляем GBR для разнообразия
    if not HAS_XGB and not HAS_CATBOOST:
        models['gbr'] = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.08,
            min_samples_leaf=5, subsample=0.85, random_state=42,
        )

    return models


def _compute_weight_local(n_days):
    if n_days < 15: return 0.0
    if n_days < 30: return 0.3
    if n_days < 60: return 0.5
    if n_days < 120: return 0.75
    return 0.9


def _walk_forward_cv(daily, features, n_splits=3):
    n = len(daily)
    if n < 40: return []
    fold_size = n // (n_splits + 1)
    results = []
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end = min(train_end + fold_size, n)
        if train_end < 15 or test_end <= train_end: continue
        train = daily.iloc[:train_end]
        test = daily.iloc[train_end:test_end]
        try:
            model = RandomForestRegressor(
                n_estimators=50, max_depth=7,
                min_samples_leaf=5, max_features='sqrt',
                random_state=42, n_jobs=-1,
            )
            X_train = train[features].fillna(0)
            y_train = train['delta']
            weights = _recency_weights(len(X_train))
            model.fit(X_train, y_train, sample_weight=weights)
            current_balance = train['balance_eod'].iloc[-1]
            preds = []
            for _, row in test.iterrows():
                f_X = pd.DataFrame([row[features]], columns=features).fillna(0)
                delta = model.predict(f_X)[0]
                current_balance += delta
                preds.append(current_balance)
            test_true = test['balance_eod'].values
            mae = mean_absolute_error(test_true, preds)
            mape = np.mean(np.abs((test_true - np.array(preds)) / np.maximum(np.abs(test_true), 1000))) * 100
            results.append({
                'fold': i + 1, 'train_days': len(train), 'test_days': len(test),
                'mae': float(mae), 'mape': float(min(mape, 100.0)),
            })
        except Exception as e:
            print(f"[forecaster] CV fold {i+1} error: {e}")
            continue
    return results


def predict_balance_ml(df, days_ahead=60, global_model=None):
    logs = ["1. Загрузка исторических данных."]

    if df.empty:
        return [], [], {}, logs, [], [], []

    df = df.copy()
    df['operation_date'] = pd.to_datetime(df['operation_date'])
    daily = (df.groupby('operation_date')['balance_eod'].last().reset_index().set_index('operation_date'))
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    daily = daily.reindex(idx).ffill().fillna(0)
    logs.append(f"2. Ряд восстановлен. Дней: {len(daily)}")

    daily = _add_features(daily)
    features = _feature_list()
    logs.append(f"3. Feature Engineering: {len(features)} фичей.")

    logs.append("4. Walk-forward CV (3 фолда)...")
    cv_results = _walk_forward_cv(daily, features, n_splits=3)
    for r in cv_results:
        logs.append(f"   Fold {r['fold']}: MAE={r['mae']:,.0f}, MAPE={r['mape']:.1f}%")

    if cv_results:
        cv_mae_mean = np.mean([r['mae'] for r in cv_results])
        cv_mae_std = np.std([r['mae'] for r in cv_results])
        logs.append(f"   Средний MAE по фолдам: {cv_mae_mean:,.0f} ± {cv_mae_std:,.0f}")
    else:
        cv_mae_mean, cv_mae_std = 0, 0

    split_idx = int(len(daily) * 0.8)
    if split_idx < 15: split_idx = len(daily) - 5

    train_df = daily.iloc[:split_idx].copy()
    test_df = daily.iloc[split_idx:].copy()

    X_train = train_df[features].fillna(0)
    y_train_delta = train_df['delta']
    weights = _recency_weights(len(X_train))

    logs.append(f"5. Обучение ансамбля на {len(train_df)} днях.")
    ensemble = _build_ensemble()
    for name, model in ensemble.items():
        model.fit(X_train, y_train_delta, sample_weight=weights)
        logs.append(f"   - {name} обучен.")

    weight_local = _compute_weight_local(len(train_df))
    logs.append(f"6. Вес Local: {weight_local:.2f}")

    # --- Бэктестинг (от реального вчерашнего баланса) ---
    test_true = test_df['balance_eod'].values
    test_pred = []
    for _, row in test_df.iterrows():
        f_X = pd.DataFrame([row[features]], columns=features).fillna(0)
        deltas = [m.predict(f_X)[0] for m in ensemble.values()]
        pred_delta = np.mean(deltas)
        if global_model is not None:
            pred_global = global_model.predict(f_X)[0]
            global_delta = pred_global - row['lag_1']
            final_delta = pred_delta * weight_local + global_delta * (1 - weight_local)
        else:
            final_delta = pred_delta
        pred_balance = row['lag_1'] + final_delta
        test_pred.append(pred_balance)
    test_pred = np.array(test_pred)

    mae = mean_absolute_error(test_true, test_pred)
    mse = mean_squared_error(test_true, test_pred)
    rmse = np.sqrt(mse)
    mape = min(np.mean(np.abs((test_true - test_pred) / np.maximum(np.abs(test_true), 1000))) * 100, 100.0)
    err_pct = np.abs((test_true - test_pred) / np.maximum(np.abs(test_true), 1000))
    accuracy_10pct = float(np.mean(err_pct < 0.10) * 100)

    y_true_bin = (test_true < 0).astype(int)
    y_pred_bin = (test_pred < 0).astype(int)
    if y_true_bin.sum() == 0:
        acc, prec, rec, f1 = 1.0, 0.0, 0.0, 0.0
        cm = [[len(y_true_bin), 0], [0, 0]]
    else:
        acc = accuracy_score(y_true_bin, y_pred_bin)
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        cm = confusion_matrix(y_true_bin, y_pred_bin).tolist()

    # --- Будущий прогноз ---
    logs.append("7. Формирование будущего прогноза.")
    last_date = daily.index.max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
    future = []
    cur_balance = df.sort_values('operation_date')['balance_eod'].iloc[-1]
    cur_row = daily.iloc[-1].copy()

    for i, f_date in enumerate(future_dates):
        row = cur_row.copy()
        row['day_of_week'] = f_date.dayofweek
        row['day_of_month'] = f_date.day
        row['is_weekend'] = int(f_date.weekday() >= 5)
        row['is_month_start'] = int(f_date.day <= 3)
        row['is_month_end'] = int(f_date.day >= 27)
        row['is_tax_day'] = int(f_date.day == 28)
        row['is_salary_day'] = int(f_date.day in [5, 10, 25])
        for k in [1, 2]:
            row[f'sin_week_{k}'] = np.sin(2 * np.pi * k * f_date.dayofweek / 7)
            row[f'cos_week_{k}'] = np.cos(2 * np.pi * k * f_date.dayofweek / 7)
            row[f'sin_month_{k}'] = np.sin(2 * np.pi * k * f_date.day / 30)
            row[f'cos_month_{k}'] = np.cos(2 * np.pi * k * f_date.day / 30)
        row['lag_1'] = cur_balance
        row['lag_3'] = cur_row.get('lag_1', cur_balance)
        row['lag_7'] = cur_row.get('lag_3', cur_balance)
        row['lag_14'] = cur_row.get('lag_7', cur_balance)
        row['rolling_mean_7'] = (cur_row['rolling_mean_7'] * 6 + cur_balance) / 7
        row['rolling_mean_14'] = (cur_row['rolling_mean_14'] * 13 + cur_balance) / 14

        X = pd.DataFrame([row[features]], columns=features).fillna(0)
        deltas = [m.predict(X)[0] for m in ensemble.values()]
        pred_delta = np.mean(deltas)
        if global_model is not None:
            pred_global = global_model.predict(X)[0]
            global_delta = pred_global - row['lag_1']
            final_delta = pred_delta * weight_local + global_delta * (1 - weight_local)
        else:
            final_delta = pred_delta
        new_balance = cur_balance + final_delta
        future.append(new_balance)
        row['delta_lag_7'] = row['delta_lag_1']
        row['delta_lag_1'] = final_delta
        cur_row = row
        cur_balance = new_balance

    # --- ВЕРОЯТНОСТНЫЙ ПРОГНОЗ (отключен, интервал строится от RMSE в app.py) ---
    logs.append("8. Вероятностный прогноз: отключен.")
    probabilistic = None

    test_dates = [d.strftime('%Y-%m-%d') for d in test_df.index]

    metrics = {
        "MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse),
        "MAPE": float(mape), "Accuracy_10pct": accuracy_10pct,
        "Accuracy": float(acc), "Precision": float(prec),
        "Recall": float(rec), "F1": float(f1), "Confusion_Matrix": cm,
        "cv_results": cv_results,
        "cv_mae_mean": float(cv_mae_mean),
        "cv_mae_std": float(cv_mae_std),
        "probabilistic": probabilistic,
    }

    return (
        [d.date() for d in future_dates],
        future,
        metrics,
        logs,
        test_dates,
        test_true.tolist(),
        test_pred.tolist(),
    )


def calculate_gap_probability(forecast, metrics):
    if metrics.get("probabilistic"):
        return metrics["probabilistic"]["prob_gap"]
    hist_std = metrics.get("MAE", 100000)
    probs = []
    uncertainty = hist_std if hist_std > 0 else 50000
    for val in forecast:
        uncertainty = np.sqrt(uncertainty ** 2 + (hist_std or 50000) ** 2) * 0.98
        if uncertainty > 0:
            z = val / uncertainty
        else:
            z = float('inf') if val > 0 else float('-inf') if val < 0 else 0
        from math import erf, sqrt
        prob_negative = 0.5 * (1 + erf(-z / sqrt(2)))
        probs.append(float(prob_negative))
    return probs


def generate_training_plots(model, X, y):
    plots = {}
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = _feature_list()
        n = min(len(importances), len(feature_names))
        fig = go.Figure(go.Bar(x=importances[:n], y=feature_names[:n], orientation='h', marker_color='#0d6efd'))
        fig.update_layout(title="Важность признаков", height=500, margin=dict(l=10, r=10, t=40, b=10))
        plots['feature_importance'] = fig
    if len(X) > 20:
        try:
            sizes, tr_scores, te_scores = learning_curve(model, X, y, cv=3, scoring='neg_mean_absolute_error', train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sizes, y=-tr_scores.mean(axis=1), name='Train MAE', mode='lines+markers', line=dict(color='#0d6efd')))
            fig.add_trace(go.Scatter(x=sizes, y=-te_scores.mean(axis=1), name='Test MAE', mode='lines+markers', line=dict(color='#fd7e14')))
            fig.update_layout(title="Кривая обучения", height=300, margin=dict(l=10, r=10, t=40, b=10))
            plots['learning_curve'] = fig
        except Exception:
            pass
    return plots