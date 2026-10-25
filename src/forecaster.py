import os
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
import plotly.graph_objects as go
from sklearn.model_selection import learning_curve


def train_global_model(path="sample_data/all_operations.csv"):
    if not os.path.exists(path):
        print(f"[forecaster] Global model: {path} не найден.")
        return None
    try:
        df = pd.read_csv(path, sep=';', dtype={'company_inn': str})
        df['operation_date'] = pd.to_datetime(df['operation_date'])

        daily = (df.groupby('operation_date')['balance_eod']
                 .mean().reset_index().set_index('operation_date'))
        idx = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
        daily = daily.reindex(idx).ffill().fillna(0)

        daily['day_of_week'] = daily.index.dayofweek
        daily['day_of_month'] = daily.index.day
        daily['is_weekend'] = daily.index.dayofweek.isin([5, 6]).astype(int)
        daily['lag_1'] = daily['balance_eod'].shift(1).fillna(daily['balance_eod'].mean())
        daily['lag_7'] = daily['balance_eod'].shift(7).fillna(daily['balance_eod'].mean())

        X = daily[['day_of_week', 'day_of_month', 'is_weekend', 'lag_1', 'lag_7']]
        y = daily['balance_eod']

        model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
        model.fit(X, y)
        print(f"[forecaster] Global модель обучена на {len(X)} днях.")
        return model
    except Exception as e:
        print(f"[forecaster] Global error: {e}")
        return None


def _compute_weight_local(n_days):
    """
    Ступенчатый вес: 
    - Мало данных -> доверяем глобальной модели.
    - Много данных -> 100% локальная модель (чтобы не тянуло к среднему по рынку).
    """
    if n_days < 15:
        return 0.0
    if n_days < 30:
        return 0.5
    if n_days < 60:
        return 0.8
    return 1.0 # <--- ГЛАВНЫЙ ФИКС: Если >60 дней, Global не используется


def predict_balance_ml(df, days_ahead=60, global_model=None):
    logs = ["1. Загрузка исторических данных."]

    if df.empty:
        return [], [], {}, logs, [], [], []

    df = df.copy()
    df['operation_date'] = pd.to_datetime(df['operation_date'])
    daily = (df.groupby('operation_date')['balance_eod']
             .last().reset_index().set_index('operation_date'))
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    daily = daily.reindex(idx).ffill().fillna(0)
    logs.append(f"2. Ряд восстановлен. Дней: {len(daily)}")

    daily['day_of_week'] = daily.index.dayofweek
    daily['day_of_month'] = daily.index.day
    daily['is_weekend'] = daily.index.dayofweek.isin([5, 6]).astype(int)
    daily['lag_1'] = daily['balance_eod'].shift(1).fillna(daily['balance_eod'].mean())
    daily['lag_7'] = daily['balance_eod'].shift(7).fillna(daily['balance_eod'].mean())
    daily['rolling_mean_7'] = daily['balance_eod'].rolling(7, min_periods=1).mean()
    logs.append("3. Feature Engineering: лаги, скользящие, флаги выходных.")

    features = ['day_of_week', 'day_of_month', 'is_weekend', 'lag_1', 'lag_7', 'rolling_mean_7']

    split_idx = int(len(daily) * 0.8)
    if split_idx < 10:
        split_idx = len(daily) - 5

    train_df = daily.iloc[:split_idx]
    test_df = daily.iloc[split_idx:]

    X_train = train_df[features]
    y_train = train_df['balance_eod']

    logs.append(f"4. Local RandomForest на {len(train_df)} днях.")
    local_model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
    local_model.fit(X_train, y_train)

    # --- Бэктестинг ---
    logs.append(f"5. Бэктестинг на {len(test_df)} днях.")
    test_true = test_df['balance_eod'].values
    test_pred = []
    weight_local = _compute_weight_local(len(train_df))
    logs.append(f"   Вес Local: {weight_local:.2f}, вес Global: {1 - weight_local:.2f}")

    for _, row in test_df.iterrows():
        f_X = pd.DataFrame([{
            'day_of_week': row['day_of_week'],
            'day_of_month': row['day_of_month'],
            'is_weekend': row['is_weekend'],
            'lag_1': row['lag_1'],
            'lag_7': row['lag_7'],
            'rolling_mean_7': row['rolling_mean_7'],
        }])

        pred_local = local_model.predict(f_X)[0]
        if global_model is not None:
            pred_global = global_model.predict(
                f_X[['day_of_week', 'day_of_month', 'is_weekend', 'lag_1', 'lag_7']]
            )[0]
            final_pred = pred_local * weight_local + pred_global * (1 - weight_local)
        else:
            final_pred = pred_local

        test_pred.append(final_pred)

    test_pred = np.array(test_pred)

    # --- Метрики ---
    mae = mean_absolute_error(test_true, test_pred)
    mse = mean_squared_error(test_true, test_pred)
    rmse = np.sqrt(mse)

    mape = np.mean(
        np.abs((test_true - test_pred) / np.maximum(np.abs(test_true), 1000))
    ) * 100
    mape = min(mape, 100.0)

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

    metrics = {
        "MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse),
        "MAPE": float(mape), "Accuracy_10pct": accuracy_10pct,
        "Accuracy": float(acc), "Precision": float(prec),
        "Recall": float(rec), "F1": float(f1), "Confusion_Matrix": cm,
    }

    # --- Будущий прогноз ---
    logs.append("6. Формирование будущего прогноза.")

    last_balance = df.sort_values('operation_date')['balance_eod'].iloc[-1]
    last_date = daily.index.max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
    future = []

    cur_lag_1 = last_balance
    cur_lag_7 = daily['balance_eod'].iloc[-7] if len(daily) >= 7 else last_balance
    cur_rolling = daily['rolling_mean_7'].iloc[-1]

    for f_date in future_dates:
        f_X = pd.DataFrame([{
            'day_of_week': f_date.dayofweek,
            'day_of_month': f_date.day,
            'is_weekend': int(f_date.weekday() in [5, 6]),
            'lag_1': cur_lag_1,
            'lag_7': cur_lag_7,
            'rolling_mean_7': cur_rolling,
        }])

        pred_local = local_model.predict(f_X)[0]
        if global_model is not None:
            pred_global = global_model.predict(
                f_X[['day_of_week', 'day_of_month', 'is_weekend', 'lag_1', 'lag_7']]
            )[0]
            final_pred = pred_local * weight_local + pred_global * (1 - weight_local)
        else:
            final_pred = pred_local

        future.append(final_pred)

        cur_lag_7 = cur_lag_1
        cur_lag_1 = final_pred
        cur_rolling = (cur_rolling * 6 + final_pred) / 7

    test_dates = [d.strftime('%Y-%m-%d') for d in test_df.index]

    return (
        [d.date() for d in future_dates],
        future,
        metrics,
        logs,
        test_dates,
        test_true.tolist(),
        test_pred.tolist(),
    )


def generate_training_plots(model, X, y):
    plots = {}

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        features = [
            'День недели', 'День месяца', 'Выходной',
            'Баланс вчера', 'Баланс 7 дней назад', 'Скользящее среднее',
        ]
        fig = go.Figure(go.Bar(x=importances, y=features, orientation='h', marker_color='#0d6efd'))
        fig.update_layout(title="Важность признаков", height=300, margin=dict(l=10, r=10, t=40, b=10))
        plots['feature_importance'] = fig

    if len(X) > 20:
        try:
            sizes, tr_scores, te_scores = learning_curve(
                model, X, y, cv=3, scoring='neg_mean_absolute_error',
                train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1,
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sizes, y=-tr_scores.mean(axis=1), name='Train MAE',
                                     mode='lines+markers', line=dict(color='#0d6efd')))
            fig.add_trace(go.Scatter(x=sizes, y=-te_scores.mean(axis=1), name='Test MAE',
                                     mode='lines+markers', line=dict(color='#fd7e14')))
            fig.update_layout(title="Кривая обучения", height=300,
                              margin=dict(l=10, r=10, t=40, b=10),
                              xaxis_title="Дней в Train", yaxis_title="MAE")
            plots['learning_curve'] = fig
        except Exception as e:
            print(f"[forecaster] LC error: {e}")

    return plots