import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.graph_objects as go
from sklearn.model_selection import learning_curve
import os

def train_global_model(all_ops_path="sample_data/all_operations.csv"):
    if not os.path.exists(all_ops_path): return None
    try:
        df = pd.read_csv(all_ops_path, sep=';', dtype={'company_inn': str})
        df['operation_date'] = pd.to_datetime(df['operation_date'])
        daily = df.groupby('operation_date')['balance_eod'].mean().reset_index().set_index('operation_date')
        full_idx = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq='D')
        daily = daily.reindex(full_idx).ffill().fillna(0)
        
        daily['day_of_week'] = daily.index.dayofweek
        daily['day_of_month'] = daily.index.day
        daily['is_weekend'] = daily.index.dayofweek.isin([5, 6]).astype(int)
        daily['lag_1'] = daily['balance_eod'].shift(1).fillna(daily['balance_eod'].mean())
        daily['lag_7'] = daily['balance_eod'].shift(7).fillna(daily['balance_eod'].mean())
        
        X = daily[['day_of_week', 'day_of_month', 'is_weekend', 'lag_1', 'lag_7']]
        y = daily['balance_eod']
        
        model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
        model.fit(X, y)
        return model
    except Exception as e:
        print(f"Global model error: {e}")
        return None

def predict_balance_ml(df, days_ahead=60, global_model=None):
    logs = ["1. Загрузка и валидация исторических данных..."]
    if df.empty: return [], [], {}, logs, [], [], []
        
    df['operation_date'] = pd.to_datetime(df['operation_date'])
    daily = df.groupby('operation_date')['balance_eod'].last().reset_index().set_index('operation_date')
    full_idx = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq='D')
    daily = daily.reindex(full_idx).ffill().fillna(0)
    logs.append(f"2. Ряд восстановлен. Дней: {len(daily)}")
    
    daily['day_of_week'] = daily.index.dayofweek
    daily['day_of_month'] = daily.index.day
    daily['is_weekend'] = daily.index.dayofweek.isin([5, 6]).astype(int)
    daily['lag_1'] = daily['balance_eod'].shift(1).fillna(daily['balance_eod'].mean())
    daily['lag_7'] = daily['balance_eod'].shift(7).fillna(daily['balance_eod'].mean())
    daily['rolling_mean_7'] = daily['balance_eod'].rolling(window=7, min_periods=1).mean()
    logs.append("3. Feature Engineering завершен (добавлены лаги и скользящие средние).")
    
    split_idx = int(len(daily) * 0.8)
    if split_idx < 5: split_idx = len(daily) - 5 
    
    train_df, test_df = daily.iloc[:split_idx], daily.iloc[split_idx:]
    features = ['day_of_week', 'day_of_month', 'is_weekend', 'lag_1', 'lag_7', 'rolling_mean_7']
    
    X_train = train_df[features]
    y_train = train_df['balance_eod']
    
    logs.append(f"4. Обучение Local RandomForest на {len(train_df)} днях (Train).")
    local_model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
    local_model.fit(X_train, y_train)
    
    # --- БЭКТЕСТИНГ ---
    logs.append(f"5. Бэктестинг на {len(test_df)} днях (Test).")
    test_balance_true = test_df['balance_eod'].values
    test_balance_pred = []
    
    for _, row in test_df.iterrows():
        f_X = pd.DataFrame([{
            'day_of_week': row['day_of_week'], 
            'day_of_month': row['day_of_month'], 
            'is_weekend': row['is_weekend'],
            'lag_1': row['lag_1'],
            'lag_7': row['lag_7'],
            'rolling_mean_7': row['rolling_mean_7']
        }])
        
        pred_loc = local_model.predict(f_X)[0]
        pred_glob = global_model.predict(f_X[['day_of_week', 'day_of_month', 'is_weekend', 'lag_1', 'lag_7']])[0] if global_model else pred_loc
        
        if global_model:
            weight_local = min(0.8, len(train_df) / 75.0) 
            final_pred = (pred_loc * weight_local) + (pred_glob * (1 - weight_local))
        else:
            final_pred = pred_loc
            
        test_balance_pred.append(final_pred)

    test_balance_pred = np.array(test_balance_pred)
    
    # --- МЕТРИКИ ---
    mae = mean_absolute_error(test_balance_true, test_balance_pred)
    mse = mean_squared_error(test_balance_true, test_balance_pred)
    rmse = np.sqrt(mse)
    
    # Считаем MAPE с защитой от нуля
    mape = np.mean(np.abs((test_balance_true - test_balance_pred) / np.maximum(np.abs(test_balance_true), 1000))) * 100
    mape = min(mape, 100.0)
    
    # Считаем Accuracy как % дней, где ошибка меньше 10%
    error_pct = np.abs((test_balance_true - test_balance_pred) / np.maximum(np.abs(test_balance_true), 1000))
    accuracy_10pct = np.mean(error_pct < 0.10) * 100
    
    y_true_bin = (test_balance_true < 0).astype(int)
    y_pred_bin = (test_balance_pred < 0).astype(int)
    
    if sum(y_true_bin) == 0:
        acc, prec, rec, f1 = 1.0, 0.0, 0.0, 0.0
        cm = [[len(y_true_bin), 0], [0, 0]]
    else:
        acc = accuracy_score(y_true_bin, y_pred_bin)
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        cm = confusion_matrix(y_true_bin, y_pred_bin)
    
    metrics = {
        "MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape, "Accuracy_10pct": accuracy_10pct,
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "Confusion_Matrix": cm
    }
    
    # --- БУДУЩИЙ ПРОГНОЗ ---
    last_real_balance = df.sort_values('operation_date')['balance_eod'].iloc[-1]
    last_date = daily.index.max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
    future_forecast = []
    
    current_lag_1 = last_real_balance
    current_lag_7 = daily['balance_eod'].iloc[-7] if len(daily) >= 7 else last_real_balance
    current_rolling = daily['rolling_mean_7'].iloc[-1]
    
    for i, f_date in enumerate(future_dates):
        f_X = pd.DataFrame([{
            'day_of_week': f_date.dayofweek, 
            'day_of_month': f_date.day, 
            'is_weekend': int(f_date.weekday() in [5, 6]),
            'lag_1': current_lag_1,
            'lag_7': current_lag_7,
            'rolling_mean_7': current_rolling
        }])
        
        pred_loc = local_model.predict(f_X)[0]
        pred_glob = global_model.predict(f_X[['day_of_week', 'day_of_month', 'is_weekend', 'lag_1', 'lag_7']])[0] if global_model else pred_loc
        
        if global_model:
            weight_local = min(0.8, len(daily) / 75.0) 
            final_pred = (pred_loc * weight_local) + (pred_glob * (1 - weight_local))
        else:
            final_pred = pred_loc
            
        future_forecast.append(final_pred)
        current_lag_7 = current_lag_1
        current_lag_1 = final_pred
        current_rolling = (current_rolling * 6 + final_pred) / 7

    logs.append("6. Будущий прогноз сформирован.")
    test_dates = [d.strftime('%Y-%m-%d') for d in test_df.index]
    
    return [d.date() for d in future_dates], future_forecast, metrics, logs, test_dates, test_balance_true.tolist(), test_balance_pred.tolist()

def generate_training_plots(model, X, y):
    """Генерирует графики Feature Importance и Learning Curve."""
    plots = {}
    
    # 1. Важность признаков
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        # Названия фичей, которые мы подаем в модель (их теперь 6)
        features = [
            'День недели', 'День месяца', 'Выходной', 
            'Вчерашний баланс', 'Баланс 7 дней назад', 'Скользящее среднее'
        ]
        
        fig_fi = go.Figure(go.Bar(
            x=importances, y=features, orientation='h', marker_color='#0d6efd'
        ))
        fig_fi.update_layout(
            title="Важность признаков (Feature Importance)", 
            height=300, margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Вклад в прогноз"
        )
        plots['feature_importance'] = fig_fi

    # 2. Кривая обучения
    if len(X) > 20: 
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=3, scoring='neg_mean_absolute_error',
                train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
            )
            train_errors = -np.mean(train_scores, axis=1)
            test_errors = -np.mean(test_scores, axis=1)
            
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(x=train_sizes, y=train_errors, name='Ошибка на Train', mode='lines+markers', line=dict(color='#0d6efd')))
            fig_lc.add_trace(go.Scatter(x=train_sizes, y=test_errors, name='Ошибка на Test (Валидация)', mode='lines+markers', line=dict(color='#fd7e14')))
            
            fig_lc.update_layout(
                title="Кривая обучения (Learning Curve)", 
                height=300, margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Количество дней (Train set)", yaxis_title="MAE (Ошибка)"
            )
            plots['learning_curve'] = fig_lc
        except Exception as e:
            print(f"Learning curve error: {e}")

    return plots