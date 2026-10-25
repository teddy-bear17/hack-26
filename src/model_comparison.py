"""Сравнение ML-моделей прогнозирования."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go


def compare_models(df, test_size=0.2):
    if df.empty or len(df) < 30:
        return None
    df = df.copy()
    df['operation_date'] = pd.to_datetime(df['operation_date'])
    daily = df.groupby('operation_date')['balance_eod'].last().reset_index().set_index('operation_date')
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq='D')
    daily = daily.reindex(idx).ffill().fillna(0)
    daily['day_of_week'] = daily.index.dayofweek
    daily['day_of_month'] = daily.index.day
    daily['lag_1'] = daily['balance_eod'].shift(1).fillna(daily['balance_eod'].mean())

    features = ['day_of_week', 'day_of_month', 'lag_1']
    X = daily[features].values
    y = daily['balance_eod'].values

    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1000))) * 100
        err_pct = np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1000))
        acc_10 = float(np.mean(err_pct < 0.10) * 100)
        results.append({'model': name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'Accuracy@10%': acc_10})
    return results


def plot_comparison(results):
    df = pd.DataFrame(results)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='MAE (руб.)', x=df['model'], y=df['MAE'], marker_color='#0d6efd'))
    fig.update_layout(title="Сравнение моделей по MAE", xaxis_title="Модель",
                      yaxis_title="MAE", height=350, margin=dict(l=10, r=10, t=40, b=10))
    return fig