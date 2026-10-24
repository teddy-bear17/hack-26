import pandas as pd
from datetime import timedelta

def find_cash_gaps(forecast_dates, forecast_balances):
    """
    Движок поиска кассовых разрывов.
    Анализирует прогноз и возвращает первый день ухода в минус.
    Включает учет налогового календаря (28-е число).
    """
    for i, balance in enumerate(forecast_balances):
        if balance < 0:
            gap_date = forecast_dates[i]
            return {
                "has_gap": True,
                "date": gap_date,
                "deficit": abs(balance),
                "is_tax_day": gap_date.day == 28,
                "index": i
            }
    return {"has_gap": False}


def apply_scenario(base_forecast, delay_days=0, delay_impact=0, extra_expense=0, preset=None):
    """
    Движок "Что если". Пересчитывает массив прогноза.
    Поддерживает как ручной ввод с ползунков, так и "Умные пресеты".
    """
    if preset == "client_lost":
        delay_days, delay_impact, extra_expense = 0, 0, 500000  # Падение выручки
    elif preset == "rent_increase":
        delay_days, delay_impact, extra_expense = 5, 0, 150000  # Рост расходов
    elif preset == "fine":
        delay_days, delay_impact, extra_expense = 2, 0, 300000  # Внезапный штраф

    scenario_forecast = []
    for i, val in enumerate(base_forecast):
        new_val = val
        if i >= delay_days:
            new_val -= delay_impact
            new_val -= extra_expense
        scenario_forecast.append(new_val)
        
    return scenario_forecast


def calculate_health_score(current_balance, upcoming_mandatory_expenses):
    """
    Расчет "Финансового Health Score" (Оценка здоровья бизнеса).
    Коэффициент покрытия: Баланс / Ожидаемые траты.
    """
    if upcoming_mandatory_expenses <= 0:
        return "green"
        
    ratio = current_balance / upcoming_mandatory_expenses
    
    if ratio < 1.2:
        return "red"
    elif ratio < 1.5:
        return "yellow"
    else:
        return "green"


def explain_cash_gap(forecast_operations_df, gap_date):
    """
    Объяснимый ИИ (Explainable AI).
    Ищет топ-3 самых крупных списаний за 3 дня до кассового разрыва.
    Адаптировано под датасет: использует operation_date и direction=='Списание'.
    """
    if forecast_operations_df is None or forecast_operations_df.empty:
        return []
        
    start_date = gap_date - timedelta(days=3)
    
    # 1. Защита от пустых значений в текстах назначений или контрагентах
    df_safe = forecast_operations_df.copy()
    df_safe['counterparty_name'] = df_safe['counterparty_name'].fillna('Неизвестный контрагент')
    df_safe['payment_purpose'] = df_safe['payment_purpose'].fillna('Без назначения')
    
    # 2. Приводим даты к единому типу date, чтобы сравнение не падало
    df_safe['operation_date_dt'] = pd.to_datetime(df_safe['operation_date']).dt.date
    
    # 3. Фильтрация по окну в 3 дня и только списания
    mask = (
        (df_safe['operation_date_dt'] >= start_date) & 
        (df_safe['operation_date_dt'] <= gap_date) & 
        (df_safe['direction'] == 'Списание')
    )
    
    # 4. Сортировка по убыванию суммы
    top_expenses = df_safe[mask].sort_values(by='amount', ascending=False).head(3)
    
    return top_expenses[['operation_date', 'amount', 'counterparty_name', 'payment_purpose']].to_dict('records')