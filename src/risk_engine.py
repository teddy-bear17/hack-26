import pandas as pd
from datetime import timedelta
import calendar
from src.mock_integrations import FNSRegistry

# --- 1. ВЕРОЯТНОСТЬ КАССОВОГО РАЗРЫВА ---
def find_cash_gaps(forecast_dates, forecast_balances, ml_lower_bounds=None):
    """
    Движок поиска кассовых разрывов.
    Возвращает дату, сумму дефицита и вероятность разрыва (в %).
    """
    start_date = pd.to_datetime(forecast_dates[0]).date()
    
    for i, balance in enumerate(forecast_balances):
        is_gap = balance < 0
        if ml_lower_bounds is not None and len(ml_lower_bounds) > i:
            if ml_lower_bounds[i] < 0:
                is_gap = True 
                
        if is_gap:
            gap_date = pd.to_datetime(forecast_dates[i]).date()
            days_to_gap = (gap_date - start_date).days
            probability = max(50, 95 - int(days_to_gap * 1.5))
            
            if balance < 0:
                probability = min(99, probability + 15)
            
            return {
                "has_gap": True,
                "date": gap_date,
                "deficit": abs(balance if balance < 0 else ml_lower_bounds[i]),
                "probability": probability,
                "is_tax_day": gap_date.day == 28,
                "index": i
            }
            
    return {"has_gap": False}

# --- 2. НЕЗАВИСИМЫЙ ДВИЖОК "ЧТО ЕСЛИ" ---
def apply_scenario(base_forecast, delay_days=0, delay_impact=0, extra_expense=0, preset=None):
    """
    Пересчитывает прогноз на основе ручных ползунков или готовых пресетов.
    """
    if preset == "client_lost":
        delay_days, delay_impact, extra_expense = 0, 0, 500000
    elif preset == "rent_increase":
        delay_days, delay_impact, extra_expense = 5, 0, 150000
    elif preset == "fine":
        delay_days, delay_impact, extra_expense = 2, 0, 300000

    scenario_forecast = []
    for i, val in enumerate(base_forecast):
        new_val = val
        if i >= delay_days:
            new_val -= delay_impact
            new_val -= extra_expense
        scenario_forecast.append(new_val)
        
    return scenario_forecast

# --- 3. РУЧНОЙ ВВОД СОБЫТИЙ ---
def inject_expected_events(base_forecast, forecast_dates, events_list, tenant_id=None):
    """
    Добавляет ручные события и данные из 1С/Банка в базовый прогноз.
    Отфильтровывает события, не относящиеся к указанному tenant_id.
    """
    if not events_list:
        return base_forecast
        
    adjusted_forecast = list(base_forecast)
    
    for event in events_list:
        # Multi-tenancy check: пропускаем события чужих бизнесов
        if tenant_id and event.get('tenant_id') and str(event.get('tenant_id')) != str(tenant_id):
            continue
            
        date_val = event.get('date') or event.get('expected_date')
        direction_val = event.get('direction') or event.get('type')
        amount = event.get('amount', 0)
        
        if not date_val or not direction_val:
            continue
            
        event_date = pd.to_datetime(date_val).date()
        is_income = direction_val.lower() == 'поступление'
        amount_signed = amount if is_income else -amount
        
        for i, f_date in enumerate(forecast_dates):
            current_date = pd.to_datetime(f_date).date()
            if current_date >= event_date:
                # Начиная со дня события, баланс изменяется на эту сумму до конца периода
                for j in range(i, len(adjusted_forecast)):
                    adjusted_forecast[j] += amount_signed
                break
                
    return adjusted_forecast

# --- 4. МОК-ОЦЕНКА РИСКА КОНТРАГЕНТОВ ---
def check_counterparty_risk(inn_list):
    """
    Проверяет ИНН контрагентов через мок ФНС.
    """
    if not inn_list:
        return {}
    response = FNSRegistry.check_counterparties(inn_list)
    if response.get("status") == "success":
        return response.get("data", {})
    return {}

# --- 5. ГЕНЕРАТОР НАЛОГОВОГО КАЛЕНДАРЯ ---
def generate_tax_calendar(start_date, days_ahead=30, estimated_tax_amount=150000):
    """
    Генерирует налоговые события (ЕНП до 28-го числа) для интеграции в inject_expected_events.
    """
    tax_events = []
    current_date = start_date
    end_date = start_date + timedelta(days=days_ahead)
    
    while current_date <= end_date:
        if current_date.day == 28:
            tax_events.append({
                "date": current_date.strftime('%Y-%m-%d'),
                "amount": estimated_tax_amount,
                "direction": "Списание",
                "purpose": "Единый налоговый платеж (ЕНП)"
            })
        current_date += timedelta(days=1)
        
    return tax_events

# --- 6. ВАЛИДАТОР "ГРЯЗНЫХ" ДАННЫХ ---
def validate_data_quality(df):
    """
    Проверяет загруженный CSV на аномалии и пропуски (Data Quality Check).
    """
    issues = []
    
    # Проверка обязательных колонок
    required_cols = {'operation_date', 'amount', 'direction'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        issues.append(f"Отсутствуют обязательные колонки: {', '.join(missing_cols)}")
        return {"is_valid": False, "issues": issues}

    # Проверка пропусков в суммах и датах
    if df['amount'].isnull().any():
        issues.append(f"Найдено {df['amount'].isnull().sum()} пустых значений в суммах")
    if df['operation_date'].isnull().any():
        issues.append("Присутствуют пустые даты операций")
        
    # Проверка отрицательных сумм (в банковской выписке суммы обычно по модулю)
    if (df['amount'] < 0).any():
        issues.append("Найдены отрицательные значения в колонке amount")

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "total_rows": len(df)
    }

# --- 7. АГРЕГАЦИЯ ДАННЫХ ДЛЯ ВКЛАДОК UI ---
def aggregate_ui_data(df):
    """
    Агрегирует данные для вкладок 'Аналитика по контрагентам' и 'Категоризация' в app.py.
    """
    if df.empty:
        return {"top_counterparties": pd.DataFrame(), "category_distribution": pd.DataFrame()}
        
    df_expenses = df[df['direction'] == 'Списание'].copy()
    
    # Топ-5 контрагентов по расходам
    if 'counterparty_name' in df_expenses.columns:
        top_cp = df_expenses.groupby('counterparty_name')['amount'].sum().reset_index()
        top_cp = top_cp.sort_values('amount', ascending=False).head(5)
    else:
        top_cp = pd.DataFrame(columns=['counterparty_name', 'amount'])
        
    # Распределение по категориям (если Никита из ML уже добавил колонку category, иначе группируем по назначению)
    group_col = 'category' if 'category' in df_expenses.columns else 'payment_purpose'
    
    if group_col in df_expenses.columns:
        cat_dist = df_expenses.groupby(group_col)['amount'].sum().reset_index()
        cat_dist = cat_dist.sort_values('amount', ascending=False).head(7)
    else:
        cat_dist = pd.DataFrame(columns=[group_col, 'amount'])

    return {
        "top_counterparties": top_cp,
        "category_distribution": cat_dist
    }

# --- ДОП. ФУНКЦИИ ---
def calculate_health_score(current_balance, upcoming_mandatory_expenses):
    if upcoming_mandatory_expenses <= 0:
        return "green"
    ratio = current_balance / upcoming_mandatory_expenses
    if ratio < 1.2: return "red"
    elif ratio < 1.5: return "yellow"
    else: return "green"

def explain_cash_gap(forecast_operations_df, gap_date):
    if forecast_operations_df is None or forecast_operations_df.empty: return []
    start_date = gap_date - timedelta(days=3)
    df_safe = forecast_operations_df.copy()
    df_safe['counterparty_name'] = df_safe['counterparty_name'].fillna('Неизвестный контрагент')
    df_safe['payment_purpose'] = df_safe['payment_purpose'].fillna('Без назначения')
    df_safe['operation_date_dt'] = pd.to_datetime(df_safe['operation_date']).dt.date
    mask = ((df_safe['operation_date_dt'] >= start_date) & (df_safe['operation_date_dt'] <= gap_date) & (df_safe['direction'] == 'Списание'))
    top_expenses = df_safe[mask].sort_values(by='amount', ascending=False).head(3)
    return top_expenses[['operation_date', 'amount', 'counterparty_inn', 'counterparty_name', 'payment_purpose']].to_dict('records')