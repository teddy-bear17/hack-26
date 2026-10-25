import pandas as pd
from datetime import timedelta
from src.mock_integrations import FNSRegistry

def find_cash_gaps(forecast_dates, forecast_balances, ml_lower_bounds=None):
    if not forecast_dates: return {"has_gap": False}
    start_date = pd.to_datetime(forecast_dates[0]).date()
    for i, balance in enumerate(forecast_balances):
        is_gap = balance < 0
        if ml_lower_bounds is not None and len(ml_lower_bounds) > i:
            if ml_lower_bounds[i] < 0: is_gap = True 
        if is_gap:
            gap_date = pd.to_datetime(forecast_dates[i]).date()
            days_to_gap = (gap_date - start_date).days
            probability = max(50, 95 - int(days_to_gap * 1.5))
            if balance < 0: probability = min(99, probability + 15)
            return {
                "has_gap": True, "date": gap_date,
                "deficit": abs(balance if balance < 0 else ml_lower_bounds[i]),
                "probability": probability, "is_tax_day": gap_date.day == 28, "index": i
            }
    return {"has_gap": False}

def apply_scenario(base_forecast, delay_days=0, delay_impact=0, extra_expense=0, preset=None):
    if preset == "client_lost": delay_days, delay_impact, extra_expense = 0, 0, 500000
    scenario_forecast = []
    for i, val in enumerate(base_forecast):
        new_val = val
        if i >= delay_days:
            new_val -= delay_impact
            new_val -= extra_expense
        scenario_forecast.append(new_val)
    return scenario_forecast

def inject_expected_events(base_forecast, forecast_dates, events_list, tenant_id=None):
    if not events_list: return base_forecast
    adjusted_forecast = list(base_forecast)
    for event in events_list:
        if tenant_id and event.get('tenant_id') and str(event.get('tenant_id')) != str(tenant_id): continue
        date_val = event.get('date') or event.get('expected_date')
        direction_val = event.get('direction') or event.get('type')
        amount = event.get('amount', 0)
        if not date_val or not direction_val: continue
        event_date = pd.to_datetime(date_val).date()
        is_income = direction_val.lower() == 'поступление'
        amount_signed = amount if is_income else -amount
        for i, f_date in enumerate(forecast_dates):
            if pd.to_datetime(f_date).date() >= event_date:
                for j in range(i, len(adjusted_forecast)):
                    adjusted_forecast[j] += amount_signed
                break
    return adjusted_forecast

def check_counterparty_risk(inn_list):
    if not inn_list: return {}
    response = FNSRegistry.check_counterparties(inn_list)
    return response.get("data", {}) if response.get("status") == "success" else {}

def aggregate_ui_data(df):
    if df.empty: return {"top_counterparties": pd.DataFrame(), "category_distribution": pd.DataFrame()}
    df_expenses = df[df['direction'] == 'Списание'].copy()
    
    if 'counterparty_name' in df_expenses.columns:
        top_cp = df_expenses.groupby('counterparty_name')['amount'].sum().reset_index().sort_values('amount', ascending=False).head(5)
    else:
        top_cp = pd.DataFrame(columns=['counterparty_name', 'amount'])
        
    group_col = 'category_ru' if 'category_ru' in df_expenses.columns else 'payment_purpose'
    if group_col in df_expenses.columns:
        cat_dist = df_expenses.groupby(group_col)['amount'].sum().reset_index().sort_values('amount', ascending=False).head(7)
    else:
        cat_dist = pd.DataFrame(columns=[group_col, 'amount'])
    return {"top_counterparties": top_cp, "category_distribution": cat_dist}

def explain_cash_gap(forecast_operations_df, gap_date):
    if forecast_operations_df is None or forecast_operations_df.empty: return []
    df_safe = forecast_operations_df.copy()
    if 'category_ru' not in df_safe.columns: df_safe['category_ru'] = 'Прочее'
    df_safe['operation_date'] = pd.to_datetime(df_safe['operation_date']).dt.date
    df_safe['day_of_month'] = pd.to_datetime(df_safe['operation_date']).dt.day
    
    target_day = gap_date.day
    mask = (df_safe['day_of_month'] == target_day) & (df_safe['direction'] == 'Списание')
    past_expenses = df_safe[mask].copy()
    
    if past_expenses.empty:
        last_30_days = gap_date - timedelta(days=30)
        mask_30 = (df_safe['operation_date'] >= last_30_days) & (df_safe['direction'] == 'Списание')
        past_expenses = df_safe[mask_30].copy()
        
    if past_expenses.empty: return []
    top_reasons = past_expenses.groupby('category_ru')['amount'].mean().reset_index().sort_values(by='amount', ascending=False).head(3)
    return top_reasons.to_dict('records')

def get_dynamic_gap_reasons(gap_date, gap_deficit, df_ops, manual_events, tenant_id):
    """
    Математический движок атрибуции (Explainable AI).
    Вычисляет точный вклад каждой категории и ручного события в кассовый разрыв.
    """
    reasons = []
    
    # 1. Считаем вклад ручных событий (то, что ввел юзер)
    for ev in manual_events:
        if str(ev.get('tenant_id')) != str(tenant_id):
            continue
        ev_date = pd.to_datetime(ev["date"]).date()
        # Если событие произошло в день разрыва или за 2 дня до него
        if 0 <= (gap_date - ev_date).days <= 2 and ev.get("direction") == "Списание":
            reasons.append({
                "source": "Ручное событие",
                "name": ev.get("purpose", "Без названия"),
                "impact": ev.get("amount", 0)
            })
            
    # 2. Считаем вклад исторических категорий (из выписки)
    if not df_ops.empty:
        df_safe = df_ops.copy()
        df_safe['operation_date'] = pd.to_datetime(df_safe['operation_date']).dt.date
        
        # Берем операции за 3 дня до разрыва
        start_date = gap_date - timedelta(days=3)
        mask = (df_safe['operation_date'] >= start_date) & (df_safe['operation_date'] <= gap_date) & (df_safe['direction'] == 'Списание')
        recent_expenses = df_safe[mask]
        
        if not recent_expenses.empty:
            # Группируем по категориям и суммируем
            cat_col = 'category_ru' if 'category_ru' in recent_expenses.columns else 'payment_purpose'
            cat_grouped = recent_expenses.groupby(cat_col)['amount'].sum().reset_index()
            
            for _, row in cat_grouped.iterrows():
                reasons.append({
                    "source": "Историческая категория",
                    "name": row[cat_col],
                    "impact": row['amount']
                })
                
    # 3. Сортируем по величине влияния (от большего к меньшему)
    reasons = sorted(reasons, key=lambda x: x['impact'], reverse=True)
    
    # 4. Формируем умный вывод
    if not reasons:
        return ["Недостаточно данных для точной атрибуции причин."]
        
    total_impact = sum(r['impact'] for r in reasons)
    explanation = []
    
    for r in reasons[:3]: # Берем топ-3 причины
        share = (r['impact'] / total_impact) * 100 if total_impact > 0 else 0
        explanation.append(f"**{r['name']}** ({r['impact']:,.0f} руб. — {share:.1f}% от всех списаний)")
        
    return explanation