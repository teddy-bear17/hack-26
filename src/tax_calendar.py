"""Налоговый календарь РФ и сезонные факторы."""
from datetime import date, timedelta
import pandas as pd


def generate_tax_events(start_date, days_ahead=60, monthly_tax_amount=None, tax_regime="УСН 6%"):
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()

    events = []
    current = start_date
    end = start_date + timedelta(days=days_ahead)
    amount = monthly_tax_amount or 150000

    while current <= end:
        if current.day == 28:
            events.append({
                "date": current.strftime('%Y-%m-%d'),
                "amount": amount,
                "direction": "Списание",
                "purpose": "ЕНП (единый налоговый платеж)",
                "category_ru": "Налоги и взносы в бюджет",
                "tenant_id": None,
            })
            if current.month in [1, 4, 7, 10]:
                events.append({
                    "date": current.strftime('%Y-%m-%d'),
                    "amount": amount * 3,
                    "direction": "Списание",
                    "purpose": "УСН (авансовый платеж)",
                    "category_ru": "Налоги и взносы в бюджет",
                    "tenant_id": None,
                })
        current += timedelta(days=1)
    return events


def estimate_monthly_tax(df_ops):
    """Оценивает среднюю налоговую нагрузку. Строгий поиск только по ФНС и ЕНП."""
    if df_ops.empty:
        return 150000
    
    df = df_ops.copy()
    df['operation_date'] = pd.to_datetime(df['operation_date'])
    
    # Ищем только по контрагенту или строгим фразам (убрали 'ндс' и 'усн', так как они есть везде)
    is_fns = df['counterparty_name'].astype(str).str.contains('казначейств|фнс', case=False, na=False)
    is_tax_text = df['payment_purpose'].astype(str).str.contains('единый налоговый|енп|налог на', case=False, na=False)
    
    mask = (is_fns | is_tax_text) & (df['direction'] == 'Списание')
    tax_ops = df[mask]
    
    if tax_ops.empty:
        return 150000
        
    cutoff = df['operation_date'].max() - timedelta(days=180)
    recent = tax_ops[tax_ops['operation_date'] >= cutoff]
    if recent.empty:
        return float(tax_ops['amount'].mean())
        
    n_months = max(1, (recent['operation_date'].max() - recent['operation_date'].min()).days // 30)
    return float(recent['amount'].sum() / n_months)


def apply_seasonality_to_forecast(forecast_dates, forecast, industry):
    if not industry:
        return forecast
    industry = industry.lower()
    adjusted = list(forecast)
    
    for i, d in enumerate(forecast_dates):
        month = d.month
        multiplier = 1.0
        
        if 'розниц' in industry or 'маркетплейс' in industry:
            if month == 12: multiplier = 1.15
            elif month == 1: multiplier = 0.90
        elif 'гостиниц' in industry or 'база отдыха' in industry:
            if month in [6, 7, 8]: multiplier = 1.20
            elif month in [11, 12, 1, 2]: multiplier = 0.80
        elif 'сельское' in industry or 'агро' in industry:
            if month in [8, 9, 10]: multiplier = 1.25
            elif month in [2, 3, 4]: multiplier = 0.85
        elif 'детский' in industry or 'образоват' in industry:
            if month in [6, 7, 8]: multiplier = 0.70
            
        if multiplier != 1.0 and i > 0:
            delta = adjusted[i] - adjusted[i-1]
            adjusted[i] = adjusted[i-1] + (delta * multiplier)
                
    return adjusted