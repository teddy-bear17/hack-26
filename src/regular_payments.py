"""Обнаружение регулярных платежей в истории операций."""
import pandas as pd
from datetime import timedelta


def detect_regular_payments(df_ops, min_occurrences=3, min_interval_days=20, max_interval_days=40):
    if df_ops.empty:
        return []
    df = df_ops.copy()
    df['operation_date'] = pd.to_datetime(df['operation_date'])
    expenses = df[df['direction'] == 'Списание'].copy()
    expenses['purpose_key'] = (expenses['payment_purpose'].astype(str).str.lower()
                               .str.replace(r'[^\w\s]', ' ', regex=True).str[:40])

    regulars = []
    for (purpose_key, counterparty), group in expenses.groupby(['purpose_key', 'counterparty_name']):
        if len(group) < min_occurrences:
            continue
        group = group.sort_values('operation_date')
        dates = group['operation_date'].tolist()
        intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates) - 1)]
        valid = [i for i in intervals if min_interval_days <= i <= max_interval_days]
        if len(valid) < len(intervals) * 0.6:
            continue
        avg_interval = sum(valid) / len(valid) if valid else 30
        last_date = dates[-1]
        next_date = last_date + timedelta(days=int(avg_interval))
        regulars.append({
            "purpose": str(group['payment_purpose'].iloc[-1])[:80],
            "counterparty": counterparty,
            "avg_amount": float(group['amount'].mean()),
            "interval_days": int(avg_interval),
            "occurrences": len(group),
            "last_date": last_date.strftime('%Y-%m-%d'),
            "next_date": next_date.strftime('%Y-%m-%d'),
        })
    regulars.sort(key=lambda x: x['avg_amount'], reverse=True)
    return regulars


def generate_regular_payments_events(regulars, start_date, days_ahead=60):
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date).date()
    end = start_date + timedelta(days=days_ahead)
    events = []
    for reg in regulars:
        last = pd.to_datetime(reg['last_date']).date()
        interval = reg['interval_days']
        next_payment = last + timedelta(days=interval)
        while next_payment <= end:
            if next_payment >= start_date:
                events.append({
                    "date": next_payment.strftime('%Y-%m-%d'),
                    "amount": reg['avg_amount'],
                    "direction": "Списание",
                    "purpose": f"{reg['purpose']} (регулярный)",
                    "category_ru": "Регулярный платёж",
                    "tenant_id": None,
                })
            next_payment += timedelta(days=interval)
    return events