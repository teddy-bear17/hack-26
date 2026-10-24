#алерты и "Что если"

def find_cash_gaps(forecast_df):
    #ищет дни, где баланс уходит в минус и возвращает список алертов"

    gaps=forecast_df[forecast_df['balance']<0]
    alerts=[]
    for _, row in gaps.iterrows():
        alerts.append({
            "date": row['date'],
            "ammount": row['balamce'],
            "reason": "Прогнозируемый дифицит"
        })
    return alerts