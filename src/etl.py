import pandas as pd

def normalize_bank_statement(df_raw: pd.DataFrame, bank_name: str) -> pd.DataFrame:
    """
    Приводит выгрузку из разных банков к единому внутреннему стандарту.
    """
    # 1. Словари маппинга
    schemas = {
        "Наш стандарт": {
            "operation_date": "operation_date", "amount": "amount",
            "direction": "direction", "balance_eod": "balance_eod",
            "payment_purpose": "payment_purpose", "counterparty_name": "counterparty_name"
        },
        "Тинькофф": {
            "Дата операции": "operation_date", "Сумма операции": "amount",
            "Тип операции": "direction", "Остаток": "balance_eod",
            "Описание": "payment_purpose", "Контрагент": "counterparty_name"
        },
        "Сбер": {
            "date": "operation_date", "sum": "amount",
            "type": "direction", "balance": "balance_eod",
            "purpose": "payment_purpose", "counterparty": "counterparty_name"
        },
        "Альфа": {
            "Дата": "operation_date", "Сумма": "amount",
            "Движение": "direction", "Исходящий остаток": "balance_eod",
            "Назначение платежа": "payment_purpose", "Получатель/Плательщик": "counterparty_name"
        }
    }
    
    # 2. Выбираем схему
    mapping = schemas.get(bank_name, schemas["Наш стандарт"])
    df_norm = df_raw.rename(columns=mapping).copy()
    
    # 3. Проверка обязательных колонок
    required_cols = ["operation_date", "amount", "direction"]
    missing = [col for col in required_cols if col not in df_norm.columns]
    if missing:
        raise ValueError(f"В файле банка {bank_name} отсутствуют обязательные колонки: {missing}")
        
    # 4. Приведение типов
    try:
        df_norm["operation_date"] = pd.to_datetime(df_norm["operation_date"]).dt.date
        df_norm["amount"] = pd.to_numeric(df_norm["amount"], errors='coerce').fillna(0)
    except Exception as e:
        raise ValueError(f"Ошибка приведения типов данных: {e}")
        
    # 5. Унификация "Поступление/Списание"
    df_norm["direction"] = df_norm["direction"].replace({
        "Дебет": "Поступление", "Кредит": "Списание",
        "Incoming": "Поступление", "Outgoing": "Списание",
        "Приход": "Поступление", "Расход": "Списание"
    })
    
    return df_norm