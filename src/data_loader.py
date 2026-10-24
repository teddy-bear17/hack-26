import pandas as pd
import os

def load_tenant_data(company_id: str, file_path: str) -> pd.DataFrame:
    """
    Загружает данные конкретного тенанта (компании), обеспечивая строгую изоляцию.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл для тенанта {company_id} не найден: {file_path}")
    
    # ИНН читаем как строки, чтобы не потерять ведущие нули
    df = pd.read_csv(file_path, sep=';', dtype={'company_inn': str, 'counterparty_inn': str})
    
    # Строгая изоляция: оставляем только данные запрошенного тенанта
    if 'company_id' in df.columns:
        df = df[df['company_id'] == company_id].copy()
        
    return df