#парсинг CSV, фикс ИНН(баг с отсутвием 2х нулей в начлае)

import pandas as pd 

def load_operations(filepath):
    #загружает CSV и фиксит этот самый баг
    df=pd.read_csv(filepath, sep=';', dtype={'company_inn':str, 'counterparty_inn':str}) #здесь инн читается как строка
    df['operation_date']=pd.to_datetime(df['operation_daate']).dt.date #убирает время из даты
    return df