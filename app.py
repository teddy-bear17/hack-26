import streamlit as st 
from src.data_loader  import load_operations
from src.risk_engine import find_cash_gaps
#from src.forecaster import predict_balance !!!НЕ ЗАБЫТЬ ПОТОМ РАССКОМЕНТИРОВАТЬ!!!

st.set_page_config(page_title="Финансовая штука", layout="wide") #название потом сменить

st.title("Поиск кассовых разрывов") #и здесь тоже

tab_forecast, tab_analytics, tab_categories, tab_operations=st.tabs([
    "Прогноз и риски",
    "Аналитика по контрагентам",
    "Категоризация",
    "Реестр операций"
])

with tab_forecast:
    st.header("Прогноз остатка на счете")
    st.info("Туте будет график и ползунки 'Что если'")
    # TODO: Вставить код графика

with tab_analytics:
    st.header("Аналитика по контрагентам")
    st.info("Тут будет код команды (Пироговая диаграмма)")
    
with tab_categories:
    st.header("Категоризация транзакций")
    st.info("Тут будет гистограмма расходов")
    
with tab_operations:
    st.header("Выписка операций")
    st.info("Тут будет таблица датафрейма (st.dataframe)")