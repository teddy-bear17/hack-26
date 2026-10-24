import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Импорты наших модулей. Пока закомментил, чтобы не падало до утра
# from src.data_loader import load_operations
# from src.risk_engine import find_cash_gaps
# from src.forecaster import predict_balance 

# Настройка страницы. Название потом поменяем на нормальное
st.set_page_config(page_title="Финансовый дашборд", layout="wide")

st.title("Поиск кассовых разрывов")

with st.sidebar:
    st.header("Параметры")
    company = st.selectbox("Компания / ИП", ["C001 - ООО «Грузоперевозки-063»", "C002 - ИП Предприниматель-156"])
    st.text("ИНН: 0098524470\nФорма: ООО")
    
    st.divider()
    
    # Загрузка файла. Чтобы не искать файлы вручную, если ничего не загружено - берем тестовый
    uploaded_file = st.file_uploader("Загрузите выписку (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=';', dtype={'company_inn': str, 'counterparty_inn': str})
            st.success("Файл загружен")
        except Exception as e:
            st.error(f"Ошибка парсинга: {e}")
    else:
        # Дефолтный тестовый файл, чтобы приложение не падало при первом запуске
        try:
            df = pd.read_csv("sample_data/C001.csv", sep=';', dtype={'company_inn': str, 'counterparty_inn': str})
            st.info("Загружен тестовый C001.csv")
        except FileNotFoundError:
            st.warning("Файл не найден. Загрузите CSV вручную.")
            df = pd.DataFrame() # Пустая заготовка, чтобы графики не падали

    st.divider()
    st.subheader("Сценарий 'Что если'")
    delay_days = st.slider("Задержка оплаты (дней)", 0, 30, 0)
    extra_expense = st.number_input("Внеплановый расход (руб.)", min_value=0, value=0, step=50000)

# Вкладки. Первая наша, остальные - ночная наработка команды
tab_forecast, tab_analytics, tab_categories, tab_operations = st.tabs([
    "Прогноз и риски", 
    "Аналитика по контрагентам", 
    "Категоризация", 
    "Реестр операций"
])

with tab_forecast:
    st.header("Прогноз остатка на счете")
    
    # ВРЕМЕННАЯ ЗАГЛУШКА: Фейковые данные, чтобы проверить верстку. 
    # Утром заменить это на вызов predict_balance()
    dates = [datetime(2026, 9, 1) + timedelta(days=i) for i in range(60)]
    history = [1500000 + i*5000 for i in range(30)]
    forecast = [1500000 + i*5000 - (i**2)*2000 for i in range(30, 60)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates[:30], y=history, name="История", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=dates[30:], y=forecast, name="Прогноз", line=dict(color="#ff7f0e", dash="dash")))
    
    # Красная линия на нуле. Ниже - зона риска
    fig.add_hline(y=0, line_dash="dot", line_color="red")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Заглушка алерта. Потом прикрутим вывод списка из find_cash_gaps()
    st.subheader("Предупреждения")
    st.error("15.11.2026: Ожидается кассовый разрыв. Не хватит 150 000 руб.")

with tab_analytics:
    st.header("Аналитика по контрагентам")
    
    # TODO: Заменить на реальный df, когда бэкендерша доделает агрегацию
    labels = ['ООО "Альфа Поставка"', 'ПАО "Аренда Центр"', 'ИП Сидоров В.М.', 'ООО "ЭнергоСбыт"']
    values = [450000, 180000, 120000, 65000]
    
    fig_pie = px.pie(names=labels, values=values, hole=0.5)
    st.plotly_chart(fig_pie, use_container_width=True)
    
with tab_categories:
    st.header("Категоризация транзакций")
    
    # TODO: Заменить на реальный df с колонкой category
    df_cat = pd.DataFrame({
        'category': ['Аренда', 'Закупка сырья', 'Коммунальные услуги', 'Логистика'],
        'amount': [180000, 450000, 65000, 120000]
    })
    
    fig_bar = px.bar(df_cat, x='category', y='amount')
    st.plotly_chart(fig_bar, use_container_width=True)
    
with tab_operations:
    st.header("Выписка операций")
    
    # TODO: Заменить на реальный df
    df_ops = pd.DataFrame({
        'date': ['2026-10-24', '2026-10-25', '2026-10-26'],
        'income': [76422, 35795, 20860],
        'outcome': [63555, 32159, 50920],
        'balance_eod': [200000, 203636, 173576]
    })
    
    st.dataframe(df_ops, use_container_width=True)