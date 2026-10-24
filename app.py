import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- ИМПОРТЫ НАШИХ МОДУЛЕЙ ---
# Пока закомментированы, чтобы тестовая сборка не падала до интеграции реального бэкенда.
# from src.data_loader import load_operations
# from src.risk_engine import find_cash_gaps
# from src.forecaster import predict_balance 

st.set_page_config(page_title="Финансовый дашборд | Центр-Инвест", layout="wide")

st.title("Поиск кассовых разрывов")
st.markdown("Прототип системы предиктивной аналитики для МСБ.")

# --- СБРОС СОСТОЯНИЯ ПРИ СМЕНЕ КОМПАНИИ ---
# Защита от дурака: чтобы сценарии одной компании не ломали график другой.
if 'current_company' not in st.session_state:
    st.session_state.current_company = None
if 'delay_days' not in st.session_state:
    st.session_state.delay_days = 0
if 'extra_expense' not in st.session_state:
    st.session_state.extra_expense = 0

# --- БОКОВАЯ ПАНЕЛЬ ---
with st.sidebar:
    st.header("Параметры")
    company = st.selectbox(
        "Компания / ИП", 
        ["C001 - ООО «Грузоперевозки-063»", "C002 - ИП Предприниматель-156", "C003 - ООО «Агро-191»"],
        key="selected_company"
    )
    
    # Если компания изменилась — обнуляем ползунки
    if company != st.session_state.current_company:
        st.session_state.delay_days = 0
        st.session_state.extra_expense = 0
        st.session_state.current_company = company

    # Имитация данных из БД
    inn = "0098524470" if "C001" in company else "003844671674" if "C002" in company else "0040581584"
    st.text(f"ИНН: {inn}\nФорма: ООО")
    
    st.divider()

    # Загрузка файла. Если пусто, берем тестовый, чтобы приложение не падало.
    uploaded_file = st.file_uploader("Загрузите выписку (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=';', dtype={'company_inn': str, 'counterparty_inn': str})
            st.success("Файл загружен")
        except Exception as e:
            st.error(f"Ошибка парсинга: {e}")
    else:
        try:
            df = pd.read_csv("sample_data/C001.csv", sep=';', dtype={'company_inn': str, 'counterparty_inn': str})
            st.info("Загружен тестовый C001.csv")
        except FileNotFoundError:
            st.warning("Файл не найден. Загрузите CSV вручную.")
            df = pd.DataFrame()
            
    st.divider()
    st.subheader("Сценарий 'Что если'")
    
    # Ручной ввод, привязанный к сессии
    delay_days = st.number_input("Задержка оплаты (дней)", min_value=0, max_value=365, key="delay_days")
    extra_expense = st.number_input("Внеплановый расход (руб.)", min_value=0, step=50000, key="extra_expense")

# --- ВКЛАДКИ ---
tab_forecast, tab_analytics, tab_categories, tab_operations = st.tabs([
    "Прогноз и риски", 
    "Аналитика по контрагентам", 
    "Категоризация", 
    "Реестр операций"
])

# ВКЛАДКА 1: ПРОГНОЗ И РИСКИ
with tab_forecast:
    st.header("Прогноз остатка на счете")
    
    # Берем системную дату. date для списков, strftime для Plotly.
    today = datetime.now().date()
    
    # Таймлайн: 30 дней назад (история) и 30 дней вперед (прогноз)
    history_dates = [today - timedelta(days=i) for i in range(30, 0, -1)]
    forecast_dates = [today + timedelta(days=i) for i in range(1, 31)]
    
    # Разные базовые цифры для разных компаний
    base_balance = 1500000 if "C001" in company else 800000 if "C002" in company else 400000
    
    # Имитация истории (тренд вверх)
    history = [base_balance + i*1000 for i in range(30)]
    
    # Имитация прогноза (тренд вниз из-за обязательных платежей)
    base_forecast = [base_balance - (i**2)*500 for i in range(1, 31)]
    
    # Влияние сценариев "Что если"
    delay_impact = delay_days * 15000 
    
    forecast = []
    for i, val in enumerate(base_forecast):
        day_val = val
        # Если есть задержка или расход — вычитаем начиная с 10-го дня
        if i >= 10: 
            day_val -= delay_impact
            day_val -= extra_expense
        forecast.append(day_val)

    # Отрисовка графика
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_dates, y=history, name="История", line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, name="Прогноз", line=dict(color="#ff7f0e", dash="dash", width=2)))
    
    # Зона риска
    fig.add_hline(y=0, line_dash="dot", line_color="red")
    fig.add_hrect(y0=-10000000, y1=0, fillcolor="red", opacity=0.1, line_width=0, layer="below")
    
    # Вертикальная линия "Сегодня". Передаем строку во избежание краша Plotly.
    fig.add_vline(
        x=today.strftime('%Y-%m-%d'), 
        line_dash="solid", 
        line_color="gray", 
        line_width=2,
        annotation_text="Сегодня", 
        annotation_position="top left"
    )
    
    fig.update_yaxes(range=[-3000000, 2000000]) 
    st.plotly_chart(fig, use_container_width=True)
    
    # --- ИЗЮМИНКА: АЛЕРТЫ С ОБЪЯСНЕНИЕМ ПРИЧИН ---
    st.subheader("Предупреждения")
    
    if any(f < 0 for f in forecast):
        first_negative_idx = next((i for i, v in enumerate(forecast) if v < 0), None)
        
        if first_negative_idx is not None:
            date_str = forecast_dates[first_negative_idx].strftime("%d.%m.%Y")
            amount_str = f"{abs(forecast[first_negative_idx]):,.0f}".replace(',', ' ')
            
            st.error(f"**{date_str}: Ожидается кассовый разрыв. Не хватит {amount_str} ₽**")
            
            # Раскладываем на колонки: Почему и Что делать (Требование кейса)
            col_cause, col_action = st.columns(2)
            
            with col_cause:
                st.markdown("**Почему это произойдет:**")
                st.markdown("- 28-е число: Уплата ЕНП (налог)\n- 10-е число: Аванс по зарплате\n- Сдвиг поступлений от клиентов")
                
            with col_action:
                st.markdown("**Рекомендуемые действия:**")
                st.markdown("1. Запросить предоплату у крупных контрагентов\n2. Связаться с банком для оформления овердрафта\n3. Отложить некритичные закупки")
    else:
        st.success("Кассовых разрывов в прогнозируемом периоде не выявлено.")

# ВКЛАДКА 2: АНАЛИТИКА ПО КОНТРАГЕНТАМ
with tab_analytics:
    st.header("Аналитика по контрагентам")
    
    # TODO: Заменить на реальный df
    labels = ['ООО "Альфа Поставка"', 'ПАО "Аренда Центр"', 'ИП Сидоров В.М.', 'ООО "ЭнергоСбыт"']
    values = [450000, 180000, 120000, 65000]
    
    fig_pie = px.pie(names=labels, values=values, hole=0.5)
    st.plotly_chart(fig_pie, use_container_width=True)

# ВКЛАДКА 3: КАТЕГОРИЗАЦИЯ
with tab_categories:
    st.header("Категоризация транзакций")
    
    # TODO: Заменить на реальный df
    df_cat = pd.DataFrame({
        'category': ['Аренда', 'Закупка сырья', 'Коммунальные услуги', 'Логистика'],
        'amount': [180000, 450000, 65000, 120000]
    })
    
    fig_bar = px.bar(df_cat, x='category', y='amount')
    st.plotly_chart(fig_bar, use_container_width=True)

# ВКЛАДКА 4: РЕЕСТР ОПЕРАЦИЙ
with tab_operations:
    st.header("Выписка операций")
    
    # Фейковые данные
    df_ops_display = pd.DataFrame({
        'Дата': [today.strftime('%d.%m.%Y'), (today - timedelta(days=1)).strftime('%d.%m.%Y'), (today - timedelta(days=2)).strftime('%d.%m.%Y')],
        'Поступления': [76422, 35795, 20860],
        'Списания': [63555, 32159, 50920],
        'Остаток': [200000, 203636, 173576]
    })
    
    # ОПТИМИЗАЦИЯ: Берем только первые 50 строк. Если подсунуть 200к операций, браузер повесится.
    # Защита от лагов при демонстрации.
    df_ops_trimmed = df_ops_display.head(50)
    
    st.dataframe(
        df_ops_trimmed,
        use_container_width=True,
        hide_index=True,
        column_config={
            # Текст выравнивается влево
            "Дата": st.column_config.TextColumn("Дата операции", width="small"),
            # Числа (благодаря типу int) Streamlit сам прижмет вправо
            "Поступления": st.column_config.NumberColumn("Поступления (₽)", format="%d"),
            "Списания": st.column_config.NumberColumn("Списания (₽)", format="%d"),
            "Остаток": st.column_config.NumberColumn("Остаток на конец дня (₽)", format="%d")
        }
    )
    
    st.caption("Отображены последние 50 операций.")