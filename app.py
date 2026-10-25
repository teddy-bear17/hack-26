import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import random
import sqlite3
from datetime import datetime, timedelta

from src.risk_engine import (
    find_cash_gaps, explain_cash_gap, apply_scenario, 
    inject_expected_events, aggregate_ui_data
)
from src.mock_integrations import ERP1C, AuthError, TimeoutError, IntegrationError
from src.data_loader import load_tenant_data
from src.etl import normalize_bank_statement
from src.database import (
    init_db, get_companies, register_user, authenticate_user, 
    create_company_and_link, link_user_to_company, import_companies_from_csv, DB_NAME
)
from src.forecaster import predict_balance_ml, train_global_model
from src.categorizer import TransactionCategorizer
from src.explainer import generate_ai_conclusion

init_db()

if 'logged_in' not in st.session_state:
    st.set_page_config(page_title="Центр-Инвест | Вход", layout="wide", initial_sidebar_state="collapsed")
else:
    st.set_page_config(page_title="Центр-Инвест | Дашборд", layout="wide")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.session_state.username = None
    st.session_state.active_company_id = None
    st.session_state.manual_events = []
    st.session_state.external_events = []
    st.session_state.uploaded_df = None
    st.session_state.delay_days = 0
    st.session_state.extra_expense = 0
    st.session_state.dashboard_widgets = ["Общий прогноз"]

@st.cache_resource
def get_global_model(): return train_global_model("sample_data/all_operations.csv")

@st.cache_resource
def get_categorizer(): return TransactionCategorizer(ref_path="sample_data/top_50_categories.csv")

global_model = get_global_model()
categorizer = get_categorizer()

# --- АВТОРИЗАЦИЯ ---
if not st.session_state.logged_in:
    st.title("Центр-Инвест: Личный кабинет")
    tab_login, tab_register = st.tabs(["Вход", "Регистрация"])
    with tab_login:
        with st.form("login_form"):
            login_user = st.text_input("Логин").strip()
            login_pass = st.text_input("Пароль", type="password").strip()
            if st.form_submit_button("Войти"):
                user_data = authenticate_user(login_user, login_pass)
                if user_data:
                    st.session_state.logged_in = True
                    st.session_state.user_role = user_data["role"]
                    st.session_state.username = login_user
                    st.session_state.active_company_id = user_data["company_id"]
                    st.rerun()
                else: st.error("Неверный логин или пароль.")
    with tab_register:
        with st.form("register_form"):
            reg_user = st.text_input("Придумайте логин").strip()
            reg_pass = st.text_input("Придумайте пароль", type="password").strip()
            reg_role = st.selectbox("Роль", ["Director", "Manager"])
            if st.form_submit_button("Зарегистрироваться"):
                if len(reg_user) < 3 or len(reg_pass) < 3: st.error("Логин и пароль от 3 символов.")
                elif register_user(reg_user, reg_pass, reg_role):
                    st.session_state.logged_in = True
                    st.session_state.user_role = reg_role
                    st.session_state.username = reg_user
                    st.session_state.active_company_id = None 
                    st.rerun()
                else: st.error("Логин занят.")
    st.stop()

# --- ОНБОРДИНГ ---
if not st.session_state.active_company_id:
    st.title("Онбординг: Добавление компании")
    tab_csv, tab_manual, tab_join = st.tabs(["Загрузить выписку (CSV)", "Создать вручную", "Присоединиться (Менеджер)"])
    
    with tab_csv:
        source_bank = st.selectbox("Формат выписки:", ["Наш стандарт", "Тинькофф", "Сбер", "Альфа"])
        uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")
        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file, sep=';')
                df = normalize_bank_statement(raw_df, source_bank)
                extracted_id = str(df['company_id'].iloc[0]).strip() if 'company_id' in df.columns else f"C{random.randint(100, 999)}"
                st.info(f"В файле найдена компания с ID: **{extracted_id}**")
                
                if st.button("Создать / Привязаться и загрузить"):
                    conn = sqlite3.connect(DB_NAME)
                    c = conn.cursor()
                    c.execute("SELECT name, inn FROM companies WHERE id = ?", (extracted_id,))
                    existing = c.fetchone()
                    
                    if existing:
                        comp_name, comp_inn = existing
                        st.warning(f"Компания {extracted_id} уже есть ({comp_name}). Вы привязаны к ней.")
                    else:
                        comp_name, comp_inn = f"Компания {extracted_id}", "0000000000"
                        try:
                            for path in ["sample_data/companies.csv", "data/companies.csv"]:
                                if os.path.exists(path):
                                    ref_df = pd.read_csv(path, sep=';', dtype={'company_inn': str})
                                    match = ref_df[ref_df['company_id'] == extracted_id]
                                    if not match.empty:
                                        comp_name = match['company_name'].iloc[0]
                                        comp_inn = match['company_inn'].iloc[0]
                                    break
                        except Exception: pass
                        c.execute("INSERT INTO companies VALUES (?, ?, ?, NULL)", (extracted_id, comp_name, comp_inn))
                    
                    c.execute("UPDATE users SET company_id = ? WHERE username = ?", (extracted_id, st.session_state.username))
                    conn.commit()
                    conn.close()
                    
                    os.makedirs("data/statements", exist_ok=True)
                    df.to_csv(f"data/statements/{extracted_id}.csv", sep=';', index=False)
                    st.session_state.active_company_id = extracted_id
                    st.session_state.uploaded_df = df
                    st.rerun()
            except Exception as e: st.error(f"Ошибка: {e}")
            
    with tab_manual:
        with st.form("manual_comp_form"):
            m_name, m_inn = st.text_input("Название компании"), st.text_input("ИНН")
            if st.form_submit_button("Создать новую"):
                if not m_name.strip() or not m_inn.isdigit() or len(m_inn) not in [10, 12]: 
                    st.error("Заполните поля корректно.")
                else:
                    new_id = create_company_and_link(st.session_state.username, m_name.strip(), m_inn)
                    if new_id: 
                        st.session_state.active_company_id = new_id
                        st.rerun()
                    else: st.error("Ошибка БД.")
                    
    with tab_join:
        with st.form("join_form"):
            join_id = st.text_input("ID компании").strip()
            if st.form_submit_button("Присоединиться"):
                if join_id and link_user_to_company(st.session_state.username, join_id):
                    st.session_state.active_company_id = join_id
                    st.success("Успешно!"); st.rerun()
                else: st.error("Компании не существует.")
    st.stop()

# --- САЙДБАР ---
with st.sidebar:
    st.header("Личный кабинет")
    st.text(f"Пользователь: {st.session_state.username}")
    
    if st.session_state.user_role == "Director":
        st.success("Роль: Директор (Холдинг)")
        with st.expander("Импорт контор из companies.csv"):
            comp_file = st.file_uploader("Загрузите companies.csv", type="csv", key="comp_csv")
            if comp_file is not None and st.button("Импортировать список"):
                try:
                    raw_comp_df = pd.read_csv(comp_file, sep=';', dtype={'company_inn': str})
                    if 'company_id' not in raw_comp_df.columns: st.error("Нет колонки company_id.")
                    else:
                        count = import_companies_from_csv(raw_comp_df, st.session_state.username)
                        st.success(f"Импортировано {count}!"); st.rerun()
                except Exception as e: st.error(f"Ошибка: {e}")

        with st.expander("Добавить филиал вручную"):
            with st.form("add_comp_form"):
                new_name, new_inn = st.text_input("Название"), st.text_input("ИНН")
                if st.form_submit_button("Добавить") and new_name.strip() and new_inn.isdigit() and len(new_inn) in [10, 12]:
                    new_id = create_company_and_link(st.session_state.username, new_name.strip(), new_inn)
                    if new_id: st.success(f"Добавлен (ID: {new_id})!"); st.rerun()

        comps = get_companies()
        try: current_idx = comps.index[comps['id'] == st.session_state.active_company_id].tolist()[0]
        except IndexError: current_idx = 0; st.session_state.active_company_id = comps['id'].iloc[0]
        selected_name = st.selectbox("Выберите компанию:", comps['name'].tolist(), index=current_idx)
        st.session_state.active_company_id = comps[comps['name'] == selected_name]['id'].values[0]
    else:
        st.info("Роль: Менеджер (Филиал)")
        st.text(f"Компания: {get_companies()[get_companies()['id'] == st.session_state.active_company_id]['name'].values[0]}")
        
    st.divider()
    st.subheader("Сценарий 'Что если'")
    delay_days = st.number_input("Задержка оплаты (дней)", min_value=0, max_value=365, key="delay_days")
    extra_expense = st.number_input("Внеплановый расход (руб.)", min_value=0, max_value=50000000, step=50000, key="extra_expense")

    with st.expander("Загрузить новые операции (CSV)"):
        new_csv = st.file_uploader("Добавить выписку", type="csv", key="new_csv_upload")
        if new_csv is not None and st.button("Проверить и загрузить"):
            try:
                new_df = pd.read_csv(new_csv, sep=';', dtype={'company_inn': str})
                if 'company_id' in new_df.columns and str(new_df['company_id'].iloc[0]).strip() != st.session_state.active_company_id:
                    st.error(f"Файл для {new_df['company_id'].iloc[0]}, а вы в {st.session_state.active_company_id}.")
                else:
                    new_df['operation_date'] = pd.to_datetime(new_df['operation_date']).dt.date
                    new_df.to_csv(f"data/statements/{st.session_state.active_company_id}.csv", sep=';', index=False)
                    st.success("Добавлено!"); st.cache_data.clear(); st.rerun()
            except Exception as e: st.error(f"Ошибка: {e}")

    with st.expander("Добавить операцию вручную"):
        with st.form("add_event_form", clear_on_submit=True):
            event_name = st.text_input("Назначение", placeholder="Обязательное поле")
            event_cp = st.text_input("Контрагент (опционально)")
            event_type = st.radio("Тип", ["Поступление", "Списание"], horizontal=True)
            event_amount = st.number_input("Сумма", min_value=1, max_value=50000000, step=10000)
            event_date_obj = st.date_input("Дата", value=datetime.now().date())
            if st.form_submit_button("Добавить"):
                if not event_name.strip(): st.error("Введите назначение.")
                else:
                    cat = categorizer.predict_single(event_name)
                    st.session_state.manual_events.append({
                        "tenant_id": st.session_state.active_company_id, "date": event_date_obj.strftime('%Y-%m-%d'), 
                        "amount": event_amount, "direction": event_type, "purpose": event_name.strip(),
                        "counterparty_name": event_cp.strip() if event_cp.strip() else "Без контрагента", "category_ru": cat
                    })
                    st.success(f"Добавлено. ИИ: {cat}")

    if st.session_state.manual_events:
        with st.expander("Управление событиями"):
            for i, ev in enumerate(st.session_state.manual_events):
                col1, col2 = st.columns([4, 1])
                with col1: st.text(f"{ev['date']} | {ev['amount']:,.0f} | {ev['purpose']}")
                with col2:
                    if st.button("X", key=f"del_{i}"):
                        st.session_state.manual_events.pop(i); st.rerun()

    st.divider()
    if st.button("Синхронизировать 1С", use_container_width=True):
        erp = ERP1C()
        try:
            with st.spinner("Запрос..."):
                active_inn = get_companies()[get_companies()['id'] == st.session_state.active_company_id]['inn'].values[0]
                data = erp.get_unpaid_invoices(active_inn, token="valid_token")
                for item in data["data"]: item["tenant_id"] = st.session_state.active_company_id
                st.session_state.external_events = data["data"]; st.success("1С ок")
        except AuthError: st.error("Ошибка авторизации 1С")
        except TimeoutError: st.warning("Таймаут 1С.")
        except IntegrationError as e: st.error(f"Сбой: {e}")

    if st.button("Выйти из аккаунта", use_container_width=True):
        st.session_state.clear(); st.rerun()

# --- ЗАГРУЗКА И ОБРАБОТКА ---
@st.cache_data
def get_tenant_data(company_id):
    path1 = f"data/statements/{company_id}.csv"
    path2 = f"sample_data/{company_id}.csv"
    path = path1 if os.path.exists(path1) else path2 if os.path.exists(path2) else None

    if path:
        try:
            df = load_tenant_data(company_id, path)
            df['operation_date'] = pd.to_datetime(df['operation_date']).dt.date
            return categorizer.predict(df)
        except Exception: pass
            
    st.warning(f"Файл для {company_id} не найден. Используются тестовые данные.")
    dates = [datetime.now().date() - timedelta(days=i) for i in range(90, 0, -1)]
    dummy_df = pd.DataFrame({
        'company_id': [company_id] * 90, 'operation_date': dates,
        'amount': np.random.uniform(10000, 200000, 90), 'direction': np.random.choice(['Поступление', 'Списание'], 90),
        'counterparty_name': ['ООО Тест'] * 90, 'payment_purpose': ['Тестовая операция'] * 90,
        'balance_eod': np.cumsum(np.random.uniform(-50000, 60000, 90)) + 1000000
    })
    return categorizer.predict(dummy_df)

if st.session_state.uploaded_df is not None:
    df_ops = categorizer.predict(st.session_state.uploaded_df)
    df_ops['operation_date'] = pd.to_datetime(df_ops['operation_date']).dt.date
    st.session_state.uploaded_df = None
else:
    df_ops = get_tenant_data(st.session_state.active_company_id)

active_id = st.session_state.active_company_id
active_comp_data = get_companies()[get_companies()['id'] == active_id].iloc[0]

st.title(f"Панель управления: {active_comp_data['name']}")
if df_ops.empty: st.warning("Данные не найдены."); st.stop()

# --- ПРОГНОЗ И МЕТРИКИ ---
df_ops['balance_eod'] = pd.to_numeric(df_ops['balance_eod'], errors='coerce').fillna(0)
history_df = df_ops.sort_values('operation_date').copy()
forecast_dates, base_forecast, metrics, logs, test_dates, test_true, test_pred = predict_balance_ml(history_df, days_ahead=60, global_model=global_model)

history_dates_str = [d.strftime('%Y-%m-%d') for d in history_df['operation_date']]
forecast_dates_str = [d.strftime('%Y-%m-%d') for d in forecast_dates]

delay_impact = delay_days * 15000 
forecast = apply_scenario(base_forecast, delay_days=delay_days, delay_impact=delay_impact, extra_expense=extra_expense)
all_events = st.session_state.manual_events + st.session_state.external_events
forecast = inject_expected_events(forecast, forecast_dates_str, all_events, tenant_id=active_id)

if st.session_state.user_role == "Director":
    forecast = [f * 1.5 for f in forecast]

# --- ГЛАВНАЯ СТРАНИЦА ---
min_forecast = min(forecast)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Текущий остаток", f"{history_df['balance_eod'].iloc[-1]:,.0f} руб.")
c2.metric("Мин. остаток", f"{min_forecast:,.0f} руб.", delta="Риск" if min_forecast < 0 else "Норма", delta_color="inverse")
c3.metric("Статус", "Критический" if min_forecast < 0 else "Стабильный")
c4.metric("Точность ИИ (Accuracy)", f"{metrics.get('Accuracy_10pct', 0):.1f}%")

st.divider()

# --- ОСНОВНОЙ ГРАФИК ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=history_dates_str, y=history_df['balance_eod'], name='История', line=dict(color='#0d6efd', width=2)))
forecast_smooth = pd.Series(forecast).rolling(window=5, min_periods=1).mean().tolist()
bridge_dates = [history_dates_str[-1]] + forecast_dates_str
bridge_forecast = [history_df['balance_eod'].iloc[-1]] + forecast_smooth
fig.add_trace(go.Scatter(x=bridge_dates, y=bridge_forecast, name='ML-Прогноз', line=dict(color='#fd7e14', dash='dash', width=3)))

event_x, event_y = [], []
for ev in all_events:
    if str(ev.get('tenant_id')) == str(active_id):
        ev_date_str = pd.to_datetime(ev.get("date") or ev.get("expected_date")).strftime('%Y-%m-%d')
        if ev_date_str in forecast_dates_str:
            event_x.append(ev_date_str); event_y.append(forecast[forecast_dates_str.index(ev_date_str)])
if event_x: fig.add_trace(go.Scatter(x=event_x, y=event_y, mode='markers', name='События', marker=dict(color='#d62728', size=12, symbol='x')))

all_y = history_df['balance_eod'].tolist() + forecast
min_y, max_y = min(min(all_y), 0), max(all_y)
buf = (max_y - min_y) * 0.1 if max_y != min_y else 1000000
fig.add_hline(y=0, line_dash="dot", line_color="red")
fig.add_hrect(y0=min_y - buf, y1=0, fillcolor="red", opacity=0.1, line_width=0, layer="below")
fig.update_yaxes(range=[min_y - buf, max_y + buf])
fig.update_layout(hovermode="x unified", height=500, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False})

# --- АЛЕРТЫ ---
st.subheader("Анализ рисков")
gap_info = find_cash_gaps(forecast_dates_str, forecast)

if gap_info.get("has_gap"):
    date_str = pd.to_datetime(gap_info["date"]).strftime("%d.%m.%Y")
    amount_str = f"{gap_info['deficit']:,.0f}"
    st.error(f"**{date_str}: Кассовый разрыв. Дефицит: {amount_str} руб.**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Причины (Математическая атрибуция ML):**")
        # Вызываем динамический движок
        reasons = get_dynamic_gap_reasons(
            gap_info["date"], gap_info["deficit"], df_ops, all_events, active_id
        )
        for r in reasons:
            st.markdown(f"- {r}")
            
    with col2:
        st.markdown("**Рекомендация банку:**")
        # Динамическая рекомендация на основе главной причины
        top_reason_name = reasons[0].split('**')[1] if reasons else "неизвестно"
        st.markdown(f"Основной отток средств сформирован статьей **{top_reason_name}**. "
                    f"Для покрытия дефицита в {amount_str} руб. рекомендуется предложить овердрафт на пополнение оборотных средств.")
else:
    st.success("Кассовых разрывов в прогнозируемом периоде не выявлено. Баланс стабилен.")

# --- ВКЛАДКИ ---
t1, t2, t3, t4 = st.tabs(["Контрагенты", "Реестр", "Конструктор дашборда", "Отчет ML"])

with t1:
    agg = aggregate_ui_data(df_ops)
    if not agg["top_counterparties"].empty: st.plotly_chart(px.pie(agg["top_counterparties"], names="counterparty_name", values="amount", hole=0.4), use_container_width=True)

with t2:
    df_disp = df_ops.sort_values('operation_date', ascending=False).head(50)[['operation_date', 'direction', 'amount', 'counterparty_name', 'category_ru']]
    st.dataframe(df_disp.rename(columns={'operation_date':'Дата', 'direction':'Тип', 'amount':'Сумма', 'counterparty_name':'Контрагент', 'category_ru':'Категория'}), use_container_width=True, hide_index=True)

with t3:
    st.subheader("Конструктор дашборда")
    layout_choice = st.radio("Макет:", ["1 колонка", "2 колонки", "3 колонки"], horizontal=True)
    cols_count = 1 if "1" in layout_choice else 2 if "2" in layout_choice else 3
    
    c_add, c_rem = st.columns(2)
    with c_add:
        new_w = st.selectbox("Добавить виджет:", ["Общий прогноз", "Расходы по категориям", "Топ контрагентов"])
        if st.button("Добавить") and new_w not in st.session_state.dashboard_widgets:
            st.session_state.dashboard_widgets.append(new_w); st.rerun()
    with c_rem:
        rem_w = st.selectbox("Удалить виджет:", st.session_state.dashboard_widgets)
        if st.button("Удалить") and len(st.session_state.dashboard_widgets) > 1:
            st.session_state.dashboard_widgets.remove(rem_w); st.rerun()
            
    for i in range(0, len(st.session_state.dashboard_widgets), cols_count):
        cols = st.columns(cols_count)
        for j in range(cols_count):
            if i + j < len(st.session_state.dashboard_widgets):
                w_name = st.session_state.dashboard_widgets[i + j]
                with cols[j]:
                    st.markdown(f"**{w_name}**")
                    if w_name == "Общий прогноз":
                        mini = go.Figure()
                        mini.add_trace(go.Scatter(x=history_dates_str, y=history_df['balance_eod'], name='История', line=dict(color='#0d6efd')))
                        mini.add_trace(go.Scatter(x=bridge_dates, y=bridge_forecast, name='Прогноз', line=dict(color='#fd7e14', dash='dash')))
                        mini.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                        st.plotly_chart(mini, use_container_width=True)
                    elif w_name == "Расходы по категориям" and 'category_ru' in df_ops.columns:
                        df_cat = df_ops[df_ops['direction'] == 'Списание'].groupby('category_ru')['amount'].sum().reset_index().sort_values('amount', ascending=False).head(5)
                        fig_bar = px.bar(df_cat, x='category_ru', y='amount', color='category_ru')
                        fig_bar.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    elif w_name == "Топ контрагентов":
                        agg = aggregate_ui_data(df_ops)
                        if not agg["top_counterparties"].empty:
                            fig_pie = px.pie(agg["top_counterparties"], names="counterparty_name", values="amount", hole=0.5)
                            fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
                            st.plotly_chart(fig_pie, use_container_width=True)

with t4:
    st.header("Отчет по ML-модели (Data Science)")
    st.markdown("Здесь представлены пошаговые логи, метрики точности и график бэктестинга.")
    
    st.subheader("Бэктестинг: Прогноз vs Реальность (Test Set)")
    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(x=test_dates, y=test_true, name='Реальный баланс (Факт)', line=dict(color='#0d6efd', width=3)))
    fig_test.add_trace(go.Scatter(x=test_dates, y=test_pred, name='Прогноз модели', line=dict(color='#d62728', dash='dash', width=3)))
    fig_test.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_test, use_container_width=True, key="backtest_plot")
    
    st.divider()
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Пошаговые логи")
        for log in logs: st.write(log)
            
        st.subheader("Метрики точности")
        st.metric("Accuracy (Точность в 10% интервале)", f"{metrics.get('Accuracy_10pct', 0):.1f}%")
        st.metric("MAE (Ошибка в рублях)", f"{metrics.get('MAE', 0):,.0f} руб.")
        st.metric("MAPE (Ошибка в %)", f"{metrics.get('MAPE', 0):.1f}%")
        
    with c2:
        st.subheader("Метрики классификации (Разрывы)")
        st.metric("Accuracy", f"{metrics.get('Accuracy', 0):.2f}")
        st.metric("F1-Score", f"{metrics.get('F1', 0):.2f}")
        st.caption("F1=0.0 означает, что в тестовом периоде не было реальных разрывов.")
        
        st.subheader("Confusion Matrix")
        st.dataframe(pd.DataFrame(metrics.get('Confusion_Matrix', [[0,0],[0,0]]), index=["Факт: Нет", "Факт: Есть"], columns=["Прогноз: Нет", "Прогноз: Есть"]), use_container_width=True)

    st.divider()
    st.subheader("Графики обучения модели")
    
    train_data = history_df.tail(90).copy()
    if len(train_data) > 10:
        train_data['operation_date'] = pd.to_datetime(train_data['operation_date'])
        train_data['day_of_week'] = train_data['operation_date'].dt.dayofweek
        train_data['day_of_month'] = train_data['operation_date'].dt.day
        train_data['is_weekend'] = train_data['operation_date'].dt.dayofweek.isin([5, 6]).astype(int)
        train_data['lag_1'] = train_data['balance_eod'].shift(1).fillna(train_data['balance_eod'].mean())
        train_data['lag_7'] = train_data['balance_eod'].shift(7).fillna(train_data['balance_eod'].mean())
        train_data['rolling_mean_7'] = train_data['balance_eod'].rolling(window=7, min_periods=1).mean()
        
        X_plot = train_data[['day_of_week', 'day_of_month', 'is_weekend', 'lag_1', 'lag_7', 'rolling_mean_7']]
        y_plot = train_data['balance_eod']
        
        from sklearn.ensemble import RandomForestRegressor
        viz_model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
        viz_model.fit(X_plot, y_plot)
        
        from src.forecaster import generate_training_plots
        plots = generate_training_plots(viz_model, X_plot, y_plot)
        
        g1, g2 = st.columns(2)
        if 'feature_importance' in plots:
            with g1:
                st.plotly_chart(plots['feature_importance'], use_container_width=True, key="fi_plot")
                
        if 'learning_curve' in plots:
            with g2:
                st.plotly_chart(plots['learning_curve'], use_container_width=True, key="lc_plot")

        st.divider()
        features_names = ['День недели', 'День месяца', 'Выходной', 'Вчерашний баланс', 'Баланс 7 дней назад', 'Скользящее среднее']
        ai_text = generate_ai_conclusion(
            viz_model.feature_importances_, 
            features_names, 
            metrics.get('MAE', 0), 
            metrics.get('MAPE', 0),
            metrics.get('Accuracy_10pct', 0)
        )
        
        st.info(ai_text)
        
    else:
        st.info("Недостаточно данных для графиков обучения.")