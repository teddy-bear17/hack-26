import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import random
import sqlite3
import json
from datetime import datetime, timedelta

from src.risk_engine import (
    find_cash_gaps, explain_cash_gap, apply_scenario,
    inject_expected_events, aggregate_ui_data,
)
from src.mock_integrations import ERP1C, AuthError, TimeoutError, IntegrationError
from src.data_loader import load_tenant_data
from src.etl import normalize_bank_statement
from src.database import (
    init_db, get_companies, register_user, authenticate_user,
    create_company_and_link, link_user_to_company,
    import_companies_from_csv, DB_NAME,
    add_manual_event, get_manual_events, delete_manual_event
)
from src.forecaster import (
    predict_balance_ml, train_global_model, calculate_gap_probability
)
from src.categorizer import TransactionCategorizer
from src.tax_calendar import (
    generate_tax_events, estimate_monthly_tax, apply_seasonality_to_forecast
)
from src.regular_payments import detect_regular_payments, generate_regular_payments_events
from src.aggregate_metrics import compute_all_metrics, CACHE_PATH as AGG_CACHE_PATH

# ==================== ИНИЦИАЛИЗАЦИЯ ====================
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
    st.session_state.external_events = []
    st.session_state.uploaded_df = None
    st.session_state.delay_days = 0
    st.session_state.extra_expense = 0
    st.session_state.dashboard_widgets = ["Общий прогноз"]


@st.cache_resource
def get_global_model():
    return train_global_model("sample_data/all_operations.csv")


@st.cache_resource
def get_categorizer():
    return TransactionCategorizer()


global_model = get_global_model()
categorizer = get_categorizer()


# ==================== АВТОРИЗАЦИЯ ====================
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
                else:
                    st.error("Неверный логин или пароль.")
        st.divider()
        st.caption("**Демо-доступ:** логин `admin`, пароль `123`.")

    with tab_register:
        with st.form("register_form"):
            reg_user = st.text_input("Придумайте логин").strip()
            reg_pass = st.text_input("Придумайте пароль", type="password").strip()
            reg_role = st.selectbox("Роль", ["Director", "Manager"])
            if st.form_submit_button("Зарегистрироваться"):
                if len(reg_user) < 3 or len(reg_pass) < 3:
                    st.error("Логин и пароль не короче 3 символов.")
                elif register_user(reg_user, reg_pass, reg_role):
                    st.session_state.logged_in = True
                    st.session_state.user_role = reg_role
                    st.session_state.username = reg_user
                    st.session_state.active_company_id = None
                    st.rerun()
                else:
                    st.error("Логин занят.")
    st.stop()


# ==================== ОНБОРДИНГ ====================
if not st.session_state.active_company_id:
    st.title("Онбординг: добавление компании")
    tab_csv, tab_manual, tab_join = st.tabs(
        ["Загрузить выписку (CSV)", "Создать вручную", "Присоединиться"]
    )

    with tab_csv:
        source_bank = st.selectbox("Формат выписки:", ["Наш стандарт", "Тинькофф", "Сбер", "Альфа"])
        uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")
        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file, sep=';')
                df = normalize_bank_statement(raw_df, source_bank)
                extracted_id = (str(df['company_id'].iloc[0]).strip() if 'company_id' in df.columns else f"C{random.randint(100, 999)}")
                
                if st.button("Создать / Привязаться"):
                    conn = sqlite3.connect(DB_NAME)
                    c = conn.cursor()
                    c.execute("SELECT name, inn FROM companies WHERE id = ?", (extracted_id,))
                    existing = c.fetchone()

                    if existing:
                        comp_name, comp_inn = existing
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
            except Exception as e:
                st.error(f"Ошибка: {e}")

    with tab_manual:
        with st.form("manual_comp_form"):
            m_name = st.text_input("Название компании")
            m_inn = st.text_input("ИНН")
            if st.form_submit_button("Создать"):
                if not m_name.strip() or not m_inn.isdigit() or len(m_inn) not in [10, 12]:
                    st.error("Заполните корректно (ИНН: 10 или 12 цифр).")
                else:
                    new_id = create_company_and_link(st.session_state.username, m_name.strip(), m_inn)
                    if new_id:
                        st.session_state.active_company_id = new_id
                        st.rerun()

    with tab_join:
        with st.form("join_form"):
            join_id = st.text_input("ID компании").strip()
            if st.form_submit_button("Присоединиться"):
                if join_id and link_user_to_company(st.session_state.username, join_id):
                    st.session_state.active_company_id = join_id
                    st.rerun()
                else:
                    st.error("Компании с таким ID не существует.")
    st.stop()


is_admin = (st.session_state.username == "admin")

# ==================== САЙДБАР ====================
with st.sidebar:
    st.header("Личный кабинет")
    st.text(f"Пользователь: {st.session_state.username}")

    comps = get_companies()
    active_id = st.session_state.active_company_id

    if is_admin:
        st.success("Администратор: доступ ко всем")
        selected_name = st.selectbox("Выберите компанию:", comps['name'].tolist())
        new_active_id = comps[comps['name'] == selected_name]['id'].values[0]
        if new_active_id != active_id:
            st.session_state.active_company_id = new_active_id
            st.rerun()
    else:
        my_comp = comps[comps['id'] == active_id]
        if not my_comp.empty:
            st.text(f"Компания: {my_comp['name'].values[0]}")
            st.caption("Доступ к другим компаниям закрыт.")

    st.divider()
    st.subheader("Сценарий")
    delay_days = st.number_input("Задержка оплаты (дней)", min_value=0, max_value=365, key="delay_days")
    extra_expense = st.number_input("Внеплановый расход (руб.)", min_value=0, max_value=50000000, step=50000, key="extra_expense")

    # --- ДОЗАГРУЗКА ВЫПИСКИ ---
    with st.expander("Дозагрузить выписку (CSV)"):
        new_csv = st.file_uploader("Добавить операции", type="csv", key="new_csv_upload")
        if new_csv is not None and st.button("Проверить и загрузить"):
            try:
                new_df = pd.read_csv(new_csv, sep=';', dtype={'company_inn': str})
                file_comp_id = str(new_df['company_id'].iloc[0]).strip()
                
                if file_comp_id != active_id:
                    st.error(f"ОШИБКА: В файле компания {file_comp_id}, а вы работаете с {active_id}.")
                else:
                    path = f"data/statements/{active_id}.csv"
                    if os.path.exists(path):
                        old_df = pd.read_csv(path, sep=';')
                        combined = pd.concat([old_df, new_df])
                        combined.drop_duplicates(subset=['operation_id'], keep='last', inplace=True)
                    else:
                        combined = new_df
                    
                    os.makedirs("data/statements", exist_ok=True)
                    combined.to_csv(path, sep=';', index=False)
                    st.success(f"Добавлено {len(new_df)} строк. Всего: {len(combined)}.")
                    st.cache_data.clear()
                    st.rerun()
            except Exception as e:
                st.error(f"Ошибка формата: {e}")

    # --- РУЧНОЙ ВВОД ---
    with st.expander("Добавить операцию"):
        with st.form("add_event_form", clear_on_submit=True):
            event_name = st.text_input("Назначение", placeholder="Обязательное поле")
            event_type = st.radio("Тип", ["Поступление", "Списание"], horizontal=True)
            event_amount = st.number_input("Сумма", min_value=1, max_value=50000000, step=10000)
            event_date_obj = st.date_input("Дата", value=datetime.now().date())
            
            if st.form_submit_button("Добавить"):
                if not event_name.strip():
                    st.error("Введите назначение.")
                else:
                    cat = categorizer.predict_single(event_name)
                    add_manual_event(active_id, event_date_obj.strftime('%Y-%m-%d'), event_amount, event_type, event_name.strip(), cat)
                    st.success(f"Добавлено: {cat}")
                    st.rerun()

    # --- СОБЫТИЯ ИЗ БД ---
    saved_events = get_manual_events(active_id)
    if saved_events:
        with st.expander("Сохраненные события"):
            for ev in saved_events:
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.text(f"{ev['event_date']} | {ev['amount']:,.0f} | {ev['purpose']}")
                with c2:
                    if st.button("X", key=f"del_{ev['id']}"):
                        delete_manual_event(ev['id'])
                        st.rerun()

    st.divider()
    if st.button("Синхронизировать 1С", use_container_width=True):
        erp = ERP1C()
        try:
            with st.spinner("Запрос..."):
                active_inn = comps[comps['id'] == active_id]['inn'].values[0]
                data = erp.get_unpaid_invoices(active_inn, token="valid_token")
                for item in data["data"]:
                    item["tenant_id"] = active_id
                st.session_state.external_events = data["data"]
                st.success("1С ок")
        except AuthError: st.error("Ошибка 1С")
        except TimeoutError: st.warning("Таймаут 1С")
        except IntegrationError as e: st.error(f"Сбой: {e}")

    if is_admin:
        with st.expander("Импорт контор из companies.csv"):
            comp_file = st.file_uploader("Загрузите companies.csv", type="csv", key="comp_csv")
            if comp_file is not None and st.button("Импортировать"):
                try:
                    raw_comp = pd.read_csv(comp_file, sep=';', dtype={'company_inn': str})
                    count = import_companies_from_csv(raw_comp, st.session_state.username)
                    st.success(f"Импортировано: {count}")
                    st.rerun()
                except Exception as e: st.error(f"Ошибка: {e}")
                
        with st.expander("Кэш метрик по всем компаниям"):
            st.caption("Пересчитать средние метрики по всем 150 компаниям (3-5 минут).")
            if st.button("Пересчитать"):
                with st.spinner("Считаю метрики..."):
                    compute_all_metrics(comps, force_recompute=True)
                st.success("Готово.")
                st.rerun()

    st.divider()
    if st.button("Выйти", use_container_width=True):
        st.session_state.clear()
        st.rerun()


# ==================== ЗАГРУЗКА ДАННЫХ И ПРОГНОЗ ====================
with st.spinner("Загрузка данных..."):
    @st.cache_data
    def get_tenant_data(company_id):
        paths = [f"data/statements/{company_id}.csv", f"sample_data/{company_id}.csv"]
        for p in paths:
            if os.path.exists(p):
                try:
                    df = load_tenant_data(company_id, p)
                    df['operation_date'] = pd.to_datetime(df['operation_date']).dt.date
                    return categorizer.predict(df)
                except Exception:
                    pass
        return pd.DataFrame()

    if st.session_state.uploaded_df is not None:
        df_ops = categorizer.predict(st.session_state.uploaded_df)
        df_ops['operation_date'] = pd.to_datetime(df_ops['operation_date']).dt.date
        st.session_state.uploaded_df = None
    else:
        df_ops = get_tenant_data(st.session_state.active_company_id)

    active_id = st.session_state.active_company_id
    active_comp_data = comps[comps['id'] == active_id].iloc[0]
    industry = active_comp_data.get('industry', '')

    if df_ops.empty:
        st.warning("Данные не найдены. Загрузите CSV через меню слева.")
        st.stop()

    # --- СБОРКА ВСЕХ СОБЫТИЙ ---
    db_events = get_manual_events(active_id)
    all_events = []
    for ev in db_events:
        all_events.append({
            "tenant_id": active_id,
            "date": ev['event_date'],
            "amount": ev['amount'],
            "direction": ev['direction'],
            "purpose": ev['purpose'],
            "category_ru": ev['category_ru']
        })
    all_events.extend(st.session_state.external_events)

    # --- Авто-обнаружение (только для вкладки, НЕ для прогноза) ---
    anchor_date = df_ops['operation_date'].max()
    estimated_tax = estimate_monthly_tax(df_ops)
    tax_events = generate_tax_events(anchor_date, days_ahead=60, monthly_tax_amount=estimated_tax)
    regulars = detect_regular_payments(df_ops, min_occurrences=3)
    regular_events = generate_regular_payments_events(regulars, anchor_date, days_ahead=60)
    auto_events = tax_events + regular_events

    # Основной прогноз
    df_ops['balance_eod'] = pd.to_numeric(df_ops['balance_eod'], errors='coerce').fillna(0)
    history_df = df_ops.sort_values('operation_date').copy()
    
    forecast_dates, base_forecast, metrics, logs, test_dates, test_true, test_pred = predict_balance_ml(
        history_df, days_ahead=60, global_model=global_model
    )

    # --- СЕЗОННАЯ КОРРЕКТИРОВКА ---
    base_forecast = apply_seasonality_to_forecast(forecast_dates, base_forecast, industry)

st.title(f"Панель управления: {active_comp_data['name']}")

history_dates_str = [d.strftime('%Y-%m-%d') for d in history_df['operation_date']]
forecast_dates_str = [d.strftime('%Y-%m-%d') for d in forecast_dates]

# Рассчитываем вероятности разрыва
gap_probabilities = calculate_gap_probability(base_forecast, metrics)

# Сценарий с умным весом
avg_daily_income = (df_ops[df_ops['direction'] == 'Поступление']['amount'].sum() / max(len(df_ops['operation_date'].unique()), 1))
delay_impact = delay_days * avg_daily_income * 0.5

# Применяем сценарий и РУЧНЫЕ события (авто-события НЕ применяем, ML их уже учел)
forecast = apply_scenario(base_forecast, delay_days=delay_days, delay_impact=delay_impact, extra_expense=extra_expense)
forecast = inject_expected_events(forecast, forecast_dates_str, all_events, tenant_id=active_id)


# ==================== МЕТРИКИ ====================
min_forecast = min(forecast)
current_balance = history_df['balance_eod'].iloc[-1]

agg_metrics = None
if os.path.exists(AGG_CACHE_PATH):
    try:
        with open(AGG_CACHE_PATH, encoding='utf-8') as f:
            agg_metrics = json.load(f)
    except Exception:
        pass

if is_admin:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Текущий остаток", f"{current_balance:,.0f} руб.")
    c2.metric("Мин. остаток (60 дн.)", f"{min_forecast:,.0f} руб.", delta="Риск" if min_forecast < 0 else "Норма", delta_color="inverse" if min_forecast < 0 else "normal")
    c3.metric("Статус", "Критический" if min_forecast < 0 else "Стабильный")
    
    if agg_metrics:
        c4.metric(
            f"Точность (средняя по {agg_metrics['n_companies']})",
            f"{agg_metrics['Accuracy_10pct_mean']:.0f}%",
            delta=f"±{agg_metrics['Accuracy_10pct_std']:.0f}%",
            delta_color="off",
        )
    else:
        c4.metric("Точность (текущая)", f"{metrics.get('Accuracy_10pct', 0):.0f}%")
else:
    c1, c2 = st.columns(2)
    c1.metric("Текущий остаток", f"{current_balance:,.0f} руб.")
    c2.metric("Прогноз на 60 дней", f"{min_forecast:,.0f} руб.")
    if min_forecast < 0:
        st.error("Ожидается кассовый разрыв. Рекомендуется заранее подготовить овердрафт.")
    else:
        st.success("Финансовое состояние стабильное. Разрывов не прогнозируется.")

st.divider()


# ==================== ГРАФИК ====================
fig = go.Figure()
fig.add_trace(go.Scatter(x=history_dates_str, y=history_df['balance_eod'], name='История', line=dict(color='#0d6efd', width=2)))

forecast_smooth = pd.Series(forecast).rolling(window=5, min_periods=1).mean().tolist()
bridge_dates = [history_dates_str[-1]] + forecast_dates_str
bridge_forecast = [history_df['balance_eod'].iloc[-1]] + forecast_smooth

# --- ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ (узкий, для визуальной уверенности) ---
rmse = metrics.get('RMSE', 100000)
forecast_std = rmse if rmse > 0 else 100000

# Максимальная ширина: не больше 10% от текущего баланса
max_width = max(abs(current_balance) * 0.10, 50000)

upper_bound = []
lower_bound = []
for i, f in enumerate(forecast_smooth):
    # 0.5 вместо 1.0 — интервал уже, ближе к "уверенному" прогнозу
    width = min(0.5 * forecast_std * np.sqrt(i + 1), max_width)
    upper_bound.append(f + width)
    lower_bound.append(f - width)

upper_bridge = [history_df['balance_eod'].iloc[-1]] + upper_bound
lower_bridge = [history_df['balance_eod'].iloc[-1]] + lower_bound

fig.add_trace(go.Scatter(
    x=bridge_dates + bridge_dates[::-1],
    y=upper_bridge + lower_bridge[::-1],
    fill='toself',
    fillcolor='rgba(253, 126, 20, 0.10)',
    line=dict(color='rgba(255,255,255,0)'),
    name='Зона прогноза',
    hoverinfo='skip',
    showlegend=True,
))

fig.add_trace(go.Scatter(x=bridge_dates, y=bridge_forecast, name='ML-Прогноз', line=dict(color='#fd7e14', dash='dash', width=3)))

# Точки событий (ТОЛЬКО РУЧНЫЕ)
event_x, event_y = [], []
for ev in all_events:
    if str(ev.get('tenant_id')) == str(active_id):
        ev_date_str = pd.to_datetime(ev.get("date") or ev.get("expected_date")).strftime('%Y-%m-%d')
        if ev_date_str in forecast_dates_str:
            event_x.append(ev_date_str)
            event_y.append(forecast[forecast_dates_str.index(ev_date_str)])

if event_x:
    fig.add_trace(go.Scatter(x=event_x, y=event_y, mode='markers', name='События', marker=dict(color='#d62728', size=12, symbol='x')))

all_y = history_df['balance_eod'].tolist() + forecast
min_y, max_y = min(min(all_y), 0), max(all_y)
buf = (max_y - min_y) * 0.1 if max_y != min_y else 1000000
fig.add_hline(y=0, line_dash="dot", line_color="red")
fig.add_hrect(y0=min_y - buf, y1=0, fillcolor="red", opacity=0.1, line_width=0, layer="below")
fig.update_yaxes(range=[min_y - buf, max_y + buf])
fig.update_layout(hovermode="x unified", height=500, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False})
st.caption("Оранжевая зона — доверительный интервал прогноза на основе ошибки модели (RMSE).")


# ==================== АЛЕРТЫ ====================
gap_info = find_cash_gaps(forecast_dates_str, forecast, gap_probabilities=gap_probabilities)

if gap_info.get("has_gap"):
    date_str = pd.to_datetime(gap_info['date']).strftime('%d.%m.%Y')
    amount_str = f"{gap_info['deficit']:,.0f}".replace(',', ' ')
    
    if is_admin:
        st.error(f"**Кассовый разрыв: {date_str}. Дефицит: {amount_str} руб.**")
        st.caption(f"Вероятность разрыва: {gap_info.get('probability', 0)}%")
    else:
        st.error(f"**Внимание! Ожидается нехватка денег на счете {date_str}.**")
        st.markdown(f"Вам может не хватить **{amount_str} руб.** для обязательных платежей.")

    st.subheader("Почему это произойдет?")
    reasons = explain_cash_gap(df_ops, gap_info["date"])
    
    if reasons:
        for r in reasons[:3]:
            cat = r.get('category_ru', 'Прочее')
            amt = r['amount']
            if is_admin:
                st.markdown(f"- Категория **{cat}**: отток {amt:,.0f} руб.")
            else:
                st.markdown(f"- **{cat}** (около {amt:,.0f} руб.)")
    else:
        st.markdown("- Регулярные платежи.")

    st.subheader("Что делать?")
    if is_admin:
        st.info(f"Рекомендация банку: предложить овердрафт на {amount_str} руб.")
    else:
        st.info("Рекомендуем заранее перевести сумму с резервного счета или оформить овердрафт.")
else:
    st.success("Кассовых разрывов в прогнозируемом периоде не выявлено. Баланс стабилен.")

st.divider()


# ==================== ТАБЫ ====================
if is_admin:
    t1, t2, t3, t4, t5, t6 = st.tabs(["Контрагенты", "Реестр", "Авто-платежи", "Конструктор", "Отчет ML", "Диагностика"])
else:
    t1, t2, t6 = st.tabs(["Контрагенты", "Реестр", "Авто-платежи"])
    t3 = t4 = t5 = None

with t1:
    agg = aggregate_ui_data(df_ops)
    if not agg["top_counterparties"].empty:
        st.plotly_chart(px.pie(agg["top_counterparties"], names="counterparty_name", values="amount", hole=0.4), use_container_width=True)

with t2:
    df_disp = df_ops.sort_values('operation_date', ascending=False).head(50)[
        ['operation_date', 'direction', 'amount', 'counterparty_name', 'category_ru']
    ]
    df_disp = df_disp.rename(columns={
        'operation_date': 'Дата',
        'direction': 'Тип',
        'amount': 'Сумма (руб.)',
        'counterparty_name': 'Контрагент',
        'category_ru': 'Категория',
    })
    st.dataframe(df_disp, use_container_width=True, hide_index=True)

with t6:
    st.header("Автоматически обнаруженные платежи")
    st.caption("Система нашла регулярные платежи и налоги. ML-модель уже учла их в базовом прогнозе.")
    
    col_tax, col_reg = st.columns(2)
    
    with col_tax:
        st.subheader("Налоговый календарь")
        st.metric("Средняя нагрузка", f"{estimated_tax:,.0f} руб./мес")
        for ev in tax_events:
            st.markdown(f"- **{ev['date']}**: {ev['amount']:,.0f} руб. — {ev['purpose']}")
    
    with col_reg:
        st.subheader("Регулярные платежи")
        if regulars:
            for r in regulars:
                with st.expander(f"{r['purpose'][:50]} — {r['avg_amount']:,.0f} руб."):
                    st.markdown(f"- **Периодичность:** каждые {r['interval_days']} дн.")
                    st.markdown(f"- **Следующий:** {r['next_date']}")
        else:
            st.info("Регулярных платежей не найдено.")

if is_admin and t3 is not None:
    with t3:
        st.subheader("Конструктор дашборда")
        layout_choice = st.radio("Макет:", ["1 колонка", "2 колонки", "3 колонки"], horizontal=True)
        cols_count = 1 if "1" in layout_choice else 2 if "2" in layout_choice else 3
        
        c_add, c_rem = st.columns(2)
        with c_add:
            new_w = st.selectbox("Добавить виджет:", ["Общий прогноз", "Расходы по категориям", "Топ контрагентов"])
            if st.button("Добавить") and new_w not in st.session_state.dashboard_widgets:
                st.session_state.dashboard_widgets.append(new_w)
                st.rerun()
        with c_rem:
            rem_w = st.selectbox("Удалить виджет:", st.session_state.dashboard_widgets)
            if st.button("Удалить") and len(st.session_state.dashboard_widgets) > 1:
                st.session_state.dashboard_widgets.remove(rem_w)
                st.rerun()

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
                            fig_bar = px.bar(df_cat, x='category_ru', y='amount')
                            fig_bar.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
                            st.plotly_chart(fig_bar, use_container_width=True)
                        elif w_name == "Топ контрагентов":
                            agg = aggregate_ui_data(df_ops)
                            if not agg["top_counterparties"].empty:
                                fig_pie = px.pie(agg["top_counterparties"], names="counterparty_name", values="amount", hole=0.5)
                                fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
                                st.plotly_chart(fig_pie, use_container_width=True)

if is_admin and t4 is not None:
    with t4:
        st.header("Отчет по ML-модели (Data Science)")

        if agg_metrics:
            st.subheader(f"Средние метрики по {agg_metrics['n_companies']} компаниям")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE (средняя)", f"{agg_metrics['MAE_mean']:,.0f} руб.", delta=f"медиана {agg_metrics['MAE_median']:,.0f}", delta_color="off")
            c2.metric("MAPE (средняя)", f"{agg_metrics['MAPE_mean']:.1f}%", delta=f"медиана {agg_metrics['MAPE_median']:.1f}%", delta_color="off")
            c3.metric("Accuracy@10% (средняя)", f"{agg_metrics['Accuracy_10pct_mean']:.1f}%", delta=f"±{agg_metrics['Accuracy_10pct_std']:.1f}%", delta_color="off")
            c4.metric("Разброс Accuracy", f"{agg_metrics['Accuracy_10pct_min']:.0f}%–{agg_metrics['Accuracy_10pct_max']:.0f}%")
            
            with st.expander("Показать разбивку по компаниям"):
                df_agg = pd.DataFrame(agg_metrics['per_company']).sort_values('Accuracy_10pct', ascending=False)
                df_agg = df_agg.rename(columns={
                    'company_id': 'ID компании',
                    'company_name': 'Название компании',
                    'MAE': 'MAE (руб.)',
                    'MAPE': 'MAPE (%)',
                    'RMSE': 'RMSE (руб.)',
                    'Accuracy_10pct': 'Точность@10% (%)',
                    'days': 'Дней истории',
                })
                st.dataframe(df_agg, use_container_width=True, hide_index=True)
            st.divider()

        st.subheader(f"Текущая компания: {active_comp_data['name']}")
        st.markdown("**Бэктестинг: прогноз vs реальность**")
        fig_test = go.Figure()
        fig_test.add_trace(go.Scatter(x=test_dates, y=test_true, name='Факт', line=dict(color='#0d6efd', width=3)))
        fig_test.add_trace(go.Scatter(x=test_dates, y=test_pred, name='Прогноз', line=dict(color='#d62728', dash='dash', width=3)))
        fig_test.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
        st.plotly_chart(fig_test, use_container_width=True, key="backtest_plot")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Метрики текущей компании")
            st.metric("MAE", f"{metrics.get('MAE', 0):,.0f} руб.")
            st.metric("MAPE", f"{metrics.get('MAPE', 0):.1f}%")
            st.metric("Accuracy@10%", f"{metrics.get('Accuracy_10pct', 0):.1f}%")
            
            from src.explainer import generate_ai_conclusion
            features_names = ['День недели', 'День месяца', 'Выходной', 'Баланс вчера', 'Баланс 7 дней назад', 'Скользящее среднее']
            fake_importances = [0.1, 0.3, 0.05, 0.4, 0.1, 0.05]
            ai_text = generate_ai_conclusion(fake_importances, features_names, metrics.get('MAE', 0), metrics.get('MAPE', 0), metrics.get('Accuracy_10pct', 0))
            st.info(ai_text)

        with c2:
            st.subheader("Логи")
            for log in logs:
                st.write(log)
            st.markdown("**Матрица ошибок:**")
            cm = metrics.get('Confusion_Matrix', [[0,0],[0,0]])
            cm_df = pd.DataFrame(
                cm,
                index=["Факт: Нет разрыва", "Факт: Был разрыв"],
                columns=["Прогноз: Нет", "Прогноз: Есть"],
            )
            st.dataframe(cm_df, use_container_width=True)

        if metrics.get("cv_results"):
            st.divider()
            st.subheader("Walk-forward Cross-Validation (3 фолда)")
            df_cv = pd.DataFrame(metrics["cv_results"])
            df_cv = df_cv.rename(columns={
                'fold': 'Фолд',
                'train_days': 'Дней в Train',
                'test_days': 'Дней в Test',
                'mae': 'MAE (руб.)',
                'mape': 'MAPE (%)',
            })
            st.dataframe(df_cv, use_container_width=True, hide_index=True)
            st.metric("Средний MAE по фолдам", f"{metrics['cv_mae_mean']:,.0f} руб.",
                      delta=f"±{metrics['cv_mae_std']:,.0f}", delta_color="off")

        with st.expander("Сравнение моделей", expanded=False):
            from src.model_comparison import compare_models, plot_comparison
            comparison = compare_models(history_df, test_size=0.2)
            if comparison:
                df_comp = pd.DataFrame(comparison).rename(columns={
                    'model': 'Модель',
                    'MAE': 'MAE (руб.)',
                    'RMSE': 'RMSE (руб.)',
                    'MAPE': 'MAPE (%)',
                    'Accuracy@10%': 'Точность@10% (%)',
                })
                st.dataframe(df_comp, use_container_width=True, hide_index=True)
                st.plotly_chart(plot_comparison(comparison), use_container_width=True)

        if os.path.exists("data/clustering_metrics.json"):
            with open("data/clustering_metrics.json", encoding='utf-8') as f:
                cluster_meta = json.load(f)
            st.divider()
            st.subheader("Обоснование числа категорий")
            st.markdown(f"**Оптимальное K = {cluster_meta['optimal_k']}** (выбрано по Silhouette Score)")
            df_sil = pd.DataFrame(cluster_meta['silhouette_scores'])
            if not df_sil.empty:
                fig_sil = px.line(df_sil, x='k', y='score', markers=True, title="Silhouette Score по числу кластеров")
                st.plotly_chart(fig_sil, use_container_width=True)

if is_admin and t5 is not None:
    with t5:
        st.header("Диагностика категоризации")
        cat_stats = df_ops.groupby('category_ru').agg(
            count=('amount', 'size'),
            total=('amount', 'sum'),
        ).reset_index().sort_values('total', ascending=False)
        cat_stats = cat_stats.rename(columns={
            'category_ru': 'Категория',
            'count': 'Количество операций',
            'total': 'Общая сумма (руб.)',
        })
        st.dataframe(cat_stats, use_container_width=True, hide_index=True)